"""
SEC EDGAR RSS Feed Scraper — Sentinel Scout Module

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts structured
metadata (CIK, filing type, accession number, filing date, company name),
and normalizes into dataclass instances for downstream Linguist and Historian
analysis. Handles feed parsing, deduplication, and graceful degradation on
network failures.

Used by: sentinel/pipeline.py (Scout phase) → sentinel/historian/ (RAG indexing)
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class SECFiling:
    """Normalized SEC EDGAR filing metadata."""
    cik: str
    company_name: str
    filing_type: str
    accession_number: str
    filing_date: str
    submission_url: str
    fetched_at: str


def _get_robust_session(max_retries: int = 3, timeout_sec: float = 10.0) -> requests.Session:
    """Create requests.Session with exponential backoff retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.timeout = timeout_sec
    return session


def fetch_sec_rss_feeds(
    filing_types: Optional[List[str]] = None,
    hours_back: int = 24,
) -> List[SECFiling]:
    """
    Fetch and parse SEC EDGAR RSS feeds for specified filing types.

    Args:
        filing_types: List of filing type codes (e.g., ["8-K", "10-Q"]).
                      Defaults to ["8-K", "10-Q"].
        hours_back: Only return filings submitted within the last N hours.

    Returns:
        List of SECFiling dataclass instances normalized from RSS entries.
    """
    if filing_types is None:
        filing_types = ["8-K", "10-Q"]

    filings: List[SECFiling] = []
    session = _get_robust_session()

    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

    for filing_type in filing_types:
        # SEC EDGAR RSS feed endpoint for each filing type
        feed_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type={filing_type}&dateb=&owner=exclude&count=100&search_text=&myHID=&newSearch=true&fromdate=&todate=&start=0&output=atom"

        try:
            resp = session.get(feed_url)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            # Parse Atom feed namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "sec": "http://www.sec.gov/cgi-bin"
            }

            for entry in root.findall("atom:entry", ns):
                try:
                    # Extract metadata from Atom entry
                    title = entry.findtext("atom:title", "", ns) or ""
                    accession_number = entry.findtext("atom:id", "", ns).split("/")[-1] if entry.findtext("atom:id", "", ns) else ""
                    filing_date_str = entry.findtext("atom:updated", "", ns) or ""

                    # Parse filing date
                    try:
                        filing_date = datetime.fromisoformat(filing_date_str.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        filing_date = datetime.utcnow()

                    # Skip if outside time window
                    if filing_date < cutoff_time:
                        continue

                    # Extract CIK and company name from title or link
                    link = entry.findtext("atom:link[@rel='alternate']/@href", "", ns)
                    if not link:
                        for link_elem in entry.findall("atom:link", ns):
                            if link_elem.get("rel") == "alternate":
                                link = link_elem.get("href", "")
                                break

                    # Parse title to extract company name and CIK
                    # Title format: "Company Name (CIK: 0000000000)"
                    cik = ""
                    company_name = ""
                    if "(" in title and ")" in title:
                        parts = title.rsplit("(", 1)
                        company_name = parts[0].strip()
                        cik_part = parts[1].rstrip(")").replace("CIK:", "").strip()
                        cik = cik_part.split()[-1] if cik_part else ""

                    if not cik or not company_name:
                        continue

                    filing = SECFiling(
                        cik=cik,
                        company_name=company_name,
                        filing_type=filing_type,
                        accession_number=accession_number,
                        filing_date=filing_date_str,
                        submission_url=link,
                        fetched_at=datetime.utcnow().isoformat(),
                    )
                    filings.append(filing)

                except (AttributeError, ValueError, IndexError) as e:
                    # Skip malformed entries
                    continue

        except requests.RequestException as e:
            print(f"Warning: Failed to fetch SEC RSS for {filing_type}: {e}")
            continue

    return filings


def deduplicate_filings(filings: List[SECFiling]) -> List[SECFiling]:
    """Remove duplicate filings by accession number, keeping most recent."""
    seen = {}
    for filing in sorted(filings, key=lambda f: f.filing_date, reverse=True):
        if filing.accession_number not in seen:
            seen[filing.accession_number] = filing
    return list(seen.values())


def store_filings_local(
    filings: List[SECFiling],
    db_path: str = "sentinel_sec_filings.db",
) -> int:
    """
    Persist filings to local SQLite database for deduplication and history.

    Args:
        filings: List of SECFiling instances to store.
        db_path: Path to SQLite database file.

    Returns:
        Number of new filings inserted (excluding duplicates by accession_number).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            accession_number TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            company_name TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            submission_url TEXT,
            fetched_at TEXT NOT NULL
        )
    """)

    inserted = 0
    for filing in filings:
        try:
            cursor.execute("""
                INSERT INTO sec_filings
                (accession_number, cik, company_name, filing_type, filing_date, submission_url, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filing.accession_number,
                filing.cik,
                filing.company_name,
                filing.filing_type,
                filing.filing_date,
                filing.submission_url,
                filing.fetched_at,
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            # Accession number already exists
            pass

    conn.commit()
    conn.close()
    return inserted


def
