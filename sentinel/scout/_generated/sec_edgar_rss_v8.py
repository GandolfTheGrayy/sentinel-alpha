"""
SEC EDGAR RSS feed scraper for Sentinel Scout pillar.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing
metadata (CIK, accession number, company name, filing date, form type),
and normalizes into dataclasses for downstream Linguist analysis.

Uses requests + xml.etree for lightweight feed parsing. Caches recent
accession numbers to avoid re-processing duplicates within a 24-hour window.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import list
import xml.etree.ElementTree as ET

import requests


EDGAR_RSS_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
EDGAR_8K_FEED = f"{EDGAR_RSS_BASE}?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&format=rss"
EDGAR_10Q_FEED = f"{EDGAR_RSS_BASE}?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&format=rss"

# Local cache DB to track seen accessions and avoid duplicates
CACHE_DB_PATH = os.path.expanduser("~/.sentinel/edgar_cache.db")


@dataclass
class SECFiling:
    """Normalized SEC EDGAR filing metadata."""
    cik: str
    accession_number: str
    company_name: str
    form_type: str
    filing_date: str
    filing_url: str
    raw_title: str


def _init_cache_db() -> None:
    """Initialize the accession cache database if it doesn't exist."""
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS edgar_accessions (
            accession_number TEXT PRIMARY KEY,
            fetched_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _is_cached(accession_number: str) -> bool:
    """Check if an accession number was fetched in the last 24 hours."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT fetched_at FROM edgar_accessions WHERE accession_number = ?",
        (accession_number,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        return False

    fetched_at = datetime.fromisoformat(row[0])
    return datetime.utcnow() - fetched_at < timedelta(hours=24)


def _mark_cached(accession_number: str) -> None:
    """Record an accession number as fetched."""
    conn = sqlite3.connect(CACHE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO edgar_accessions (accession_number, fetched_at) VALUES (?, ?)",
        (accession_number, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def _parse_rss_feed(feed_url: str) -> list[SECFiling]:
    """Fetch and parse an SEC EDGAR RSS feed, returning list of SECFiling objects."""
    try:
        response = requests.get(feed_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {feed_url}: {e}")
        return []

    filings = []
    try:
        root = ET.fromstring(response.content)
        # SEC RSS uses standard RSS item structure
        ns = {"": "http://www.sec.gov/cgi-bin/browse-edgar"}
        items = root.findall(".//item")

        for item in items:
            # Extract title: "Company Name 8-K/10-Q filing"
            title_elem = item.find("title")
            title = title_elem.text if title_elem is not None else ""

            # Extract link and derive CIK + accession
            link_elem = item.find("link")
            link = link_elem.text if link_elem is not None else ""

            # Extract description for form type
            desc_elem = item.find("description")
            description = desc_elem.text if desc_elem is not None else ""

            # Extract pub date
            pubdate_elem = item.find("pubDate")
            pub_date = pubdate_elem.text if pubdate_elem is not None else ""

            # Parse link: https://www.sec.gov/cgi-bin/viewer?action=view&cik=...&accession_number=...
            accession_number = _extract_accession_from_link(link)
            if not accession_number or _is_cached(accession_number):
                continue

            cik = _extract_cik_from_link(link)
            form_type = _extract_form_type_from_title(title)
            company_name = _extract_company_name_from_title(title)

            filing = SECFiling(
                cik=cik,
                accession_number=accession_number,
                company_name=company_name,
                form_type=form_type,
                filing_date=pub_date,
                filing_url=link,
                raw_title=title,
            )
            filings.append(filing)
            _mark_cached(accession_number)

    except ET.ParseError as e:
        print(f"Error parsing RSS XML: {e}")
        return []

    return filings


def _extract_cik_from_link(link: str) -> str:
    """Extract CIK from SEC EDGAR filing link."""
    if "cik=" in link:
        parts = link.split("cik=")
        if len(parts) > 1:
            cik = parts[1].split("&")[0]
            return cik.lstrip("0") or "0"
    return ""


def _extract_accession_from_link(link: str) -> str:
    """Extract accession number from SEC EDGAR filing link."""
    if "accession_number=" in link:
        parts = link.split("accession_number=")
        if len(parts) > 1:
            accession = parts[1].split("&")[0]
            return accession.replace("-", "")
    return ""


def _extract_form_type_from_title(title: str) -> str:
    """Extract form type (8-K, 10-Q, etc.) from RSS title."""
    if "8-K" in title:
        return "8-K"
    elif "10-Q" in title:
        return "10-Q"
    elif "10-K" in title:
        return "10-K"
    return "UNKNOWN"


def _extract_company_name_from_title(title: str) -> str:
    """Extract company name from RSS title (format: 'Company Name 8-K/10-Q')."""
    # Title format is typically "Company Name 8-K" or similar
    for form in ["8-K", "10-Q", "10-K", "8-K/A"]:
        if form in title:
            return title.split(form)[0].strip()
    return title.strip()


def fetch_8k_filings() -> list[SECFiling]:
    """Fetch recent 8-K filings from SEC EDGAR RSS feed."""
    _init_cache_db()
    return _parse_rss_feed(EDGAR_8K_FEED)


def fetch_10q_filings() -> list[SECFiling]:
    """Fetch recent 10-Q filings from SEC EDGAR RSS feed."""
    _init_cache_db()
    return _parse_rss_feed(EDGAR_10Q_FEED)


def fetch_all_recent_filings() -> list[SECFiling]:
    """Fetch both 8-K and 10-Q filings, deduplicated by
