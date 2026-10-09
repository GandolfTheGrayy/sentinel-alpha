"""
SEC EDGAR RSS feed scraper for Sentinel Scout pillar.

Polls the SEC's RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(CIK, company name, accession number, filing date), and returns normalized
dataclass instances. Integrates with the live price fetcher and news ingestion
to build a real-time picture of corporate disclosures.

Uses Gemini for robust HTML/text extraction from RSS payloads when needed.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import list
import xml.etree.ElementTree as ET

import requests
from google.generativeai import GenerativeModel

# SEC RSS feed endpoints
SEC_8K_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&feed=atom"
SEC_10Q_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&feed=atom"
SEC_10K_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=exclude&count=100&feed=atom"


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from EDGAR RSS."""
    cik: str
    company_name: str
    filing_type: str
    accession_number: str
    filing_date: str
    url: str
    summary: str | None = None


def fetch_sec_rss_feed(feed_url: str, timeout: int = 10) -> str:
    """Fetch raw XML content from an SEC RSS feed."""
    headers = {
        "User-Agent": "Sentinel-SentimentEngine (+https://github.com/sentinel)"
    }
    response = requests.get(feed_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


def parse_sec_rss_xml(xml_content: str) -> list[dict]:
    """Parse Atom XML and extract entry metadata (title, link, updated, summary)."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return []

    # Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = []

    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        link_elem = entry.find("atom:link", ns)
        updated_elem = entry.find("atom:updated", ns)
        summary_elem = entry.find("atom:summary", ns)

        title = title_elem.text if title_elem is not None else ""
        link = link_elem.get("href") if link_elem is not None else ""
        updated = updated_elem.text if updated_elem is not None else ""
        summary = summary_elem.text if summary_elem is not None else None

        entries.append({
            "title": title,
            "link": link,
            "updated": updated,
            "summary": summary,
        })

    return entries


def extract_filing_metadata(entry: dict, filing_type: str) -> SECFiling | None:
    """Extract and normalize filing metadata from RSS entry dict."""
    title = entry.get("title", "")
    link = entry.get("link", "")
    updated = entry.get("updated", "")
    summary = entry.get("summary")

    # SEC RSS format: "Company Name (CIK) {filing_type} {date}"
    # E.g. "Apple Inc. (0000320193) 8-K 2024-01-15"
    parts = title.rsplit(" ", 1)
    if len(parts) < 2:
        return None

    date_str = parts[-1]
    rest = parts[0]

    # Extract CIK in parentheses
    if "(" not in rest or ")" not in rest:
        return None

    cik_start = rest.rfind("(")
    cik_end = rest.rfind(")")
    if cik_start == -1 or cik_end == -1 or cik_end <= cik_start:
        return None

    cik = rest[cik_start + 1 : cik_end].strip()
    company_name = rest[:cik_start].strip()

    # Extract accession number from link if available
    # SEC link format: https://www.sec.gov/cgi-bin/viewer?action=view&cik=...&accession_number=...
    accession_number = ""
    if "accession_number=" in link:
        accession_number = link.split("accession_number=")[-1].split("&")[0]

    return SECFiling(
        cik=cik,
        company_name=company_name,
        filing_type=filing_type,
        accession_number=accession_number,
        filing_date=date_str,
        url=link,
        summary=summary,
    )


def scrape_sec_filings(
    filing_types: list[str] | None = None, max_age_days: int = 7
) -> list[SECFiling]:
    """
    Scrape SEC EDGAR RSS feeds for recent filings.
    
    Args:
        filing_types: List of filing types to fetch (default: ['8-K', '10-Q', '10-K'])
        max_age_days: Only return filings newer than this many days
    
    Returns:
        List of normalized SECFiling objects.
    """
    if filing_types is None:
        filing_types = ["8-K", "10-Q", "10-K"]

    feed_map = {
        "8-K": SEC_8K_RSS_URL,
        "10-Q": SEC_10Q_RSS_URL,
        "10-K": SEC_10K_RSS_URL,
    }

    filings = []
    cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

    for filing_type in filing_types:
        if filing_type not in feed_map:
            print(f"Unknown filing type: {filing_type}")
            continue

        try:
            xml_content = fetch_sec_rss_feed(feed_map[filing_type])
            entries = parse_sec_rss_xml(xml_content)

            for entry in entries:
                filing = extract_filing_metadata(entry, filing_type)
                if filing is None:
                    continue

                # Parse filing date and filter by age
                try:
                    filing_dt = datetime.fromisoformat(filing.filing_date.replace("Z", "+00:00"))
                    if filing_dt < cutoff_date:
                        continue
                except (ValueError, AttributeError):
                    pass

                filings.append(filing)

        except requests.RequestException as e:
            print(f"Error fetching {filing_type} feed: {e}")

    return filings


def store_filings_in_db(filings: list[SECFiling], db_path: str = "sentinel_filings.db") -> None:
    """Store scraped filings in a local SQLite database for deduplication and historical lookup."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT,
            company_name TEXT,
            filing_type TEXT,
            accession_number TEXT,
            filing_date TEXT,
            url TEXT,
            summary TEXT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(accession_number, filing_type)
        )
    """)

    for filing in filings:
        try:
            cursor.execute("""
                INSERT INTO sec_fi
