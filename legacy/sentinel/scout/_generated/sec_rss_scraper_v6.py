"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC's 8-K and 10-Q RSS feeds, extracts filing metadata
(ticker, CIK, filing date, form type, accession number), and normalizes
into dataclass objects for downstream linguistic and historical analysis.

Used by sentinel/pipeline.py to ingest regulatory filings in real-time.
"""

import os
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional
import requests


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from RSS feed."""
    ticker: str
    cik: str
    accession_number: str
    form_type: str
    filing_date: str
    company_name: str
    feed_url: str
    rss_title: str


class SECRSSFeedError(Exception):
    """Raised when SEC RSS feed fetch or parse fails."""
    pass


def fetch_sec_rss_feed(feed_url: str) -> str:
    """Fetch raw XML content from SEC EDGAR RSS feed with timeout and retries."""
    headers = {
        "User-Agent": "Sentinel-Sentiment-Engine (sentinel@local)"
    }
    max_retries = 2
    timeout_sec = 10

    for attempt in range(max_retries):
        try:
            resp = requests.get(feed_url, headers=headers, timeout=timeout_sec)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise SECRSSFeedError(f"Failed to fetch {feed_url} after {max_retries} attempts: {e}") from e


def parse_sec_rss_feed(xml_content: str, feed_url: str) -> list[SECFiling]:
    """Parse SEC RSS XML and extract filings into normalized dataclass list."""
    filings = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise SECRSSFeedError(f"Invalid XML in feed {feed_url}: {e}") from e

    # Define namespace (SEC uses default RSS + custom elements)
    ns = {
        "content": "http://purl.org/rss/1.0/modules/content/",
    }

    for item in root.findall(".//item"):
        try:
            title_elem = item.find("title")
            title = title_elem.text if title_elem is not None else ""

            link_elem = item.find("link")
            link = link_elem.text if link_elem is not None else ""

            pub_date_elem = item.find("pubDate")
            pub_date = pub_date_elem.text if pub_date_elem is not None else ""

            # Extract ticker and form type from title (format: "TICKER Form 8-K ...")
            ticker, form_type, cik, accession_num = _extract_filing_metadata(title, link)
            if not ticker:
                continue

            # Parse company name from title (typically precedes form type)
            company_name = _extract_company_name(title)

            filing = SECFiling(
                ticker=ticker,
                cik=cik,
                accession_number=accession_num,
                form_type=form_type,
                filing_date=pub_date,
                company_name=company_name,
                feed_url=feed_url,
                rss_title=title
            )
            filings.append(filing)
        except Exception as e:
            # Log parse errors but continue with next item
            pass

    return filings


def _extract_filing_metadata(title: str, link: str) -> tuple[str, str, str, str]:
    """Extract ticker, form type, CIK, and accession number from RSS title and link."""
    ticker = ""
    form_type = ""
    cik = ""
    accession_num = ""

    # Title format: "AAPL 8-K (0000320193) 2024-01-15"
    parts = title.split()
    if parts:
        ticker = parts[0].upper()
        if len(parts) > 1:
            form_type = parts[1]

    # Extract CIK from parentheses in title
    if "(" in title and ")" in title:
        start = title.rfind("(") + 1
        end = title.rfind(")")
        cik = title[start:end].strip()

    # Extract accession number from link (format: .../cgi-bin/browse-edgar?action=getcompany&CIK=...&type=8-K&dateb=...&owner=exclude&count=100&search_text=)
    # Alternatively from link structure: .../Archives/edgar/container/accession_number/...
    if "/Archives/edgar/" in link:
        parts_link = link.split("/Archives/edgar/")
        if len(parts_link) > 1:
            path_parts = parts_link[1].split("/")
            # Accession format: 0000320193-24-000001
            for part in path_parts:
                if "-" in part and len(part) > 10:
                    accession_num = part
                    break

    return ticker, form_type, cik, accession_num


def _extract_company_name(title: str) -> str:
    """Extract company name from RSS title (portion before form type)."""
    # Typical format: "AAPL 8-K (0000320193) 2024-01-15"
    # Company name might be multi-word, so extract intelligently
    parts = title.split()
    if len(parts) >= 2:
        # First part is ticker, second part is form type or company identifier
        # For now, return empty as SEC RSS typically uses ticker, not full name
        return ""
    return ""


def fetch_and_parse_8k_feed() -> list[SECFiling]:
    """Fetch SEC 8-K RSS feed and return parsed filings."""
    feed_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&output=xml"
    xml_content = fetch_sec_rss_feed(feed_url)
    return parse_sec_rss_feed(xml_content, feed_url)


def fetch_and_parse_10q_feed() -> list[SECFiling]:
    """Fetch SEC 10-Q RSS feed and return parsed filings."""
    feed_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&output=xml"
    xml_content = fetch_sec_rss_feed(feed_url)
    return parse_sec_rss_feed(xml_content, feed_url)


def save_filings_to_sqlite(filings: list[SECFiling], db_path: str = "sentinel/data/sec_filings.db") -> int:
    """Persist filings to SQLite; return count inserted."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            accession_number TEXT UNIQUE NOT NULL,
            form_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            company_name TEXT,
            feed_url TEXT,
            rss_title TEXT,
            inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    inserted = 0
    for filing in filings:
        try:
            cursor.execute("""
                INSERT INTO sec_filings (ticker, cik, accession_number, form_type, filing_date
