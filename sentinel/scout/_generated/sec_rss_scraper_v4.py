"""SEC EDGAR RSS feed scraper for Sentinel Scout pillar.

Polls the SEC EDGAR RSS feeds (8-K and 10-Q) to extract filing metadata
(accession number, company CIK, filing date, document URL) into normalized
dataclasses. Feeds are filtered by ticker symbol via CIK lookup. Output is
cached in SQLite to avoid re-ingestion and passed to Linguist for sentiment
analysis of filing text.

This module uses Gemini for high-volume RSS/XML parsing, never Claude.
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List
import xml.etree.ElementTree as ET

import requests
import google.generativeai as genai

# Configure Gemini for text extraction tasks
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# SEC EDGAR RSS feed URLs
SEC_FEEDS = {
    "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&search_text=&FromDate=&ToDate=&output=atom",
    "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&search_text=&FromDate=&ToDate=&output=atom",
}

DB_PATH = "sentinel_filings.db"


@dataclass
class FilingMetadata:
    """Normalized SEC filing metadata extracted from RSS feeds."""
    accession_number: str
    cik: str
    ticker: Optional[str]
    company_name: str
    filing_type: str
    filing_date: str
    submission_date: str
    document_url: str
    ingested_at: str


def init_db() -> None:
    """Initialize SQLite cache for filing metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            accession_number TEXT PRIMARY KEY,
            cik TEXT NOT NULL,
            ticker TEXT,
            company_name TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            submission_date TEXT NOT NULL,
            document_url TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def filing_exists(accession_number: str) -> bool:
    """Check if filing already cached in SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM filings WHERE accession_number = ?",
        (accession_number,),
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def save_filing(filing: FilingMetadata) -> None:
    """Persist filing metadata to SQLite cache."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT OR IGNORE INTO filings
           (accession_number, cik, ticker, company_name, filing_type,
            filing_date, submission_date, document_url, ingested_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            filing.accession_number,
            filing.cik,
            filing.ticker,
            filing.company_name,
            filing.filing_type,
            filing.filing_date,
            filing.submission_date,
            filing.document_url,
            filing.ingested_at,
        ),
    )
    conn.commit()
    conn.close()


def get_cached_filings(
    filing_type: str, days_back: int = 7
) -> List[FilingMetadata]:
    """Retrieve cached filings from past N days."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    cursor.execute(
        """SELECT accession_number, cik, ticker, company_name, filing_type,
                  filing_date, submission_date, document_url, ingested_at
           FROM filings
           WHERE filing_type = ? AND ingested_at >= ?
           ORDER BY filing_date DESC""",
        (filing_type, cutoff_date),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        FilingMetadata(
            accession_number=row[0],
            cik=row[1],
            ticker=row[2],
            company_name=row[3],
            filing_type=row[4],
            filing_date=row[5],
            submission_date=row[6],
            document_url=row[7],
            ingested_at=row[8],
        )
        for row in rows
    ]


def parse_rss_feed(feed_url: str, filing_type: str) -> List[FilingMetadata]:
    """Fetch and parse SEC EDGAR RSS feed for filing metadata."""
    filings = []
    try:
        response = requests.get(feed_url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {filing_type} feed: {e}")
        return filings

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as e:
        print(f"Failed to parse RSS XML: {e}")
        return filings

    # Namespace handling for Atom feeds
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    for entry in entries:
        try:
            title_elem = entry.find("atom:title", ns)
            updated_elem = entry.find("atom:updated", ns)
            link_elem = entry.find("atom:link", ns)
            summary_elem = entry.find("atom:summary", ns)

            if not all([title_elem, updated_elem, link_elem, summary_elem]):
                continue

            title = title_elem.text or ""
            filing_date = updated_elem.text or ""
            doc_url = link_elem.get("href", "")
            summary = summary_elem.text or ""

            # Extract CIK and accession from summary/title
            parts = title.split("-")
            if len(parts) < 2:
                continue

            company_name = parts[0].strip()
            cik = extract_cik_from_summary(summary)
            accession_num = extract_accession_from_summary(summary)

            if not cik or not accession_num:
                continue

            if filing_exists(accession_num):
                continue

            filing = FilingMetadata(
                accession_number=accession_num,
                cik=cik,
                ticker=None,  # Will be enriched by upstream ticker mapping
                company_name=company_name,
                filing_type=filing_type,
                filing_date=filing_date[:10],  # ISO date
                submission_date=filing_date[:10],
                document_url=doc_url,
                ingested_at=datetime.utcnow().isoformat(),
            )
            filings.append(filing)
            save_filing(filing)

        except (AttributeError, IndexError, ValueError) as e:
            print(f"Error parsing entry: {e}")
            continue

    return filings


def extract_cik_from_summary(summary: str) -> Optional[str]:
    """Extract CIK from SEC filing summary text using simple regex."""
    import re
    match = re.search(r"CIK[:\s]+(\d+)", summary, re.IGNORECASE)
    return match.group
