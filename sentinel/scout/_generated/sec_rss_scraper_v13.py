"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls SEC EDGAR RSS feeds (8-K, 10-Q, 10-K) to extract filing metadata
(CIK, accession number, filing date, company name, form type) into normalized
dataclasses. Feeds filings into the Historian RAG pipeline for sentiment analysis.

Uses Gemini (via google-generativeai) for robust HTML/XML parsing of feed entries.
Caches parsed results in SQLite to avoid re-processing identical accession numbers.
"""

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import xml.etree.ElementTree as ET

import google.generativeai as genai
import requests

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from EDGAR RSS."""

    cik: str
    """Central Index Key (company identifier)."""

    accession_number: str
    """Unique accession number (e.g., 0001193125-23-123456)."""

    company_name: str
    """Official company name from SEC."""

    form_type: str
    """Filing form type (8-K, 10-Q, 10-K, etc.)."""

    filing_date: datetime
    """Date filing was submitted to SEC."""

    feed_url: str
    """RSS feed URL source."""

    raw_summary: Optional[str] = None
    """Raw filing summary/excerpt from RSS entry."""


def _init_cache_db(db_path: str = "sentinel_sec_cache.db") -> None:
    """Initialize SQLite cache for SEC filings."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sec_filings (
            accession_number TEXT PRIMARY KEY,
            cik TEXT,
            company_name TEXT,
            form_type TEXT,
            filing_date TEXT,
            feed_url TEXT,
            raw_summary TEXT,
            cached_at TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def _get_cached_filing(accession_number: str, db_path: str = "sentinel_sec_cache.db") -> Optional[SECFiling]:
    """Retrieve cached filing by accession number."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cik, company_name, form_type, filing_date, feed_url, raw_summary
            FROM sec_filings WHERE accession_number = ?
        """,
            (accession_number,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return SECFiling(
                cik=row[0],
                accession_number=accession_number,
                company_name=row[1],
                form_type=row[2],
                filing_date=datetime.fromisoformat(row[3]),
                feed_url=row[4],
                raw_summary=row[5],
            )
    except sqlite3.Error:
        pass
    return None


def _cache_filing(filing: SECFiling, db_path: str = "sentinel_sec_cache.db") -> None:
    """Store filing in cache."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sec_filings
            (accession_number, cik, company_name, form_type, filing_date, feed_url, raw_summary, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                filing.accession_number,
                filing.cik,
                filing.company_name,
                filing.form_type,
                filing.filing_date.isoformat(),
                filing.feed_url,
                filing.raw_summary,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error:
        pass


def _parse_rss_entry_with_gemini(entry_xml: str) -> dict:
    """Use Gemini to parse RSS entry XML and extract structured fields."""
    if not GEMINI_API_KEY:
        return {}

    prompt = f"""
Extract the following fields from this SEC EDGAR RSS entry XML. Return a JSON object with keys:
cik, accession_number, company_name, form_type, filing_date (ISO format), raw_summary.

If a field is missing, use null. Accession numbers are typically in the format like 0001193125-23-123456.
filing_date should be in YYYY-MM-DD format.

RSS Entry XML:
{entry_xml}

Return only valid JSON, no markdown or extra text.
"""

    try:
        response = genai.generate_text(prompt=prompt, max_output_tokens=500)
        if response and response.result:
            import json

            return json.loads(response.result)
    except Exception:
        pass
    return {}


def fetch_sec_8k_filings(limit: int = 50) -> list[SECFiling]:
    """
    Fetch latest 8-K filings from SEC EDGAR RSS feed.

    Args:
        limit: Maximum number of filings to return.

    Returns:
        List of SECFiling dataclass instances.
    """
    _init_cache_db()
    feed_url = "https://www.sec.gov/rss/cgi-svc/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&start=0&count=100&myHID=&SortTicker=&hidefilings=false&output=atom"
    return _fetch_sec_feed(feed_url, form_type="8-K", limit=limit)


def fetch_sec_10q_filings(limit: int = 50) -> list[SECFiling]:
    """
    Fetch latest 10-Q filings from SEC EDGAR RSS feed.

    Args:
        limit: Maximum number of filings to return.

    Returns:
        List of SECFiling dataclass instances.
    """
    _init_cache_db()
    feed_url = "https://www.sec.gov/rss/cgi-svc/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&start=0&count=100&myHID=&SortTicker=&hidefilings=false&output=atom"
    return _fetch_sec_feed(feed_url, form_type="10-Q", limit=limit)


def fetch_sec_10k_filings(limit: int = 50) -> list[SECFiling]:
    """
    Fetch latest 10-K filings from SEC EDGAR RSS feed.

    Args:
        limit: Maximum number of filings to return.

    Returns:
        List of SECFiling dataclass instances.
    """
    _init_cache_db()
    feed_url = "https://www.sec.gov/rss/cgi-svc/browse-edgar?action=getcompany&type=10-K&dateb=&owner=exclude&start=0&count=100&myHID=&SortTicker=&hidefilings=false&output=atom"
    return _fetch_sec_feed(feed_url, form_type="10-K", limit=limit)


def _fetch_sec_feed(feed_url: str, form_type: str, limit: int = 50) -> list[SECFiling]:
    """
    Poll SEC EDGAR RSS feed and extract filing metadata.

    Args:
        feed_url: Full SEC RSS feed URL.
        form_type: Expected form type (8-K, 10-Q, 10-K).
        limit: Maximum filings to return.

    Returns:
        List of SECFiling instances.
