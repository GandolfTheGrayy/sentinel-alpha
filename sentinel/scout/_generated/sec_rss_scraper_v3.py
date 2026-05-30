"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR 8-K and 10-Q RSS feeds, extracts filing metadata
(CIK, ticker, filing date, accession number, form type), normalizes into
dataclass records, and stores in a local SQLite index for fast lookup by
the Historian and Judge pillars.

This module is called by sentinel/scout/sec_filings.py as a supplementary
ingestion path to maintain real-time awareness of material disclosures.
"""

import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Optional
import requests


@dataclass
class SECFiling:
    """Normalized SEC EDGAR filing record."""
    cik: str
    ticker: Optional[str]
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str
    rss_feed_url: str
    raw_summary: str


class SECRSSIndex:
    """SQLite-backed index of SEC EDGAR filings from RSS feeds."""

    def __init__(self, db_path: str = "sentinel.db"):
        """Initialize SQLite connection and schema."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they do not exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                ticker TEXT,
                company_name TEXT NOT NULL,
                form_type TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                accession_number TEXT UNIQUE NOT NULL,
                rss_feed_url TEXT,
                raw_summary TEXT,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON sec_filings(ticker, filing_date DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_accession
            ON sec_filings(accession_number)
        """)
        conn.commit()
        conn.close()

    def insert_filings(self, filings: List[SECFiling]) -> int:
        """Insert filings, skipping duplicates by accession_number."""
        conn = sqlite3.connect(self.db_path)
        inserted = 0
        for filing in filings:
            try:
                conn.execute("""
                    INSERT INTO sec_filings
                    (cik, ticker, company_name, form_type, filing_date,
                     accession_number, rss_feed_url, raw_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (filing.cik, filing.ticker, filing.company_name,
                      filing.form_type, filing.filing_date,
                      filing.accession_number, filing.rss_feed_url,
                      filing.raw_summary))
                inserted += 1
            except sqlite3.IntegrityError:
                # Duplicate accession number; skip.
                pass
        conn.commit()
        conn.close()
        return inserted

    def query_by_ticker_recent(self, ticker: str, days: int = 7) -> List[SECFiling]:
        """Query filings for a ticker from the past N days."""
        conn = sqlite3.connect(self.db_path)
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        rows = conn.execute("""
            SELECT cik, ticker, company_name, form_type, filing_date,
                   accession_number, rss_feed_url, raw_summary
            FROM sec_filings
            WHERE ticker = ? AND filing_date >= ?
            ORDER BY filing_date DESC
        """, (ticker, cutoff)).fetchall()
        conn.close()
        return [SECFiling(*row) for row in rows]

    def query_by_form_type(self, form_type: str, limit: int = 100) -> List[SECFiling]:
        """Query most recent filings of a given form type."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT cik, ticker, company_name, form_type, filing_date,
                   accession_number, rss_feed_url, raw_summary
            FROM sec_filings
            WHERE form_type = ?
            ORDER BY filing_date DESC
            LIMIT ?
        """, (form_type, limit)).fetchall()
        conn.close()
        return [SECFiling(*row) for row in rows]


class SECRSSPoller:
    """Polls SEC EDGAR RSS feeds and extracts filing metadata."""

    FEEDS = {
        "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                "&type=8-K&dateb=&owner=exclude&count=100&myHID=&search_text="
                "&CIK=&fromdate=&todate=&output=atom",
        "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                "&type=10-Q&dateb=&owner=exclude&count=100&myHID=&search_text="
                "&CIK=&fromdate=&todate=&output=atom",
    }

    def __init__(self, timeout: int = 10):
        """Initialize poller with request timeout."""
        self.timeout = timeout

    def poll_feed(self, feed_url: str, form_type: str) -> List[SECFiling]:
        """Fetch and parse a single SEC EDGAR RSS feed."""
        filings = []
        try:
            resp = requests.get(feed_url, timeout=self.timeout)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as e:
            print(f"Error fetching {form_type} feed: {e}")
            return filings

        # SEC EDGAR Atom feeds use namespace 'http://www.w3.org/2005/Atom'
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            try:
                title = entry.findtext("atom:title", default="").strip()
                summary = entry.findtext("atom:summary", default="").strip()
                updated = entry.findtext("atom:updated", default="").strip()

                # Parse title: typically "Company Name (CIK: 0000000123) 8-K"
                cik, company_name = self._parse_title(title)
                ticker = self._lookup_ticker_from_cik(cik)
                accession = self._extract_accession(summary)

                if cik and accession:
                    filing = SECFiling(
                        cik=cik,
                        ticker=ticker,
                        company_name=company_name,
                        form_type=form_type,
                        filing_date=updated[:10],  # YYYY-MM-DD
                        accession_number=accession,
                        rss_feed_url=feed_url,
                        raw_summary=summary[:500],
                    )
                    filings.append(filing)
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue

        return filings

    def poll_all_feeds(self) -> List[SECFiling]:
        """Poll all configured feeds and return combined filings."""
        all_filings = []
        for form_type, feed_url in self.FEEDS.items():
            filings = self.poll_feed(feed_url, form_type)
            all_filings.extend(filings)
        return all_filings

    @staticmethod
    def _parse_title(title: str) -> tuple[str, str]:
        """Extract CIK and company
