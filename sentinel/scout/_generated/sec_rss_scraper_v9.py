"""
SEC EDGAR RSS scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts metadata
(CIK, company name, filing type, accession number, filing date), and normalizes
into a dataclass for downstream ingestion into the Historian RAG pipeline.

Uses Gemini (via google-generativeai) for efficient RSS/XML parsing when needed,
but defaults to stdlib xml.etree for performance on high-volume feeds.
"""

import os
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List
import requests
from google import generativeai as genai


@dataclass
class SecFiling:
    """Normalized SEC filing metadata."""
    cik: str
    company_name: str
    filing_type: str
    accession_number: str
    filing_date: str
    submission_date: str
    document_url: str
    raw_xml: Optional[str] = None


class SecRssScraper:
    """Poll SEC EDGAR RSS feeds and extract normalized filing records."""

    # SEC EDGAR RSS feed endpoints for 8-K and 10-Q
    FEEDS = {
        "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&output=atom",
        "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&output=atom",
    }

    DB_PATH = "sentinel_sec_filings.db"

    def __init__(self) -> None:
        """Initialize scraper and set up local cache DB."""
        self._ensure_db()
        # Gemini key optional; used only if explicit parsing needed
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)

    def _ensure_db(self) -> None:
        """Create SQLite table for filing deduplication."""
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                accession_number TEXT PRIMARY KEY,
                cik TEXT NOT NULL,
                company_name TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                filing_date TEXT NOT NULL,
                submission_date TEXT NOT NULL,
                document_url TEXT NOT NULL,
                fetched_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def fetch_feed(self, filing_type: str) -> str:
        """Fetch raw RSS/Atom feed XML for a filing type."""
        if filing_type not in self.FEEDS:
            raise ValueError(f"Unknown filing type: {filing_type}")
        url = self.FEEDS[filing_type]
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text

    def parse_atom_feed(self, xml_text: str, filing_type: str) -> List[SecFiling]:
        """Parse SEC EDGAR Atom feed and extract filing metadata using stdlib XML."""
        filings = []
        try:
            root = ET.fromstring(xml_text)
            # Atom namespace
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "xhtml": "http://www.w3.org/1999/xhtml",
            }
            # Extract entry elements
            for entry in root.findall("atom:entry", ns):
                title = entry.findtext("atom:title", "", ns) or ""
                link = entry.find("atom:link", ns)
                doc_url = link.get("href") if link is not None else ""
                summary = entry.findtext("atom:summary", "", ns) or ""
                updated = entry.findtext("atom:updated", "", ns) or ""

                # Parse title: e.g., "Apple Inc - 8-K/A (0000320193) - Filed 2025-01-15"
                parts = title.split(" - ")
                company_name = parts[0] if parts else ""
                cik = self._extract_cik_from_title(title)
                accession = self._extract_accession(doc_url)

                if cik and accession:
                    filing = SecFiling(
                        cik=cik,
                        company_name=company_name,
                        filing_type=filing_type,
                        accession_number=accession,
                        filing_date=self._parse_date(updated),
                        submission_date=self._parse_date(updated),
                        document_url=doc_url,
                        raw_xml=summary,
                    )
                    filings.append(filing)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
        return filings

    def _extract_cik_from_title(self, title: str) -> Optional[str]:
        """Extract 10-digit CIK from SEC filing title."""
        import re
        match = re.search(r"\((\d{10})\)", title)
        return match.group(1) if match else None

    def _extract_accession(self, url: str) -> Optional[str]:
        """Extract accession number from SEC document URL."""
        import re
        # e.g., .../0000320193-25-000016
        match = re.search(r"(\d{10}-\d{2}-\d{6})", url)
        return match.group(1) if match else None

    def _parse_date(self, iso_str: str) -> str:
        """Parse ISO date string to YYYY-MM-DD."""
        try:
            dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            return iso_str[:10] if iso_str else ""

    def scrape_all(self) -> List[SecFiling]:
        """Poll all feeds, parse, deduplicate, and return new filings."""
        all_filings = []
        for filing_type in self.FEEDS.keys():
            try:
                xml = self.fetch_feed(filing_type)
                filings = self.parse_atom_feed(xml, filing_type)
                all_filings.extend(filings)
            except requests.RequestException as e:
                print(f"Error fetching {filing_type} feed: {e}")
        
        # Write to DB and filter for new entries
        new_filings = self._deduplicate_and_store(all_filings)
        return new_filings

    def _deduplicate_and_store(self, filings: List[SecFiling]) -> List[SecFiling]:
        """Store filings in DB, return only new ones."""
        conn = sqlite3.connect(self.DB_PATH)
        c = conn.cursor()
        new_filings = []
        now = datetime.utcnow().isoformat()

        for filing in filings:
            try:
                c.execute(
                    "INSERT INTO sec_filings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        filing.accession_number,
                        filing.cik,
                        filing.company_name,
                        filing.filing_type,
                        filing.filing_date,
                        filing.submission_date,
                        filing.document_url,
                        now,
                    ),
                )
                new_filings.append(filing)
            except sqlite3.IntegrityError:
                # Already in DB
                pass

        conn.commit()
        conn.close()
        return new_filings

    def get_recent_filings(self, days: int = 7) -> List[SecFiling]:
        """Retrieve filings from DB from the last N
