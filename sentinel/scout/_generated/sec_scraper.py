"""
SEC EDGAR RSS scraper for Sentinel Scout agent.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(company name, CIK, accession number, filing date, form type), and normalizes
into a unified Filing dataclass for downstream Linguist analysis.

Part of the Scout pillar's data ingestion pipeline.
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Filing:
    """Normalized SEC filing metadata for Sentinel ingestion."""
    cik: str
    company_name: str
    accession_number: str
    filing_date: str
    form_type: str
    html_url: str
    txt_url: str
    fetched_at: str
    raw_xml: Optional[str] = None


# ============================================================================
# SEC EDGAR RSS Scraper
# ============================================================================

class SECEdgarScraper:
    """Polls SEC EDGAR RSS feeds and extracts filing metadata."""
    
    # SEC EDGAR RSS feed URLs (company search, 8-K and 10-Q)
    EDGAR_RSS_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    def __init__(self, db_path: str = "sentinel_filings.db"):
        """Initialize scraper and create database schema if needed."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Create SQLite schema for filings if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cik TEXT NOT NULL,
                company_name TEXT NOT NULL,
                accession_number TEXT UNIQUE NOT NULL,
                filing_date TEXT NOT NULL,
                form_type TEXT NOT NULL,
                html_url TEXT,
                txt_url TEXT,
                fetched_at TEXT NOT NULL,
                raw_xml TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_accession ON filings(accession_number)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_filing_date ON filings(filing_date)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_form_type ON filings(form_type)
        """)
        conn.commit()
        conn.close()
    
    def fetch_rss_feed(self, form_type: str = "8-K", count: int = 100) -> Optional[str]:
        """Fetch SEC EDGAR RSS feed for a given form type.
        
        Args:
            form_type: "8-K", "10-Q", "10-K", etc.
            count: Number of recent filings to fetch (SEC max 100 per request).
        
        Returns:
            Raw RSS XML as string, or None on network error.
        """
        params = {
            "action": "getcompany",
            "type": form_type,
            "dateb": "",
            "owner": "exclude",
            "count": count,
            "output": "xml"
        }
        try:
            response = requests.get(self.EDGAR_RSS_BASE, params=params, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching SEC RSS feed: {e}")
            return None
    
    def parse_rss_feed(self, xml_content: str) -> List[Filing]:
        """Parse SEC EDGAR RSS XML and extract filing metadata.
        
        Args:
            xml_content: Raw RSS XML from SEC EDGAR.
        
        Returns:
            List of normalized Filing dataclass instances.
        """
        filings = []
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            print(f"Error parsing RSS XML: {e}")
            return filings
        
        # Namespaces in SEC RSS feed
        ns = {
            "": "http://www.sec.gov/cgi-bin",
            "atom": "http://www.w3.org/2005/Atom"
        }
        
        # Find all entry elements (each is one filing)
        entries = root.findall(".//entry", ns) or root.findall(".//entry")
        
        for entry in entries:
            try:
                # Extract text content from subelements
                company_name_elem = entry.find("company-info")
                company_name = (
                    company_name_elem.text if company_name_elem is not None else "Unknown"
                )
                
                cik_elem = entry.find("cik")
                cik = cik_elem.text if cik_elem is not None else ""
                
                accession_elem = entry.find("accession-number")
                accession_number = accession_elem.text if accession_elem is not None else ""
                
                filing_date_elem = entry.find("filing-date")
                filing_date = filing_date_elem.text if filing_date_elem is not None else ""
                
                form_type_elem = entry.find("form-type")
                form_type = form_type_elem.text if form_type_elem is not None else ""
                
                # Extract URLs from links
                html_url = ""
                txt_url = ""
                links = entry.findall(".//link", ns) or entry.findall(".//link")
                for link in links:
                    href = link.get("href", "")
                    rel = link.get("rel", "")
                    if "htm" in href.lower():
                        html_url = href
                    elif "txt" in href.lower() or rel == "alternate":
                        txt_url = href
                
                # Construct Filing dataclass
                filing = Filing(
                    cik=cik.strip(),
                    company_name=company_name.strip(),
                    accession_number=accession_number.strip(),
                    filing_date=filing_date.strip(),
                    form_type=form_type.strip(),
                    html_url=html_url,
                    txt_url=txt_url,
                    fetched_at=datetime.utcnow().isoformat(),
                    raw_xml=ET.tostring(entry, encoding="unicode")
                )
                
                # Only append if we have essential fields
                if filing.cik and filing.accession_number:
                    filings.append(filing)
            
            except (AttributeError, ValueError) as e:
                print(f"Error parsing entry: {e}")
                continue
        
        return filings
    
    def store_filings(self, filings: List[Filing]) -> int:
        """Store filings into SQLite database, skipping duplicates.
        
        Args:
            filings: List of Filing dataclass instances.
        
        Returns:
            Number of successfully inserted filings.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        inserted = 0
        
        for filing in filings:
            try:
                cursor.execute("""
                    INSERT INTO filings (
                        cik, company_name, accession_number, filing_date,
                        form_type, html_url, txt_url, fetched_at, raw_xml
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    filing.cik,
                    filing.company_name,
                    filing.accession_number,
                    filing.filing_date,
                    filing.form_type,
                    filing.html_url,
                    filing.txt_url,
                    filing.fetched_at,
                    filing.raw_xml
                ))
                inserted += 1
            except
