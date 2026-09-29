"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(CIK, accession number, filing date, company name, form type), normalizes into
dataclasses, and stores in local SQLite for deduplication and historical lookup.
Integrates with the Scout pillar to supply raw filing signals to Linguist analysis.
"""

import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
from xml.etree import ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """Normalized SEC EDGAR filing metadata."""
    cik: str
    accession_number: str
    company_name: str
    filing_date: str
    form_type: str
    filing_url: str
    extracted_at: str


def _create_session() -> requests.Session:
    """Create a requests Session with retry strategy for SEC EDGAR."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504)
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Sentinel-Sentiment-Engine (sentinelbot@example.com)"
    })
    return session


def fetch_sec_rss(form_types: List[str] = None, max_entries: int = 100) -> List[SECFiling]:
    """
    Fetch SEC EDGAR filings from official RSS feeds.
    
    Args:
        form_types: List of form types to fetch (e.g. ['8-K', '10-Q']). Defaults to ['8-K', '10-Q'].
        max_entries: Maximum entries to parse per feed.
    
    Returns:
        List of normalized SECFiling dataclass instances.
    """
    if form_types is None:
        form_types = ["8-K", "10-Q"]
    
    filings = []
    session = _create_session()
    
    # SEC EDGAR RSS feed URLs by form type
    rss_feeds = {
        "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&search_text=&format=rss",
        "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&search_text=&format=rss",
        "10-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=exclude&count=100&myHID=&search_text=&format=rss",
    }
    
    extracted_at = datetime.utcnow().isoformat()
    
    for form_type in form_types:
        if form_type not in rss_feeds:
            logger.warning(f"Unsupported form type: {form_type}")
            continue
        
        feed_url = rss_feeds[form_type]
        logger.info(f"Fetching {form_type} RSS feed from {feed_url}")
        
        try:
            resp = session.get(feed_url, timeout=10)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            
            # Parse RSS items
            ns = {"": "http://www.sec.gov/cgi-bin/browse-edgar"}
            items = root.findall(".//item")
            
            for i, item in enumerate(items):
                if i >= max_entries:
                    break
                
                try:
                    # Extract title: "Company Name (CIK) Form 8-K"
                    title_elem = item.find("title")
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Parse title format
                    parts = title.rsplit(" ", 1)
                    if len(parts) == 2:
                        company_info, extracted_form = parts[0], parts[1]
                        # company_info is "Name (CIK)"
                        if "(" in company_info and ")" in company_info:
                            name_part = company_info.rsplit("(", 1)[0].strip()
                            cik_part = company_info.rsplit("(", 1)[1].rstrip(")")
                        else:
                            name_part = company_info
                            cik_part = "UNKNOWN"
                    else:
                        name_part = title
                        cik_part = "UNKNOWN"
                        extracted_form = form_type
                    
                    # Extract filing date from pubDate
                    pub_date_elem = item.find("pubDate")
                    filing_date = pub_date_elem.text if pub_date_elem is not None else extracted_at
                    
                    # Extract accession number and URL from description/link
                    description_elem = item.find("description")
                    description = description_elem.text if description_elem is not None else ""
                    
                    link_elem = item.find("link")
                    filing_url = link_elem.text if link_elem is not None else ""
                    
                    # Extract accession number from description (format: Accession Number: 0000950154-24-001234)
                    accession_number = "UNKNOWN"
                    if "Accession Number:" in description:
                        acc_part = description.split("Accession Number:")[-1].strip().split()[0]
                        accession_number = acc_part
                    
                    filing = SECFiling(
                        cik=cik_part,
                        accession_number=accession_number,
                        company_name=name_part,
                        filing_date=filing_date,
                        form_type=extracted_form,
                        filing_url=filing_url,
                        extracted_at=extracted_at
                    )
                    filings.append(filing)
                    logger.debug(f"Parsed filing: {filing.company_name} {filing.form_type} {filing.accession_number}")
                
                except Exception as e:
                    logger.error(f"Error parsing item in {form_type} feed: {e}", exc_info=True)
                    continue
        
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {form_type} RSS feed: {e}")
            continue
    
    session.close()
    logger.info(f"Extracted {len(filings)} SEC filings from RSS feeds")
    return filings


def init_filings_db(db_path: str = "sentinel_filings.db") -> None:
    """
    Initialize SQLite database for SEC filings with deduplication index.
    
    Args:
        db_path: Path to SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT NOT NULL,
            accession_number TEXT UNIQUE NOT NULL,
            company_name TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            form_type TEXT NOT NULL,
            filing_url TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_accession ON sec_filings(accession_number)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_cik
