"""
SEC EDGAR RSS scraper for Sentinel Scout pillar.

Polls the SEC EDGAR 8-K and 10-Q RSS feeds, extracts filing metadata
(CIK, accession number, filing date, company name), and normalizes into
dataclasses for downstream Linguist and Historian analysis.

Uses Gemini for robust HTML parsing of SEC feed entries when needed.
Integrates with scout's unified ingestion pipeline.
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List
import xml.etree.ElementTree as ET

import requests
import google.generativeai as genai
from bs4 import BeautifulSoup


GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
SEC_RSS_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"


@dataclass
class SECFiling:
    """Normalized SEC filing metadata."""
    cik: str
    company_name: str
    accession_number: str
    filing_type: str
    filing_date: str
    submission_url: str
    raw_html: Optional[str] = None


def _init_gemini() -> None:
    """Initialize Gemini API client from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)


def fetch_sec_rss_feed(feed_type: str = "8-K") -> str:
    """
    Fetch raw SEC EDGAR RSS feed (8-K or 10-Q).
    
    Args:
        feed_type: "8-K" or "10-Q"
    
    Returns:
        Raw XML string from SEC feed.
    """
    params = {
        "action": "getcompany",
        "type": feed_type,
        "dateb": "",
        "owner": "exclude",
        "count": 100,
        "output": "atom",
    }
    resp = requests.get(SEC_RSS_BASE, params=params, timeout=10)
    resp.raise_for_status()
    return resp.text


def parse_sec_rss_feed(xml_content: str, feed_type: str = "8-K") -> List[SECFiling]:
    """
    Parse SEC EDGAR Atom feed XML and extract filing metadata.
    
    Args:
        xml_content: Raw Atom feed XML
        feed_type: "8-K" or "10-Q" for context
    
    Returns:
        List of normalized SECFiling dataclasses.
    """
    filings = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse SEC RSS XML: {e}")

    # Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    # Extract entries
    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        link_elem = entry.find("atom:link", ns)
        updated_elem = entry.find("atom:updated", ns)
        summary_elem = entry.find("atom:summary", ns)
        
        if title_elem is None or link_elem is None:
            continue
        
        title = (title_elem.text or "").strip()
        submission_url = link_elem.get("href", "")
        filing_date = (updated_elem.text or "")[:10]  # YYYY-MM-DD
        summary_text = (summary_elem.text or "").strip()
        
        # Parse title: typically "Company Name (CIK) - 8-K" or similar
        parts = title.rsplit(" - ", 1)
        company_info = parts[0] if parts else title
        
        # Extract CIK from parentheses
        cik = ""
        if "(" in company_info and ")" in company_info:
            cik_part = company_info[company_info.rfind("(") + 1 : company_info.rfind(")")]
            cik = cik_part.strip()
        
        company_name = company_info.replace(f"({cik})", "").strip()
        
        # Extract accession number from URL
        accession_number = ""
        if "accession_number=" in submission_url:
            accession_number = submission_url.split("accession_number=")[-1].split("&")[0]
        
        filings.append(
            SECFiling(
                cik=cik,
                company_name=company_name,
                accession_number=accession_number,
                filing_type=feed_type,
                filing_date=filing_date,
                submission_url=submission_url,
                raw_html=summary_text,
            )
        )
    
    return filings


def extract_filing_details_with_gemini(filing: SECFiling) -> SECFiling:
    """
    Use Gemini to extract additional details from SEC filing summary if needed.
    
    Args:
        filing: SECFiling object with raw_html populated
    
    Returns:
        Enriched SECFiling (may add parsed context to raw_html field).
    """
    if not filing.raw_html:
        return filing
    
    prompt = f"""
    Analyze this SEC filing summary and extract key points (max 3 sentences):
    
    {filing.raw_html}
    
    Return only the extracted key points.
    """
    
    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        filing.raw_html = response.text
    except Exception as e:
        # Silently degrade if Gemini fails; preserve original
        print(f"Gemini extraction failed for {filing.accession_number}: {e}")
    
    return filing


def store_filings_in_db(filings: List[SECFiling], db_path: str = "sentinel_sec.db") -> None:
    """
    Store normalized filings in SQLite for deduplication and historical lookup.
    
    Args:
        filings: List of SECFiling objects
        db_path: Path to SQLite database file
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cik TEXT,
            company_name TEXT,
            accession_number TEXT UNIQUE,
            filing_type TEXT,
            filing_date TEXT,
            submission_url TEXT,
            raw_html TEXT,
            ingested_at TEXT
        )
    """)
    
    for filing in filings:
        cursor.execute("""
            INSERT OR REPLACE INTO sec_filings
            (cik, company_name, accession_number, filing_type, filing_date, submission_url, raw_html, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            filing.cik,
            filing.company_name,
            filing.accession_number,
            filing.filing_type,
            filing.filing_date,
            filing.submission_url,
            filing.raw_html,
            datetime.utcnow().isoformat(),
        ))
    
    conn.commit()
    conn.close()


def scrape_sec_filings(
    feed_types: Optional[List[str]] = None,
    use_gemini: bool = False,
    store_db: bool = True,
) -> List[SECFiling]:
    """
    Main entry point: fetch, parse, optionally enrich, and store SEC filings.
    
    Args:
        feed_types: List of filing types to scrape (default ["8-K", "10-Q"])
        use_gemini: If True, use Gemini to extract additional details
        store_db: If True, persist filings to SQLite
