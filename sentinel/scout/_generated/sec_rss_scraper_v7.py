"""
SEC EDGAR RSS feed scraper for Sentinel Scout pillar.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(company name, CIK, accession number, filing date, form type), and normalizes into
dataclasses for downstream Linguist and Historian ingestion.

Uses `requests` + `xml.etree.ElementTree` for lightweight RSS parsing.
No LLM calls — pure data extraction pipeline.
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List
import xml.etree.ElementTree as ET

import requests


@dataclass
class SECFiling:
    """Normalized SEC EDGAR filing metadata."""
    cik: str
    company_name: str
    form_type: str
    accession_number: str
    filing_date: str
    acceptance_datetime: str
    href: str
    html_href: Optional[str] = None
    txt_href: Optional[str] = None
    source_feed: str = "sec_edgar_rss"


class SECRSSScraperError(Exception):
    """Raised when RSS polling or parsing fails."""
    pass


def fetch_sec_rss_feed(feed_url: str, timeout: int = 15) -> str:
    """Fetch raw XML from SEC EDGAR RSS feed."""
    try:
        resp = requests.get(feed_url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        raise SECRSSScraperError(f"Failed to fetch {feed_url}: {e}") from e


def parse_sec_rss_feed(xml_text: str, form_types: Optional[List[str]] = None) -> List[SECFiling]:
    """
    Parse SEC EDGAR RSS XML and extract filings.
    
    Args:
        xml_text: Raw RSS XML string from SEC feed.
        form_types: Optional list of form types to filter (e.g., ['8-K', '10-Q']).
                    If None, all forms are included.
    
    Returns:
        List of normalized SECFiling dataclasses.
    """
    if form_types is None:
        form_types = []
    
    filings = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise SECRSSScraperError(f"Failed to parse RSS XML: {e}") from e
    
    # SEC RSS uses standard RSS namespace; items are in channel/item
    ns = {"": "http://www.w3.org/2005/Atom"}
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    
    for item in items:
        # Extract fields from RSS item
        title_elem = item.find("title")
        title = title_elem.text if title_elem is not None else ""
        
        link_elem = item.find("link")
        link = link_elem.text if link_elem is not None else ""
        if link_elem is not None and link_elem.get("href"):
            link = link_elem.get("href")
        
        pubdate_elem = item.find("pubDate") or item.find("{http://www.w3.org/2005/Atom}published")
        pubdate = pubdate_elem.text if pubdate_elem is not None else ""
        
        summary_elem = item.find("description") or item.find("{http://www.w3.org/2005/Atom}summary")
        summary = summary_elem.text if summary_elem is not None else ""
        
        # Parse title: typically "Company Name (CIK) Form Type Filed"
        # e.g., "Tesla Inc (0001318605) 8-K Filed"
        parts = title.split(" (")
        company_name = parts[0].strip() if parts else ""
        
        cik = ""
        form_type = ""
        if len(parts) > 1:
            cik_and_rest = parts[1]
            cik_parts = cik_and_rest.split(")")
            cik = cik_parts[0].strip() if cik_parts else ""
            
            form_info = cik_parts[1].strip() if len(cik_parts) > 1 else ""
            form_tokens = form_info.split()
            if form_tokens:
                form_type = form_tokens[0]
        
        # Filter by form type if specified
        if form_types and form_type not in form_types:
            continue
        
        # Extract accession number from link if available
        # SEC links typically: https://www.sec.gov/cgi-bin/viewer?action=view&cik=...&accession_number=...
        accession_number = ""
        if "accession_number=" in link:
            acc_part = link.split("accession_number=")[1].split("&")[0]
            accession_number = acc_part
        
        filing = SECFiling(
            cik=cik,
            company_name=company_name,
            form_type=form_type,
            accession_number=accession_number,
            filing_date=pubdate,
            acceptance_datetime=pubdate,
            href=link,
            html_href=None,
            txt_href=None,
            source_feed="sec_edgar_rss",
        )
        filings.append(filing)
    
    return filings


def scrape_sec_filings(
    form_types: Optional[List[str]] = None,
    hours_back: int = 24,
) -> List[SECFiling]:
    """
    Poll SEC EDGAR RSS feeds for recent filings.
    
    Args:
        form_types: List of form types to fetch (e.g., ['8-K', '10-Q', '10-K']).
                    If None, fetches both 8-K and 10-Q.
        hours_back: Only return filings from the last N hours (for caching logic).
    
    Returns:
        List of SECFiling dataclasses.
    """
    if form_types is None:
        form_types = ["8-K", "10-Q"]
    
    all_filings = []
    
    # SEC provides separate RSS feeds per form type
    feed_urls = {
        "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&output=atom",
        "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&output=atom",
        "10-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K&dateb=&owner=exclude&count=100&myHID=&output=atom",
    }
    
    for form_type in form_types:
        if form_type not in feed_urls:
            continue
        
        try:
            xml_text = fetch_sec_rss_feed(feed_urls[form_type])
            filings = parse_sec_rss_feed(xml_text, form_types=[form_type])
            all_filings.extend(filings)
        except SECRSSScraperError as e:
            print(f"Warning: Failed to scrape {form_type} feed: {e}")
    
    return all_filings


def cache_filings_to_sqlite(
    filings: List[SECFiling],
    db_path: str = "sentinel_sec_filings.db",
) -> None:
    """
    Persist filings to SQLite for deduplication and historical querying.
    
    Args:
        filings: List of SECFiling objects.
        db_path: Path to SQLite database file.
    """
    conn
