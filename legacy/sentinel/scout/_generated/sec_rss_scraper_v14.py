"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds (8-K, 10-Q, 10-K) in real-time to extract
filing metadata (CIK, accession number, company name, filing date, form type).
Normalizes output into dataclass records for downstream Historian RAG indexing
and Linguist sentiment analysis.

Uses requests + xml.etree for lightweight feed parsing; Gemini (via scout scraper
pattern) only if HTML extraction from filing details is needed. Primary job:
pull feed metadata fast and normalize it.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import list
import xml.etree.ElementTree as ET
import requests
import logging

logger = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """Normalized SEC filing record."""
    cik: str
    accession_number: str
    company_name: str
    form_type: str
    filing_date: datetime
    url: str
    raw_title: str


def fetch_sec_rss_feed(feed_url: str, timeout: int = 10) -> str:
    """Fetch raw XML from SEC EDGAR RSS feed."""
    try:
        resp = requests.get(feed_url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error(f"Failed to fetch SEC RSS feed {feed_url}: {e}")
        return ""


def parse_sec_rss_feed(xml_content: str) -> list[SECFiling]:
    """Parse SEC EDGAR RSS XML and extract filing metadata into dataclass list."""
    if not xml_content:
        return []
    
    filings = []
    try:
        root = ET.fromstring(xml_content)
        # SEC RSS uses default namespace; handle both with and without it
        ns = {'': 'http://www.sec.gov/cgi-bin'}
        items = root.findall('.//item', ns) or root.findall('.//item')
        
        for item in items:
            try:
                # Standard RSS fields
                title = item.findtext('title', '').strip()
                link = item.findtext('link', '').strip()
                pub_date_str = item.findtext('pubDate', '').strip()
                description = item.findtext('description', '').strip()
                
                if not title or not link:
                    continue
                
                # Parse pubDate (RFC 2822 format, e.g., "Fri, 10 Jan 2025 12:34:56 -0500")
                try:
                    filing_date = datetime.strptime(
                        pub_date_str, "%a, %d %b %Y %H:%M:%S %z"
                    )
                except ValueError:
                    filing_date = datetime.now()
                
                # Extract form type and company name from title
                # SEC format: "Company Name form 8-K" or similar
                parts = title.rsplit(' ', 1)
                if len(parts) == 2:
                    company_name, form_type = parts
                    form_type = form_type.upper()
                else:
                    company_name = title
                    form_type = "UNKNOWN"
                
                # Extract CIK and accession from link
                # Format: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001018724&type=8-K&dateb=&owner=exclude&count=100
                # or direct filing link
                cik = ""
                accession_number = ""
                
                if "CIK=" in link:
                    cik_part = link.split("CIK=")[1].split("&")[0]
                    cik = cik_part.lstrip('0') or '0'
                
                # Accession number typically in description or derived from link
                if "accession" in description.lower():
                    # Parse from description if present
                    for word in description.split():
                        if len(word) == 20 and word.count('-') == 2:
                            accession_number = word
                            break
                
                filing = SECFiling(
                    cik=cik,
                    accession_number=accession_number or f"ACC-{filing_date.timestamp()}",
                    company_name=company_name.strip(),
                    form_type=form_type,
                    filing_date=filing_date,
                    url=link,
                    raw_title=title
                )
                filings.append(filing)
            except Exception as e:
                logger.warning(f"Failed to parse SEC RSS item: {e}")
                continue
        
    except ET.ParseError as e:
        logger.error(f"Failed to parse SEC RSS XML: {e}")
        return []
    
    return filings


def scrape_sec_filings(form_types: list[str] = None) -> list[SECFiling]:
    """
    Poll SEC EDGAR RSS feeds for given form types and return normalized filings.
    
    Args:
        form_types: List of form types to fetch (e.g., ['8-K', '10-Q', '10-K']).
                   Defaults to ['8-K', '10-Q'].
    
    Returns:
        List of SECFiling dataclass instances.
    """
    if form_types is None:
        form_types = ['8-K', '10-Q']
    
    all_filings = []
    
    for form_type in form_types:
        # SEC EDGAR RSS feed URLs for each form type
        feed_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type={form_type}&dateb=&owner=exclude&match=&count=100&myHID=&outputformat=atom"
        
        logger.info(f"Fetching SEC RSS feed for {form_type}...")
        xml_content = fetch_sec_rss_feed(feed_url)
        
        if xml_content:
            filings = parse_sec_rss_feed(xml_content)
            all_filings.extend(filings)
            logger.info(f"Parsed {len(filings)} {form_type} filings from SEC RSS")
    
    return all_filings


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Example: fetch latest 8-K and 10-Q filings
    filings = scrape_sec_filings(form_types=['8-K', '10-Q'])
    
    for filing in filings[:5]:
        print(f"{filing.company_name} | {filing.form_type} | {filing.filing_date}")
