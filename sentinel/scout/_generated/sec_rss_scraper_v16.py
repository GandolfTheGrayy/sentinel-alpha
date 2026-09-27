"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(CIK, company name, accession number, filing date, form type), and normalizes into
dataclasses. Integrates with the main Scout pipeline to enrich sentiment analysis
with regulatory event signals.

Uses Gemini for robust HTML parsing of SEC RSS feed entries when needed.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import xml.etree.ElementTree as ET

import requests
import google.generativeai as genai
from bs4 import BeautifulSoup

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

SEC_RSS_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
SEC_RSS_FEED_8K = f"{SEC_RSS_BASE}?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&search_text=&CIK=&filenum=&State=&SIC=0100&output=atom"
SEC_RSS_FEED_10Q = f"{SEC_RSS_BASE}?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&search_text=&CIK=&filenum=&State=&SIC=0100&output=atom"


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from EDGAR RSS feeds."""
    cik: str
    company_name: str
    accession_number: str
    filing_date: datetime
    form_type: str
    document_url: str
    raw_xml_entry: str


def fetch_rss_feed(feed_url: str, timeout: int = 10) -> Optional[str]:
    """Fetch raw XML content from SEC EDGAR RSS feed."""
    try:
        response = requests.get(feed_url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching SEC RSS feed: {e}")
        return None


def parse_rss_entry_with_gemini(entry_xml: str) -> Optional[dict]:
    """Use Gemini to parse ambiguous SEC RSS entry XML and extract structured metadata."""
    try:
        prompt = f"""
        Parse this SEC EDGAR RSS feed entry XML and extract:
        - CIK (company identifier, usually 10 digits)
        - Company name
        - Accession number (format: XXXX-XX-XXXXXX)
        - Filing date (ISO format YYYY-MM-DD)
        - Form type (e.g., 8-K, 10-Q)
        - Document URL
        
        Return ONLY valid JSON with keys: cik, company_name, accession_number, filing_date, form_type, document_url.
        If any field cannot be extracted, use null.
        
        XML Entry:
        {entry_xml}
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        import json
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini parsing failed: {e}")
        return None


def parse_rss_feed_xml(feed_xml: str) -> List[SECFiling]:
    """Parse SEC EDGAR Atom feed XML and extract filing metadata."""
    filings = []
    
    try:
        root = ET.fromstring(feed_xml)
    except ET.ParseError as e:
        print(f"XML parse error: {e}")
        return filings
    
    # SEC EDGAR uses Atom namespace
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    
    entries = root.findall("atom:entry", ns)
    
    for entry in entries:
        try:
            entry_xml_str = ET.tostring(entry, encoding="unicode")
            
            # Try standard parsing first
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            updated_elem = entry.find("atom:updated", ns)
            link_elem = entry.find("atom:link", ns)
            
            title = title_elem.text if title_elem is not None else ""
            summary = summary_elem.text if summary_elem is not None else ""
            updated = updated_elem.text if updated_elem is not None else ""
            link_href = link_elem.get("href") if link_elem is not None else ""
            
            # If parsing is incomplete, use Gemini for robust extraction
            if not (title and summary and updated):
                parsed = parse_rss_entry_with_gemini(entry_xml_str)
                if parsed:
                    filing = SECFiling(
                        cik=parsed.get("cik", ""),
                        company_name=parsed.get("company_name", ""),
                        accession_number=parsed.get("accession_number", ""),
                        filing_date=datetime.fromisoformat(parsed.get("filing_date", "")) if parsed.get("filing_date") else datetime.now(),
                        form_type=parsed.get("form_type", ""),
                        document_url=parsed.get("document_url", ""),
                        raw_xml_entry=entry_xml_str
                    )
                    filings.append(filing)
                continue
            
            # Standard parsing: extract CIK, company name, accession from title/summary
            # Title format typically: "Company Name (CIK) - Form Type - Filed Date"
            parts = title.split(" - ")
            company_info = parts[0] if parts else ""
            form_type = parts[1] if len(parts) > 1 else ""
            
            # Extract CIK from company_info (typically in parentheses)
            cik = ""
            company_name = company_info
            if "(" in company_info and ")" in company_info:
                cik = company_info.split("(")[1].split(")")[0]
                company_name = company_info.split("(")[0].strip()
            
            # Extract accession number from summary or link
            accession_number = ""
            if "Accession Number:" in summary:
                accession_number = summary.split("Accession Number:")[1].split("\n")[0].strip()
            elif link_href:
                # Accession number often in URL path
                parts = link_href.split("/")
                if len(parts) >= 4:
                    accession_number = parts[-2]
            
            # Parse filing date from updated (ISO format)
            try:
                filing_date = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                filing_date = datetime.now()
            
            filing = SECFiling(
                cik=cik,
                company_name=company_name,
                accession_number=accession_number,
                filing_date=filing_date,
                form_type=form_type.strip(),
                document_url=link_href,
                raw_xml_entry=entry_xml_str
            )
            filings.append(filing)
        
        except Exception as e:
            print(f"Error parsing individual entry: {e}")
            continue
    
    return filings


def scrape_sec_8k_filings() -> List[SECFiling]:
    """Fetch and parse latest 8-K filings from SEC EDGAR RSS feed."""
    feed_xml = fetch_rss_feed(SEC_RSS_FEED_8K)
    if not feed_xml:
        return []
    return parse_rss_feed_xml(feed_xml)


def scrape_sec_10q_filings() -> List[SECFiling]:
    """Fetch and parse latest 10-Q filings from SEC EDGAR RSS feed."""
    feed
