"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing metadata
(CIK, accession number, company name, filing type, date), and normalizes into
dataclasses for downstream Historian ingestion and Linguist analysis.

Uses Gemini (google-generativeai) for high-volume RSS/XML parsing per LLM policy.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import xml.etree.ElementTree as ET

import requests
from google.generativeai import GenerativeModel


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from EDGAR RSS feed."""

    cik: str
    accession_number: str
    company_name: str
    filing_type: str
    filed_date: datetime
    report_date: Optional[datetime]
    document_url: str
    raw_xml: str

    def __repr__(self) -> str:
        return (
            f"SECFiling(cik={self.cik}, type={self.filing_type}, "
            f"company={self.company_name}, date={self.filed_date.date()})"
        )


def fetch_sec_rss_feed(feed_type: str = "8k") -> str:
    """
    Fetch raw SEC EDGAR RSS feed for 8-K or 10-Q filings.

    Args:
        feed_type: Either "8k" or "10q" (case-insensitive).

    Returns:
        Raw XML string from SEC RSS endpoint.

    Raises:
        ValueError: If feed_type not recognized.
        requests.RequestException: On network failure.
    """
    feed_type = feed_type.lower()
    if feed_type not in ("8k", "10q"):
        raise ValueError(f"feed_type must be '8k' or '10q', got {feed_type!r}")

    form_type = "8-K" if feed_type == "8k" else "10-Q"
    url = f"https://www.sec.gov/rss/edgar/feeds/{form_type}.xml"

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.text


def parse_sec_rss_with_gemini(xml_content: str, feed_type: str = "8k") -> list[SECFiling]:
    """
    Parse SEC EDGAR RSS XML using Gemini for robust extraction.

    Gemini handles malformed XML, special characters, and ambiguous date formats
    more reliably than naive regex. Returns structured filing metadata.

    Args:
        xml_content: Raw SEC RSS XML string.
        feed_type: Either "8k" or "10q" for context.

    Returns:
        List of SECFiling dataclass instances.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var not set")

    model = GenerativeModel("gemini-1.5-flash-latest")

    prompt = f"""
You are parsing SEC EDGAR RSS feeds. Extract ALL filings from this XML.

For each <item> in the feed, extract:
1. CIK (from link or accession number pattern)
2. Accession Number (e.g., "0000950144-24-001234")
3. Company Name (from title or description)
4. Filing Type ("{feed_type.upper()}")
5. Filed Date (ISO 8601 format, e.g., "2024-01-15")
6. Report Date if present (ISO 8601 or "unknown")
7. Document URL (full URL to filing)

Output as JSON array:
[
  {{
    "cik": "...",
    "accession_number": "...",
    "company_name": "...",
    "filing_type": "{feed_type.upper()}",
    "filed_date": "YYYY-MM-DD",
    "report_date": "YYYY-MM-DD",
    "document_url": "..."
  }}
]

XML to parse:
{xml_content[:8000]}
"""

    response = model.generate_content(prompt)
    raw_json = response.text

    filings = []
    try:
        import json

        data = json.loads(raw_json)
        if not isinstance(data, list):
            data = [data]

        for item in data:
            try:
                filed_dt = datetime.fromisoformat(item.get("filed_date", ""))
            except (ValueError, TypeError):
                filed_dt = datetime.now()

            report_date_str = item.get("report_date", "unknown")
            try:
                report_dt = (
                    datetime.fromisoformat(report_date_str)
                    if report_date_str != "unknown"
                    else None
                )
            except ValueError:
                report_dt = None

            filing = SECFiling(
                cik=item.get("cik", ""),
                accession_number=item.get("accession_number", ""),
                company_name=item.get("company_name", ""),
                filing_type=item.get("filing_type", feed_type.upper()),
                filed_date=filed_dt,
                report_date=report_dt,
                document_url=item.get("document_url", ""),
                raw_xml=xml_content,
            )
            if filing.cik and filing.accession_number:
                filings.append(filing)

    except json.JSONDecodeError:
        filings = _fallback_regex_parse(xml_content, feed_type)

    return filings


def _fallback_regex_parse(xml_content: str, feed_type: str = "8k") -> list[SECFiling]:
    """
    Fallback naive regex parser for SEC RSS when Gemini fails or times out.

    Args:
        xml_content: Raw SEC RSS XML string.
        feed_type: Either "8k" or "10q".

    Returns:
        List of SECFiling dataclass instances.
    """
    filings = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return filings

    ns = {
        "": "http://www.w3.org/2005/Atom",
        "content": "http://purl.org/rss/1.0/modules/content/",
    }

    for item in root.findall(".//item", ns) or root.findall(".//item"):
        title_elem = item.find("title")
        link_elem = item.find("link")
        pub_date_elem = item.find("pubDate")
        desc_elem = item.find("description")

        title = title_elem.text if title_elem is not None else ""
        link = link_elem.text if link_elem is not None else ""
        pub_date_str = pub_date_elem.text if pub_date_elem is not None else ""
        desc = desc_elem.text if desc_elem is not None else ""

        cik_match = re.search(r"CIK=(\d+)", link or "")
        cik = cik_match.group(1) if cik_match else ""

        acc_match = re.search(r"(\d{10}-\d{2}-\d{6})", desc or title or "")
        accession_number = acc_match.group(1) if acc_match else ""

        company_name = title.split("|")[0].strip() if title else ""

        try:
            filed_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        except (ValueError, TypeError):
            try:
                filed_date = datetime.fromisoformat(pub_date_str)
            except (ValueError, TypeError):
                filed_date = datetime.now()

        if cik and accession_number:
            filing = SECFiling(
                cik=cik,
                accession_number=accession_number,
