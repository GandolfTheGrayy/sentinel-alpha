"""
SEC EDGAR RSS feed scraper for Sentinel Scout pillar.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing
metadata (CIK, accession number, filing date, company name), and normalizes
into dataclass structures for downstream linguistic and historical analysis.

Uses feedparser to handle RSS parsing and requests for resilience.
Integrates with the Scout data ingestion pipeline.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# SEC EDGAR RSS feed URLs
SEC_RSS_FEEDS = {
    "8-K": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=exclude&count=100&myHID=&search_text=&CIK=&company_name=&output=atom",
    "10-Q": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-Q&dateb=&owner=exclude&count=100&myHID=&search_text=&CIK=&company_name=&output=atom",
}


@dataclass
class SECFiling:
    """Normalized SEC filing metadata extracted from EDGAR RSS feed."""

    cik: str
    """Central Index Key (10-digit identifier)."""

    accession_number: str
    """Accession number (unique filing identifier)."""

    filing_type: str
    """Filing type: '8-K', '10-Q', etc."""

    company_name: str
    """Company legal name."""

    filing_date: datetime
    """Date filing was submitted to SEC."""

    document_url: str
    """Direct URL to filing document."""

    raw_feed_entry: Optional[dict] = None
    """Original RSS entry dict for debugging."""


def _get_resilient_session() -> requests.Session:
    """
    Return a requests Session with retry logic for transient failures.

    Retries up to 3 times on 5xx or connection errors with exponential backoff.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def scrape_sec_rss_feed(
    feed_type: str = "8-K", max_retries: int = 3
) -> list[SECFiling]:
    """
    Fetch and parse a single SEC EDGAR RSS feed, returning normalized filings.

    Args:
        feed_type: '8-K' or '10-Q' (key in SEC_RSS_FEEDS).
        max_retries: Number of HTTP retries before giving up.

    Returns:
        List of SECFiling dataclass instances.

    Raises:
        ValueError if feed_type not recognized.
        requests.RequestException if feed fetch fails after retries.
    """
    if feed_type not in SEC_RSS_FEEDS:
        raise ValueError(f"Unknown feed type: {feed_type}. Use '8-K' or '10-Q'.")

    feed_url = SEC_RSS_FEEDS[feed_type]
    logger.info(f"Fetching SEC {feed_type} RSS feed from {feed_url}")

    session = _get_resilient_session()
    try:
        response = session.get(feed_url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch SEC RSS feed for {feed_type}: {e}")
        raise

    # Parse Atom/RSS feed
    feed = feedparser.parse(response.content)

    if feed.bozo:
        logger.warning(f"Feed parser warning for {feed_type}: {feed.bozo_exception}")

    filings = []
    for entry in feed.entries:
        try:
            filing = _parse_sec_entry(entry, feed_type)
            if filing:
                filings.append(filing)
        except Exception as e:
            logger.warning(f"Failed to parse SEC entry: {e}")
            continue

    logger.info(f"Extracted {len(filings)} filings from {feed_type} feed.")
    return filings


def _parse_sec_entry(entry: dict, filing_type: str) -> Optional[SECFiling]:
    """
    Parse a single SEC EDGAR RSS entry into a SECFiling dataclass.

    Args:
        entry: A feedparser entry dict from an SEC RSS feed.
        filing_type: '8-K', '10-Q', etc.

    Returns:
        SECFiling instance or None if parsing fails.
    """
    try:
        # Extract CIK from entry link or summary
        # SEC Atom feeds embed CIK in the entry link: /cgi-bin/browse-edgar?action=getcompany&CIK=<CIK>&...
        link = entry.get("link", "")
        cik = _extract_cik_from_url(link)

        # Accession number often in title or summary
        title = entry.get("title", "")
        accession = _extract_accession(title, entry.get("summary", ""))

        # Company name from title (format: "CompanyName CIK Accession ...")
        company_name = _extract_company_name(title)

        # Filing date from updated field
        filing_date_str = entry.get("updated", "")
        filing_date = _parse_date(filing_date_str)

        # Document URL
        document_url = link or ""

        if not (cik and accession and company_name and filing_date):
            logger.debug(
                f"Incomplete entry: cik={cik}, accession={accession}, company={company_name}, date={filing_date}"
            )
            return None

        return SECFiling(
            cik=cik,
            accession_number=accession,
            filing_type=filing_type,
            company_name=company_name,
            filing_date=filing_date,
            document_url=document_url,
            raw_feed_entry=entry,
        )

    except Exception as e:
        logger.debug(f"Failed to parse SEC entry: {e}")
        return None


def _extract_cik_from_url(url: str) -> Optional[str]:
    """Extract 10-digit CIK from SEC EDGAR URL."""
    import re

    match = re.search(r"CIK=(\d{10})", url)
    if match:
        return match.group(1)
    return None


def _extract_accession(title: str, summary: str) -> Optional[str]:
    """Extract accession number from title or summary (format: 0001234567-89-012345)."""
    import re

    pattern = r"\d{10}-\d{2}-\d{6}"
    match = re.search(pattern, title)
    if match:
        return match.group(0)

    match = re.search(pattern, summary)
    if match:
        return match.group(0)

    return None


def _extract_company_name(title: str) -> Optional[str]:
    """Extract company name from RSS entry title (usually first token before CIK)."""
    parts = title.split()
    if parts:
        # First non-empty token is typically company name or abbreviation
        return parts[0]
    return None


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse ISO 8601 or RFC 2822 date string into datetime object."""
    if not date_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%
