"""
SEC EDGAR RSS feed scraper for Sentinel Scout.

Polls the SEC EDGAR RSS feeds for 8-K and 10-Q filings, extracts filing
metadata (accession number, company CIK, filing date, form type), and
normalizes into dataclass objects for downstream Linguist and Historian
analysis. Uses feedparser to parse Atom feeds and beautifulsoup4 to extract
structured data from filing summaries.

Integrated into the Scout pillar as a real-time signal source alongside
live prices and news headlines.
"""

import dataclasses
import datetime
import re
from typing import Optional
import sqlite3
import feedparser
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup


@dataclasses.dataclass
class SECFiling:
    """Normalized SEC EDGAR filing metadata extracted from RSS."""

    accession_number: str
    """Filing accession number (e.g., 0001234567-24-000001)."""

    cik: str
    """Central Index Key (company identifier)."""

    company_name: str
    """Official company name from SEC."""

    ticker: Optional[str]
    """Stock ticker symbol (best-effort extraction)."""

    form_type: str
    """Filing form type (8-K, 10-Q, 10-K, S-1, etc.)."""

    filing_date: datetime.date
    """Date the filing was submitted to SEC."""

    period_end: Optional[datetime.date]
    """End of reporting period (for periodic filings)."""

    document_url: str
    """Full URL to the filing document on SEC EDGAR."""

    summary: Optional[str]
    """Brief filing summary or item descriptions."""

    ingested_at: datetime.datetime
    """Timestamp when this filing was ingested by Sentinel."""


def parse_sec_rss_feed(feed_url: str, form_types: Optional[list[str]] = None) -> list[SECFiling]:
    """
    Parse an SEC EDGAR RSS feed and extract filing metadata.

    Args:
        feed_url: URL to the SEC EDGAR RSS feed (e.g., 8-K or 10-Q feed).
        form_types: Optional list of form types to filter (e.g., ['8-K', '10-Q']).
                   If None, all entries are returned.

    Returns:
        List of normalized SECFiling dataclass instances.
    """
    filings: list[SECFiling] = []

    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        print(f"Error parsing RSS feed {feed_url}: {e}")
        return filings

    for entry in feed.get("entries", []):
        try:
            filing = _extract_filing_from_entry(entry)
            if filing is None:
                continue

            if form_types and filing.form_type not in form_types:
                continue

            filings.append(filing)
        except Exception as e:
            print(f"Error extracting filing from entry: {e}")
            continue

    return filings


def _extract_filing_from_entry(entry: dict) -> Optional[SECFiling]:
    """
    Extract SECFiling from a single feedparser entry.

    Args:
        entry: A feedparser entry dict from an SEC RSS feed.

    Returns:
        A SECFiling instance or None if extraction fails.
    """
    try:
        title = entry.get("title", "")
        link = entry.get("link", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")

        form_type, company_name, accession_number, cik = _parse_title(title)
        if not (form_type and company_name and accession_number and cik):
            return None

        filing_date = _parse_date(published)
        if filing_date is None:
            return None

        period_end = _extract_period_end(summary)
        ticker = _extract_ticker_from_summary(summary)

        return SECFiling(
            accession_number=accession_number,
            cik=cik,
            company_name=company_name,
            ticker=ticker,
            form_type=form_type,
            filing_date=filing_date,
            period_end=period_end,
            document_url=link,
            summary=summary[:500] if summary else None,
            ingested_at=datetime.datetime.utcnow(),
        )
    except Exception:
        return None


def _parse_title(title: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Parse SEC RSS feed title to extract form type, company name, accession, and CIK.

    Typical format: "8-K Company Name CIK: 0001234567 Accession Number: 0001234567-24-000001"

    Args:
        title: Raw title string from RSS entry.

    Returns:
        Tuple of (form_type, company_name, accession_number, cik).
    """
    form_type = None
    company_name = None
    cik = None
    accession_number = None

    form_match = re.search(r"^(\d+-[KQA])", title)
    if form_match:
        form_type = form_match.group(1)

    cik_match = re.search(r"CIK:\s*(\d+)", title)
    if cik_match:
        cik = cik_match.group(1).lstrip("0") or "0"

    accession_match = re.search(r"Accession Number:\s*([0-9\-]+)", title)
    if accession_match:
        accession_number = accession_match.group(1)

    if form_type and accession_number:
        company_part = title.split("CIK:")[0].strip()
        if company_part.startswith(form_type):
            company_name = company_part[len(form_type):].strip()

    return form_type, company_name, accession_number, cik


def _parse_date(date_str: str) -> Optional[datetime.date]:
    """
    Parse ISO 8601 date string into datetime.date.

    Args:
        date_str: Date string (typically ISO 8601 format from RSS).

    Returns:
        datetime.date instance or None if parsing fails.
    """
    if not date_str:
        return None

    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.date()
    except (ValueError, AttributeError):
        try:
            dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
            return dt.date()
        except (ValueError, TypeError):
            return None


def _extract_period_end(summary: str) -> Optional[datetime.date]:
    """
    Extract period end date from filing summary text.

    Looks for patterns like "Period ending" or similar in the summary.

    Args:
        summary: Filing summary text.

    Returns:
        datetime.date or None if not found.
    """
    if not summary:
        return None

    patterns = [
        r"(?:For the (?:six|nine) months|period) ending (\d{4}-\d{2}-\d{2})",
        r"(?:As of|Ended) (\d{4}-\d{2}-\d{2})",
        r"(\d{4}-\d{2}-\d{2})",
    ]

    for pattern in patterns:
        match = re.search(pattern, summary, re.IGNORECASE)
        if match:
            try:
                return datetime.datetime.strptime(match.group(1), "%Y-%m-%d").date()
            except (ValueError, IndexError):
                continue

    return None


def _extract_ticker_from_summary(summary: str) -> Optional[str]:
    """
    Best-effort extraction of stock ticker from filing summary.

    Looks for patterns like [TICKER] or trading under symbol.

    Args:
        summary: Filing summary text.

    Returns:
