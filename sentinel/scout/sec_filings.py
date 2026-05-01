"""Scout: pull most-recent 8-K / 10-Q filing text per ticker from SEC EDGAR.

SEC requires a User-Agent header with contact info. Set SEC_USER_AGENT env var
(e.g. "Jane Doe jane@example.com") or this module returns None.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache

import requests
from bs4 import BeautifulSoup

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
FORMS = {"8-K", "10-Q", "10-K"}
MAX_TEXT_CHARS = 4000


def _ua() -> str | None:
    """Return SEC User-Agent or None if unset."""
    return os.environ.get("SEC_USER_AGENT") or None


def _get(url: str, ua: str) -> requests.Response:
    """HTTP GET with SEC-friendly headers and basic backoff."""
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"}, timeout=20)
    if r.status_code == 429:
        time.sleep(1.5)
        r = requests.get(url, headers={"User-Agent": ua}, timeout=20)
    r.raise_for_status()
    return r


@lru_cache(maxsize=1)
def _ticker_to_cik() -> dict[str, int]:
    """Build ticker → CIK map (cached for run lifetime)."""
    ua = _ua()
    if not ua:
        return {}
    try:
        data = _get(TICKER_MAP_URL, ua).json()
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    except Exception:
        return {}


def latest_filing(ticker: str) -> dict | None:
    """Return {form, filed, accession, primary_doc, text} for most recent 8-K/10-Q/10-K."""
    ua = _ua()
    if not ua:
        return None
    cik = _ticker_to_cik().get(ticker.upper())
    if not cik:
        return None
    try:
        sub = _get(SUBMISSIONS_URL.format(cik=cik), ua).json()
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, form in enumerate(forms):
            if form not in FORMS:
                continue
            acc = recent["accessionNumber"][i]
            doc = recent["primaryDocument"][i]
            filed = recent["filingDate"][i]
            url = ARCHIVE_URL.format(cik=cik, acc_nodash=acc.replace("-", ""), doc=doc)
            try:
                raw = _get(url, ua).text
                text = _strip(raw)
            except Exception:
                text = ""
            return {"form": form, "filed": filed, "accession": acc, "primary_doc": doc, "url": url, "text": text[:MAX_TEXT_CHARS]}
    except Exception:
        return None
    return None


def _strip(html: str) -> str:
    """Strip tags + whitespace from filing HTML/text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text)
