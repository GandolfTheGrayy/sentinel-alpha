"""Scout: aggregate recent insider (Form 4) buy/sell activity per ticker via SEC EDGAR."""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from functools import lru_cache

import requests

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def _ua() -> str | None:
    return os.environ.get("SEC_USER_AGENT") or None


def _get(url: str, ua: str) -> requests.Response:
    """HTTP GET with backoff on 429."""
    r = requests.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip"}, timeout=20)
    if r.status_code == 429:
        time.sleep(1.5)
        r = requests.get(url, headers={"User-Agent": ua}, timeout=20)
    r.raise_for_status()
    return r


@lru_cache(maxsize=1)
def _ticker_to_cik() -> dict[str, int]:
    """Return ticker → CIK map, cached per process."""
    ua = _ua()
    if not ua:
        return {}
    try:
        data = _get(TICKER_MAP_URL, ua).json()
        return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
    except Exception:
        return {}


def _parse_form4(xml_text: str) -> dict | None:
    """Extract net acquired/disposed shares + dollar value from a Form 4 primary doc."""
    try:
        root = ET.fromstring(re.sub(r"<\?xml[^>]*\?>", "", xml_text).strip())
    except ET.ParseError:
        return None
    insider = ""
    title = ""
    owner = root.find(".//reportingOwner")
    if owner is not None:
        name_el = owner.find(".//reportingOwnerId/rptOwnerName")
        if name_el is not None and name_el.text:
            insider = name_el.text.strip()
        title_el = owner.find(".//reportingOwnerRelationship/officerTitle")
        if title_el is not None and title_el.text:
            title = title_el.text.strip()
    shares_acquired = 0.0
    shares_disposed = 0.0
    value = 0.0
    for tx in root.findall(".//nonDerivativeTransaction"):
        try:
            sh = float(tx.findtext(".//transactionShares/value") or 0)
            price = float(tx.findtext(".//transactionPricePerShare/value") or 0)
            code = (tx.findtext(".//transactionAcquiredDisposedCode/value") or "").upper()
            if code == "A":
                shares_acquired += sh
                value += sh * price
            elif code == "D":
                shares_disposed += sh
                value -= sh * price
        except (ValueError, TypeError):
            continue
    return {
        "insider": insider,
        "title": title,
        "shares_acquired": shares_acquired,
        "shares_disposed": shares_disposed,
        "net_value": round(value, 2),
    }


def recent_insider_activity(ticker: str, lookback_days: int = 60, max_filings: int = 15) -> dict | None:
    """Return aggregated insider Form 4 activity in last `lookback_days` for ticker."""
    ua = _ua()
    if not ua:
        return None
    cik = _ticker_to_cik().get(ticker.upper())
    if not cik:
        return None
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        sub = _get(SUBMISSIONS_URL.format(cik=cik), ua).json()
        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filings: list[dict] = []
        for i, form in enumerate(forms):
            if form != "4":
                continue
            filed = recent["filingDate"][i]
            if filed < cutoff:
                continue
            filings.append({
                "filed": filed,
                "accession": recent["accessionNumber"][i],
                "primary_doc": recent["primaryDocument"][i],
            })
            if len(filings) >= max_filings:
                break
    except Exception:
        return None

    transactions: list[dict] = []
    for f in filings:
        try:
            url = ARCHIVE_URL.format(cik=cik, acc_nodash=f["accession"].replace("-", ""), doc=f["primary_doc"])
            xml_text = _get(url, ua).text
            parsed = _parse_form4(xml_text)
            if parsed:
                transactions.append({**parsed, "filed": f["filed"]})
        except Exception:
            continue

    if not transactions:
        return {
            "ticker": ticker.upper(),
            "lookback_days": lookback_days,
            "filings_count": 0,
            "net_value": 0.0,
            "buys": 0,
            "sells": 0,
            "transactions": [],
            "sentiment": "neutral",
        }

    buys = sum(1 for t in transactions if t["shares_acquired"] > t["shares_disposed"])
    sells = sum(1 for t in transactions if t["shares_disposed"] > t["shares_acquired"])
    net_value = sum(t["net_value"] for t in transactions)
    sentiment = "bullish" if net_value > 100_000 else "bearish" if net_value < -100_000 else "neutral"
    return {
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "filings_count": len(transactions),
        "net_value": round(net_value, 2),
        "buys": buys,
        "sells": sells,
        "transactions": transactions[:5],
        "sentiment": sentiment,
    }
