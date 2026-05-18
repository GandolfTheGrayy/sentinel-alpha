"""Scout: broad market news (not ticker-specific) from Finnhub `category=general`."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

URL = "https://finnhub.io/api/v1/news"


def fetch_general(limit: int = 40) -> list[dict]:
    """Return up to `limit` recent general-market news items."""
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    try:
        r = requests.get(URL, params={"category": "general", "token": key}, timeout=15)
        r.raise_for_status()
        items = r.json() or []
    except Exception:
        return []
    out: list[dict] = []
    for it in items[:limit * 2]:
        headline = (it.get("headline") or "").strip()
        if not headline:
            continue
        ts = it.get("datetime") or 0
        when = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
        out.append({
            "headline": headline,
            "summary": (it.get("summary") or "")[:300],
            "source": it.get("source") or "",
            "category": it.get("category") or "",
            "when": when[:16],
            "url": it.get("url") or "",
        })
        if len(out) >= limit:
            break
    return out


def fetch_company(ticker: str, lookback_days: int = 7, limit: int = 5) -> list[dict]:
    """Return ticker-specific recent company news for exit-evaluation context."""
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return []
    from datetime import date, timedelta
    today = date.today().isoformat()
    since = (date.today() - timedelta(days=lookback_days)).isoformat()
    try:
        r = requests.get(URL.replace("/news", "/company-news"), params={"symbol": ticker, "from": since, "to": today, "token": key}, timeout=15)
        r.raise_for_status()
        items = r.json() or []
    except Exception:
        return []
    return [{
        "headline": (it.get("headline") or "").strip(),
        "source": it.get("source") or "",
        "summary": (it.get("summary") or "")[:240],
    } for it in items[:limit] if it.get("headline")]
