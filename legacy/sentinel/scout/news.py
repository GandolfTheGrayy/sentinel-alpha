"""Scout: fetch latest news headline per ticker via yfinance."""
from __future__ import annotations

import yfinance as yf


def latest_headline(ticker: str) -> dict | None:
    """Return {'title','publisher'} for the most recent news item, or None."""
    try:
        items = yf.Ticker(ticker).news or []
        for it in items:
            c = it.get("content") if isinstance(it.get("content"), dict) else it
            title = c.get("title") or it.get("title") or ""
            if not title:
                continue
            pub = ""
            p = c.get("provider")
            if isinstance(p, dict):
                pub = p.get("displayName", "") or ""
            if not pub:
                pub = it.get("publisher", "") or ""
            return {"title": title.strip(), "publisher": pub.strip()}
    except Exception:
        pass
    return None
