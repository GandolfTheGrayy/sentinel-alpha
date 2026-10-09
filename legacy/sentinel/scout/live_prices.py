"""Scout: fetch recent OHLCV per ticker via yfinance with stooq fallback."""
from __future__ import annotations

import csv
import io

import requests
import yfinance as yf

WATCHLIST: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN"]
STOOQ_URL = "https://stooq.com/q/d/l/?s={sym}&i=d"


def _stooq(ticker: str) -> dict | None:
    """Fallback: fetch recent daily closes from stooq CSV."""
    try:
        r = requests.get(STOOQ_URL.format(sym=ticker.lower() + ".us"), timeout=10)
        r.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(r.text)))
        if len(rows) < 6:
            return None
        recent = rows[-6:]
        first = float(recent[0]["Close"])
        last = float(recent[-1]["Close"])
        if not first:
            return None
        return {"ticker": ticker, "close": round(last, 2), "pct_change": round((last - first) / first * 100.0, 2), "source": "stooq"}
    except Exception:
        return None


def _yfinance(ticker: str, period: str) -> dict | None:
    """Primary: yfinance Ticker history."""
    try:
        h = yf.Ticker(ticker).history(period=period, auto_adjust=True)
        if h.empty:
            return None
        first, last = float(h["Close"].iloc[0]), float(h["Close"].iloc[-1])
        pct = (last - first) / first * 100.0 if first else 0.0
        return {"ticker": ticker, "close": round(last, 2), "pct_change": round(pct, 2), "source": "yfinance"}
    except Exception:
        return None


def fetch_summary(tickers: list[str] | None = None, period: str = "5d") -> list[dict]:
    """Return per-ticker close + period % change, with yfinance→stooq fallback."""
    tickers = tickers or WATCHLIST
    out: list[dict] = []
    for t in tickers:
        rec = _yfinance(t, period) or _stooq(t)
        out.append(rec or {"ticker": t, "error": "all sources failed"})
    return out
