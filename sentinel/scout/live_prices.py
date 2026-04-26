"""Scout: fetch recent OHLCV for the watchlist via yfinance."""
from __future__ import annotations

import yfinance as yf

WATCHLIST: list[str] = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN"]


def fetch_summary(tickers: list[str] | None = None, period: str = "5d") -> list[dict]:
    """Return per-ticker latest close + period % change."""
    tickers = tickers or WATCHLIST
    out: list[dict] = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(period=period, auto_adjust=True)
            if h.empty:
                continue
            first, last = float(h["Close"].iloc[0]), float(h["Close"].iloc[-1])
            pct = (last - first) / first * 100.0 if first else 0.0
            out.append({"ticker": t, "close": round(last, 2), "pct_change": round(pct, 2)})
        except Exception as exc:
            out.append({"ticker": t, "error": str(exc)[:80]})
    return out
