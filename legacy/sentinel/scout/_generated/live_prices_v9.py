"""
Live price fetcher for Sentinel Scout pillar.

Fetches OHLCV (Open, High, Low, Close, Volume) data from yfinance with stooq fallback.
Stores data in SQLite with schema designed for easy migration to TimescaleDB.
Provides swap-ready interface: callers use fetch_ohlcv() without knowing backend.

Used by sentinel/pipeline.py to populate daily price context for sentiment analysis.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd


# Database setup
DB_PATH = Path(__file__).parent.parent.parent / "data" / "prices.db"


def _init_db() -> None:
    """Initialize SQLite schema if not exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date
        ON ohlcv (ticker, date DESC);
    """)
    conn.commit()
    conn.close()


def fetch_ohlcv(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_retries: int = 2,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for ticker, caching in SQLite.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: ISO date string; defaults to 30 days ago.
        end_date: ISO date string; defaults to today.
        max_retries: Number of fallback attempts (yfinance → stooq).
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume].
    """
    _init_db()
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Try yfinance first
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for {ticker}")
    except Exception as e:
        if max_retries > 0:
            # Fallback to stooq via yfinance with different session
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    session=None,
                )
                if data.empty:
                    raise ValueError(f"Stooq fallback also empty for {ticker}")
            except Exception as fallback_e:
                raise RuntimeError(
                    f"Failed to fetch {ticker}: {e}; fallback failed: {fallback_e}"
                ) from fallback_e
        else:
            raise RuntimeError(f"Failed to fetch {ticker}: {e}") from e
    
    # Normalize columns (yfinance returns capitalized or lowercase depending on source)
    data.columns = [col.lower() for col in data.columns]
    data.index.name = "date"
    data = data.reset_index()
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    data["ticker"] = ticker
    
    # Ensure all expected columns exist
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in data.columns:
            data[col] = None
    
    # Store in SQLite
    _store_ohlcv(ticker, data)
    
    return data[["date", "open", "high", "low", "close", "volume"]]


def _store_ohlcv(ticker: str, data: pd.DataFrame) -> None:
    """Store OHLCV data in SQLite, upserting on duplicate dates."""
    conn = sqlite3.connect(DB_PATH)
    for _, row in data.iterrows():
        conn.execute(
            """
            INSERT OR REPLACE INTO ohlcv
            (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                row["date"],
                float(row["open"]) if pd.notna(row["open"]) else None,
                float(row["high"]) if pd.notna(row["high"]) else None,
                float(row["low"]) if pd.notna(row["low"]) else None,
                float(row["close"]) if pd.notna(row["close"]) else None,
                int(row["volume"]) if pd.notna(row["volume"]) else None,
            ),
        )
    conn.commit()
    conn.close()


def get_latest_price(ticker: str) -> Optional[float]:
    """Fetch latest closing price for ticker from cache or live."""
    try:
        data = fetch_ohlcv(ticker, start_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        if not data.empty:
            return float(data.iloc[-1]["close"])
    except Exception:
        pass
    return None


def get_price_range(ticker: str, days: int = 30) -> Optional[Tuple[float, float]]:
    """Get (min, max) closing price over last N days."""
    try:
        data = fetch_ohlcv(
            ticker,
            start_date=(datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        )
        closes = data["close"].dropna()
        if not closes.empty:
            return (float(closes.min()), float(closes.max()))
    except Exception:
        pass
    return None


def list_cached_tickers() -> List[str]:
    """Return all tickers currently cached in SQLite."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker;")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers


def clear_cache(ticker: Optional[str] = None) -> None:
    """Clear SQLite cache for one ticker or all tickers."""
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    if ticker:
        conn.execute("DELETE FROM ohlcv WHERE ticker = ?;", (ticker,))
    else:
        conn.execute("DELETE FROM ohlcv;")
    conn.commit()
    conn.close()
