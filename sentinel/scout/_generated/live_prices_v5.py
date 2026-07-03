"""
Live Price Fetcher for Sentinel Sentiment Engine.

Fetches OHLCV (Open, High, Low, Close, Volume) data from yfinance with
stooq fallback. Stores snapshots in SQLite with schema designed for easy
migration to TimescaleDB. Part of the Scout pillar for real-time market
data ingestion.

Public interface:
  - fetch_ohlcv() → dict of ticker → DataFrame
  - store_snapshot() → writes to SQLite with timestamp
  - get_latest_prices() → retrieves cached prices for downstream analysis
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import yfinance as yf
import pandas as pd
import requests


def _init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite schema for OHLCV storage with TimescaleDB compatibility."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main OHLCV table: designed for easy migration to TimescaleDB hypertable
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp)
        )
    """)
    
    # Index for efficient time-range queries (mimics TimescaleDB chunk alignment)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_time 
        ON ohlcv_snapshots(ticker, timestamp DESC)
    """)
    
    # Metadata table: tracks last fetch per ticker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_metadata (
            ticker TEXT PRIMARY KEY,
            last_fetch DATETIME,
            status TEXT,
            error_message TEXT
        )
    """)
    
    conn.commit()
    return conn


def fetch_ohlcv(
    tickers: List[str],
    period: str = "1d",
    interval: str = "1d",
    retries: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance with automatic fallback to stooq on failure.
    
    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "GOOGL"])
        period: Data period (default "1d" for most recent day)
        interval: Candle interval (default "1d" for daily)
        retries: Number of fallback attempts before giving up
    
    Returns:
        Dict mapping ticker → DataFrame with columns [Open, High, Low, Close, Volume]
    """
    results = {}
    
    # Try yfinance first
    try:
        data = yf.download(
            " ".join(tickers),
            period=period,
            interval=interval,
            progress=False
        )
        
        # Handle single ticker vs. multiple tickers response structure
        if len(tickers) == 1:
            results[tickers[0]] = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        else:
            for ticker in tickers:
                if ticker in data.columns.get_level_values(0):
                    results[ticker] = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
        
        return results
    except Exception as e:
        print(f"[yfinance] Failed: {e}. Attempting stooq fallback...")
    
    # Fallback to stooq
    for ticker in tickers:
        for attempt in range(retries):
            try:
                df = _fetch_stooq(ticker, period)
                results[ticker] = df
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"[stooq] {ticker} failed after {retries} attempts: {e}")
                    results[ticker] = pd.DataFrame()
    
    return results


def _fetch_stooq(ticker: str, period: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV from stooq API as fallback source.
    
    Args:
        ticker: Ticker symbol
        period: Period string (converted to stooq interval code)
    
    Returns:
        DataFrame with OHLCV columns
    """
    interval_map = {
        "1m": "5",
        "5m": "5",
        "15m": "15",
        "1h": "60",
        "1d": "d",
        "1wk": "w",
        "1mo": "m"
    }
    stooq_interval = interval_map.get(period, "d")
    
    url = f"https://stooq.com/q/l/?s={ticker.upper()}&f=sd2t2ohlcv&h&e=csv"
    
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(
            pd.io.common.StringIO(resp.text),
            parse_dates=["Date"],
            index_col="Date"
        )
        df.columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
        return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    except Exception as e:
        raise ValueError(f"stooq fetch failed for {ticker}: {e}")


def store_snapshot(
    db_path: str,
    tickers: List[str],
    ohlcv_data: Dict[str, pd.DataFrame],
    timestamp: Optional[datetime] = None
) -> None:
    """
    Store OHLCV snapshots in SQLite with TimescaleDB-compatible schema.
    
    Args:
        db_path: Path to SQLite database file
        tickers: List of ticker symbols
        ohlcv_data: Dict mapping ticker → DataFrame with OHLCV columns
        timestamp: Override timestamp (defaults to now)
    """
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    conn = _init_db(db_path)
    cursor = conn.cursor()
    
    for ticker in tickers:
        if ticker not in ohlcv_data or ohlcv_data[ticker].empty:
            continue
        
        df = ohlcv_data[ticker]
        
        # Insert most recent row
        if len(df) > 0:
            latest = df.iloc[-1]
            try:
                cursor.execute("""
                    INSERT INTO ohlcv_snapshots
                    (ticker, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    timestamp.isoformat(),
                    float(latest["Open"]) if pd.notna(latest["Open"]) else None,
                    float(latest["High"]) if pd.notna(latest["High"]) else None,
                    float(latest["Low"]) if pd.notna(latest["Low"]) else None,
                    float(latest["Close"]) if pd.notna(latest["Close"]) else None,
                    int(latest["Volume"]) if pd.notna(latest["Volume"]) else None,
                ))
                
                # Update fetch metadata
                cursor.execute("""
                    INSERT INTO fetch_metadata (ticker, last_fetch, status)
                    VALUES (?, ?, 'success')
                    ON CONFLICT(ticker) DO UPDATE SET
                        last_fetch = excluded.last_fetch,
                        status = excluded.status
                """, (ticker, timestamp.isoformat()))
            except sqlite3.IntegrityError:
                # Duplicate snapshot for this ticker at this timestamp — skip
                pass
    
    conn.commit()
    conn.close()


def get_latest_prices(
