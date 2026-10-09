"""
Live Price Fetcher — Sentinel Scout Module

Fetches OHLCV (Open, High, Low, Close, Volume) data for given tickers using yfinance,
stores the data in SQLite with a schema compatible with future TimescaleDB migration.
Provides a swap-ready interface: same function signatures work whether backing store
is SQLite or TimescaleDB. Designed to run on a schedule (e.g., hourly) to populate
the Sentinel historical price database.

Part of sentinel/scout/ — data ingestion layer for live market signals.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd


# ============================================================================
# SCHEMA & INITIALIZATION
# ============================================================================

DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel_prices.db")


def _init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite schema for OHLCV price data with TimescaleDB-compatible layout."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main prices table: TimescaleDB-ready (timestamp as primary ordering key).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            UNIQUE(ticker, ts)
        )
    """)
    
    # Index for fast lookups by ticker and time range.
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_ticker_ts 
        ON prices (ticker, ts DESC)
    """)
    
    # Metadata table: tracks last fetch time per ticker (avoids re-fetching).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            ticker TEXT PRIMARY KEY,
            last_fetch TIMESTAMP NOT NULL,
            record_count INTEGER DEFAULT 0
        )
    """)
    
    conn.commit()
    conn.close()


# ============================================================================
# CORE FETCHING & STORAGE
# ============================================================================

def fetch_and_store_prices(
    tickers: List[str],
    period: str = "1d",
    interval: str = "1h",
    db_path: str = DB_PATH
) -> Dict[str, int]:
    """
    Fetch OHLCV data from yfinance and store in SQLite.
    
    Args:
        tickers: List of stock tickers (e.g., ["AAPL", "MSFT"])
        period: yfinance period string ("1d", "5d", "1mo", etc.)
        interval: yfinance interval string ("1m", "5m", "1h", "1d", etc.)
        db_path: Path to SQLite database file
    
    Returns:
        Dict mapping ticker → number of rows inserted/updated
    """
    _init_db(db_path)
    results = {}
    
    for ticker in tickers:
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                prepost=False
            )
            
            if data.empty:
                results[ticker] = 0
                continue
            
            # Ensure DatetimeIndex is timezone-naive for SQLite compatibility.
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Rename columns to match our schema (yfinance uses capitalized names).
            data = data.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Insert/upsert into database.
            count = _upsert_prices(ticker, data, db_path)
            results[ticker] = count
            
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            results[ticker] = -1
    
    return results


def _upsert_prices(ticker: str, df: pd.DataFrame, db_path: str) -> int:
    """
    Upsert price records into SQLite, skipping duplicates.
    
    Args:
        ticker: Stock ticker symbol
        df: DataFrame with OHLCV columns and DatetimeIndex
        db_path: Path to SQLite database
    
    Returns:
        Number of rows inserted
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    inserted = 0
    
    for ts, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO prices 
                (ticker, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                ts.isoformat(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"])
            ))
            inserted += cursor.rowcount
        except Exception as e:
            print(f"Error inserting {ticker} @ {ts}: {e}")
    
    # Update fetch log.
    cursor.execute("""
        INSERT INTO fetch_log (ticker, last_fetch, record_count)
        VALUES (?, ?, ?)
        ON CONFLICT(ticker) DO UPDATE SET
            last_fetch = excluded.last_fetch,
            record_count = excluded.record_count
    """, (ticker, datetime.utcnow().isoformat(), inserted))
    
    conn.commit()
    conn.close()
    
    return inserted


# ============================================================================
# QUERY INTERFACE (Swap-Ready for TimescaleDB)
# ============================================================================

def get_latest_price(ticker: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """
    Retrieve the most recent OHLCV record for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        db_path: Path to SQLite database
    
    Returns:
        Dict with keys {ticker, ts, open, high, low, close, volume} or None if not found
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, ts, open, high, low, close, volume
        FROM prices
        WHERE ticker = ?
        ORDER BY ts DESC
        LIMIT 1
    """, (ticker,))
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def get_price_range(
    ticker: str,
    start_ts: datetime,
    end_ts: datetime,
    db_path: str = DB_PATH
) -> List[Dict]:
    """
    Retrieve OHLCV records within a time range.
    
    Args:
        ticker: Stock ticker symbol
        start_ts: Start timestamp (inclusive)
        end_ts: End timestamp (inclusive)
        db_path: Path to SQLite database
    
    Returns:
        List of dicts with OHLCV data, sorted by timestamp ascending
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, ts, open, high, low, close, volume
        FROM prices
        WHERE ticker = ? AND ts >= ? AND ts <= ?
        ORDER BY ts ASC
    """, (ticker, start_ts.isoformat(), end_ts.isoformat()))
    
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return rows


def get_last_n_prices(ticker: str, n: int = 100, db_path: str = DB_PATH) -> List[Dict]:
