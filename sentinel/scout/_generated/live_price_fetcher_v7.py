"""
Live price fetcher for Sentinel Scout pillar.

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data via yfinance
and persists to SQLite with a schema designed for easy migration to TimescaleDB.
Provides swap-ready interface: queries return dicts that work whether backing
store is SQLite or TimescaleDB.

Used by sentinel/pipeline.py to populate the historian with current market data
before sentiment analysis and prediction.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import yfinance as yf
import pandas as pd


# Schema version for migrations
SCHEMA_VERSION = 1


def _ensure_db_initialized(db_path: str) -> None:
    """Initialize SQLite schema if not present."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create prices table with TimescaleDB-compatible schema
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume INTEGER NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, timestamp)
    );
    """)
    
    # Index for common query patterns
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_prices_ticker_timestamp 
    ON prices(ticker, timestamp DESC);
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_prices_ticker 
    ON prices(ticker);
    """)
    
    # Metadata table for schema tracking
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS schema_meta (
        id INTEGER PRIMARY KEY,
        version INTEGER NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    # Check if schema_meta is populated
    cursor.execute("SELECT COUNT(*) FROM schema_meta;")
    if cursor.fetchone()[0] == 0:
        cursor.execute(
            "INSERT INTO schema_meta (version) VALUES (?);",
            (SCHEMA_VERSION,)
        )
    
    conn.commit()
    conn.close()


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    db_path: str = "sentinel.db",
    force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch OHLCV data for a ticker and store in SQLite.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL", "TSLA").
        period: Historical period ("1d", "5d", "1mo", "3mo", "6mo", "1y", etc.).
        interval: Candle interval ("1m", "5m", "15m", "30m", "60m", "1d", "1wk", "1mo").
        db_path: SQLite database file path.
        force_refresh: If True, ignore cached data and re-fetch from yfinance.
    
    Returns:
        List of dicts with keys: ticker, timestamp, open, high, low, close, volume.
    """
    _ensure_db_initialized(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Fetch from yfinance
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        conn.close()
        return []
    
    if data.empty:
        conn.close()
        return []
    
    # Reset index to get date as a column
    data.reset_index(inplace=True)
    data.columns = [col.lower() for col in data.columns]
    
    rows = []
    for _, row in data.iterrows():
        timestamp = row["date"]
        if pd.api.types.is_datetime64_any_dtype(timestamp):
            timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp = str(timestamp)
        
        record = {
            "ticker": ticker,
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"])
        }
        rows.append(record)
        
        # Insert or replace into SQLite
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO prices 
            (ticker, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (
                record["ticker"],
                record["timestamp"],
                record["open"],
                record["high"],
                record["low"],
                record["close"],
                record["volume"]
            ))
        except sqlite3.IntegrityError:
            # Duplicate or constraint error; skip
            pass
    
    conn.commit()
    conn.close()
    
    return rows


def get_latest_price(ticker: str, db_path: str = "sentinel.db") -> Optional[Dict[str, Any]]:
    """
    Retrieve the most recent OHLCV record for a ticker from SQLite.
    
    Args:
        ticker: Stock symbol.
        db_path: SQLite database file path.
    
    Returns:
        Dict with keys: ticker, timestamp, open, high, low, close, volume,
        or None if no data exists.
    """
    _ensure_db_initialized(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT ticker, timestamp, open, high, low, close, volume
    FROM prices
    WHERE ticker = ?
    ORDER BY timestamp DESC
    LIMIT 1;
    """, (ticker,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row is None:
        return None
    
    return {
        "ticker": row["ticker"],
        "timestamp": row["timestamp"],
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"]
    }


def get_price_range(
    ticker: str,
    start_date: str,
    end_date: str,
    db_path: str = "sentinel.db"
) -> List[Dict[str, Any]]:
    """
    Retrieve OHLCV records for a ticker within a date range.
    
    Args:
        ticker: Stock symbol.
        start_date: ISO format start date (e.g., "2024-01-01").
        end_date: ISO format end date (e.g., "2024-12-31").
        db_path: SQLite database file path.
    
    Returns:
        List of dicts with keys: ticker, timestamp, open, high, low, close, volume.
    """
    _ensure_db_initialized(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT ticker, timestamp, open, high, low, close, volume
    FROM prices
    WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
    ORDER BY timestamp ASC;
    """, (ticker, start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "ticker": row["ticker"],
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
