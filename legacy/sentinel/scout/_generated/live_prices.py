"""
Sentinel Scout: Live Price Fetcher with SQLite Storage

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data from yfinance
with automatic fallback to stooq. Stores data in SQLite with schema designed
for easy migration to TimescaleDB. Provides swap-ready interface for both
in-memory queries and persistent storage.

Used by sentinel/pipeline.py to continuously refresh price signals for
sentiment-to-price correlation analysis.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================================
# SQLite Schema & Connection Management
# ============================================================================

def init_database(db_path: str = "sentinel_prices.db") -> sqlite3.Connection:
    """Initialize SQLite database with OHLCV schema; idempotent."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
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
        )
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
        ON ohlcv(ticker, timestamp DESC)
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            last_fetch_timestamp DATETIME,
            last_fetch_source TEXT,
            record_count INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn


def get_connection(db_path: str = "sentinel_prices.db") -> sqlite3.Connection:
    """Get or create a database connection with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================================
# yfinance Fetcher (Primary Source)
# ============================================================================

def fetch_ohlcv_yfinance(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance for a single ticker; returns None on failure.
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            prepost=False
        )
        if data.empty:
            return None
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        return data
    except Exception as e:
        print(f"[yfinance] Failed to fetch {ticker}: {e}")
        return None


def fetch_ohlcv_stooq(
    ticker: str,
    period: str = "1y"
) -> Optional[pd.DataFrame]:
    """
    Fallback OHLCV fetcher using stooq API (lightweight alternative).
    """
    try:
        import requests
        from io import StringIO
        
        # stooq CSV endpoint (e.g., aapl.us)
        base_ticker = ticker.replace("-", ".")
        url = f"https://stooq.com/q/l/?s={base_ticker}&f=sd2t2ohlcv&h&e=csv"
        
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        
        data = pd.read_csv(StringIO(resp.text))
        if data.empty or "Date" not in data.columns:
            return None
        
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date").reset_index(drop=True)
        
        # Normalize column names
        data.columns = [col.lower().strip() for col in data.columns]
        return data
    except Exception as e:
        print(f"[stooq] Failed to fetch {ticker}: {e}")
        return None


def fetch_ohlcv_with_fallback(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Attempt yfinance first; fallback to stooq if yfinance fails or returns empty.
    """
    data = fetch_ohlcv_yfinance(ticker, period, interval)
    if data is not None and not data.empty:
        return data
    
    print(f"[scout] yfinance failed for {ticker}, trying stooq...")
    data = fetch_ohlcv_stooq(ticker, period)
    return data if data is not None and not data.empty else None


# ============================================================================
# Storage & Query Interface
# ============================================================================

def store_ohlcv(
    ticker: str,
    data: pd.DataFrame,
    db_path: str = "sentinel_prices.db",
    source: str = "yfinance"
) -> int:
    """
    Insert or replace OHLCV records into SQLite; return count of inserted rows.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Normalize column names
    data.columns = [col.lower() for col in data.columns]
    
    # Ensure date column is datetime
    date_col = "date" if "date" in data.columns else "datetime"
    if date_col not in data.columns:
        raise ValueError(f"Expected 'date' or 'datetime' column in data")
    
    data[date_col] = pd.to_datetime(data[date_col])
    
    inserted = 0
    for _, row in data.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv
                (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                row[date_col],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"])
            ))
            inserted += 1
        except Exception as e:
            print(f"[store] Error inserting {ticker} row {row[date_col]}: {e}")
            continue
    
    # Update metadata
    cursor.execute("""
        INSERT OR REPLACE INTO fetch_metadata
        (ticker, last_fetch_timestamp, last_fetch_source, record_count, updated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
    """, (
        ticker,
        datetime.utcnow(),
        source,
        inserted
    ))
    
    conn.commit()
    conn.close()
    
    return inserted


def query_ohlcv(
    ticker: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db_path: str = "sentinel_prices.db"
) -> pd.DataFrame:
    """
    Query stored OHLCV data; return as DataFrame (empty if no records found).
    """
    conn = get_connection(db_path)
    
    query = "SELECT * FROM ohlcv WHERE ticker = ?"
    params = [ticker]
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC"
    
    data
