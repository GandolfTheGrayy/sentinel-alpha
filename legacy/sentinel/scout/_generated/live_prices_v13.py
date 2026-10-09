"""
Live price fetcher for Sentinel Scout — acquires OHLCV data from yfinance
with SQLite storage and TimescaleDB-compatible schema for swappable backends.

This module provides real-time and historical OHLC candle ingestion, storing
normalized price data in a swap-ready relational format. The schema supports
future migration to TimescaleDB without refactoring downstream consumers.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json

import yfinance as yf
import pandas as pd


# ============================================================================
# SQLite Schema & Connection Management
# ============================================================================

def init_db(db_path: str = "sentinel_prices.db") -> sqlite3.Connection:
    """Initialize SQLite database with OHLCV schema compatible with TimescaleDB migration."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_candles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            adjusted_close REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON price_candles(ticker, timestamp DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON price_candles(timestamp DESC)"
    )
    conn.commit()
    return conn


def get_db(db_path: str = "sentinel_prices.db") -> sqlite3.Connection:
    """Get or initialize SQLite connection."""
    if not os.path.exists(db_path):
        return init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ============================================================================
# Live Price Ingestion
# ============================================================================

def fetch_and_store_live_price(
    ticker: str,
    db_path: str = "sentinel_prices.db",
    period: str = "1d",
    interval: str = "1h"
) -> Tuple[bool, Optional[str]]:
    """
    Fetch latest OHLCV candle for ticker via yfinance and store in SQLite.
    
    Returns (success, error_message).
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if data.empty:
            return False, f"No data returned for {ticker}"
        
        # Handle single vs. multi-candle result
        if isinstance(data.index, pd.DatetimeIndex):
            # Multi-candle: take the last row
            row = data.iloc[-1]
            timestamp = data.index[-1]
        else:
            row = data
            timestamp = datetime.utcnow()
        
        conn = get_db(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            INSERT OR REPLACE INTO price_candles
            (ticker, timestamp, open, high, low, close, volume, adjusted_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                timestamp,
                float(row.get("Open", 0)),
                float(row.get("High", 0)),
                float(row.get("Low", 0)),
                float(row.get("Close", 0)),
                int(row.get("Volume", 0)),
                float(row.get("Adj Close", row.get("Close", 0)))
            )
        )
        conn.commit()
        conn.close()
        return True, None
    
    except Exception as e:
        return False, str(e)


def fetch_batch_prices(
    tickers: List[str],
    db_path: str = "sentinel_prices.db",
    period: str = "5d",
    interval: str = "1h"
) -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Fetch and store OHLCV for multiple tickers. Returns status dict {ticker: (success, error)}.
    """
    results = {}
    for ticker in tickers:
        success, error = fetch_and_store_live_price(ticker, db_path, period, interval)
        results[ticker] = (success, error)
    return results


def fetch_historical_range(
    ticker: str,
    start_date: str,
    end_date: str,
    db_path: str = "sentinel_prices.db",
    interval: str = "1d"
) -> Tuple[bool, Optional[str]]:
    """
    Ingest historical OHLCV range (e.g., "2023-01-01" to "2024-01-01") into SQLite.
    
    Returns (success, error_message).
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return False, f"No historical data for {ticker} in range {start_date}–{end_date}"
        
        conn = get_db(db_path)
        cursor = conn.cursor()
        
        for timestamp, row in data.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO price_candles
                    (ticker, timestamp, open, high, low, close, volume, adjusted_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ticker,
                        timestamp,
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row["Volume"]),
                        float(row["Adj Close"])
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        
        conn.commit()
        conn.close()
        return True, None
    
    except Exception as e:
        return False, str(e)


# ============================================================================
# Query Interface
# ============================================================================

def get_latest_candle(ticker: str, db_path: str = "sentinel_prices.db") -> Optional[Dict]:
    """Fetch the most recent price candle for a ticker."""
    conn = get_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT id, ticker, timestamp, open, high, low, close, volume, adjusted_close, created_at
        FROM price_candles
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT 1
        """,
        (ticker,)
    )
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "ticker": row[1],
        "timestamp": row[2],
        "open": row[3],
        "high": row[4],
        "low": row[5],
        "close": row[6],
        "volume": row[7],
        "adjusted_close": row[8],
        "created_at": row[9]
    }


def get_price_range(
    ticker: str,
    days_back: int = 30,
    db_path: str = "sentinel_prices.db"
) -> List[Dict]:
    """
    Fetch OHLCV candles for the past N days for a ticker.
    
    Returns sorted list of price dicts [oldest...newest].
    """
    conn = get_db(db_path)
    cursor = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    
    cursor.execute(
        """
        SELECT id
