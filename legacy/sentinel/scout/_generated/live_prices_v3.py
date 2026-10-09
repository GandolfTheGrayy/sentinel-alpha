"""
Sentinel Scout: Live Price Fetcher
==================================

This module fetches real-time OHLCV (Open, High, Low, Close, Volume) data for
equities using yfinance, storing results in SQLite with a swap-ready schema
compatible with TimescaleDB migrations. Provides both snapshot queries and
historical rolling windows for sentiment-price correlation analysis.

Role in Sentinel: Scout pillar's primary data source for live market signals,
feeding price history to Historian RAG queries and Judge calibration loops.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json
import logging

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# SQLite database path — override via SENTINEL_DB env var
SENTINEL_DB = os.getenv("SENTINEL_DB", "sentinel.db")


def init_ohlcv_schema() -> None:
    """Initialize SQLite schema for OHLCV time-series data with TimescaleDB-ready columns."""
    conn = sqlite3.connect(SENTINEL_DB)
    cursor = conn.cursor()
    
    # Main OHLCV table — TimescaleDB will convert time to TIMESTAMP, ohlcv to composite
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            time DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adjusted_close REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, time)
        )
    """)
    
    # Metadata table for fetch tracking and cache invalidation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            last_fetch_utc DATETIME,
            last_price REAL,
            last_volume INTEGER,
            data_points INTEGER,
            status TEXT DEFAULT 'ok',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indices for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, time DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv(symbol)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_metadata_symbol ON price_metadata(symbol)")
    
    conn.commit()
    conn.close()
    logger.info("OHLCV schema initialized")


def fetch_and_store(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    force_refresh: bool = False
) -> bool:
    """
    Fetch OHLCV data for a symbol and insert/upsert into SQLite.
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        period: yfinance period ('1d', '1mo', '1y', 'max')
        interval: Candle interval ('1m', '5m', '1h', '1d')
        force_refresh: If True, skip cache check and re-fetch
    
    Returns:
        True if fetch succeeded, False otherwise
    """
    try:
        logger.info(f"Fetching {symbol} ({period}/{interval})")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            _update_metadata(symbol, status="no_data")
            return False
        
        # Ensure timezone-naive datetime for SQLite
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        conn = sqlite3.connect(SENTINEL_DB)
        cursor = conn.cursor()
        
        inserted = 0
        for timestamp, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, time, open, high, low, close, volume, adjusted_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    timestamp.isoformat(),
                    float(row.get("Open", 0)),
                    float(row.get("High", 0)),
                    float(row.get("Low", 0)),
                    float(row.get("Close", 0)),
                    int(row.get("Volume", 0)),
                    float(row.get("Adj Close", row.get("Close", 0)))
                ))
                inserted += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping row {timestamp} for {symbol}: {e}")
                continue
        
        conn.commit()
        
        # Update metadata
        latest = df.iloc[-1]
        _update_metadata(
            symbol,
            last_price=float(latest["Close"]),
            last_volume=int(latest["Volume"]),
            data_points=len(df),
            status="ok"
        )
        
        conn.close()
        logger.info(f"Inserted {inserted} rows for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        _update_metadata(symbol, status=f"error: {str(e)[:50]}")
        return False


def get_latest_price(symbol: str) -> Optional[Dict[str, float]]:
    """
    Retrieve the most recent OHLCV candle for a symbol.
    
    Args:
        symbol: Stock ticker
    
    Returns:
        Dict with keys 'close', 'high', 'low', 'open', 'volume', 'time' or None
    """
    conn = sqlite3.connect(SENTINEL_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT symbol, time, open, high, low, close, volume, adjusted_close
        FROM ohlcv
        WHERE symbol = ?
        ORDER BY time DESC
        LIMIT 1
    """, (symbol,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "symbol": row["symbol"],
        "time": row["time"],
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "volume": row["volume"],
        "adjusted_close": row["adjusted_close"]
    }


def get_window(
    symbol: str,
    days: int = 30,
    limit: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Retrieve OHLCV data for a symbol over a rolling window (most recent N days).
    
    Args:
        symbol: Stock ticker
        days: Number of days to look back
        limit: Optional cap on returned rows (e.g., 20 for top 20 most recent)
    
    Returns:
        List of dicts with 'time', 'open', 'high', 'low', 'close', 'volume'
    """
    conn = sqlite3.connect(SENTINEL_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    query = """
        SELECT symbol, time, open, high, low, close, volume, adjusted_close
        FROM ohlcv
        WHERE symbol = ? AND time >= ?
        ORDER BY time DESC
    """
    params = [symbol, cutoff]
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
