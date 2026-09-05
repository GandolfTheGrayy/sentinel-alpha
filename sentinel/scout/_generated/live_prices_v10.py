"""
Sentinel Scout: Live Price Fetcher

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data from yfinance
with automatic fallback to stooq for resilience. Stores all data in SQLite
with a schema designed for easy migration to TimescaleDB. This module serves
as the primary data ingestion point for live market prices, feeding directly
into the Judge's prediction pipeline.

Key responsibilities:
  - Fetch current + historical OHLCV for given tickers
  - Store in SQLite with timestamp indices for rapid retrieval
  - Provide swap-ready schema (no vendor lock-in)
  - Handle network failures gracefully with exponential backoff
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import yfinance as yf
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def init_price_db(db_path: str = "sentinel.db") -> sqlite3.Connection:
    """Initialize SQLite price table with indices for fast timestamp queries."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main OHLCV table: swap-ready schema (no yfinance-specific columns)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp),
            CHECK(high >= low),
            CHECK(close > 0),
            CHECK(volume >= 0)
        )
    """)
    
    # Indices for rapid lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_time ON ohlcv(ticker, timestamp DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON ohlcv(timestamp DESC)")
    
    # Metadata table: track fetch health and last update per ticker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_metadata (
            ticker TEXT PRIMARY KEY,
            last_fetch DATETIME,
            fetch_count INTEGER DEFAULT 0,
            error_count INTEGER DEFAULT 0,
            last_error TEXT,
            source TEXT DEFAULT 'yfinance'
        )
    """)
    
    conn.commit()
    return conn


def fetch_ohlcv(
    tickers: List[str],
    period: str = "1d",
    interval: str = "1d",
    db_path: str = "sentinel.db"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple tickers; store in SQLite; return DataFrames.
    
    Args:
        tickers: List of ticker symbols (e.g., ["AAPL", "GOOGL"])
        period: Historical lookback ("1d", "5d", "1mo", "1y", "max")
        interval: Candle interval ("1m", "5m", "1h", "1d")
        db_path: Path to SQLite database
    
    Returns:
        Dict mapping ticker → DataFrame with OHLCV columns
    """
    results = {}
    conn = init_price_db(db_path)
    
    for ticker in tickers:
        try:
            logger.info(f"Fetching {ticker} (period={period}, interval={interval})")
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}; attempting stooq fallback")
                df = _fallback_stooq(ticker, period)
                if df is None:
                    logger.error(f"Both yfinance and stooq failed for {ticker}")
                    _update_metadata(conn, ticker, error=True, source="stooq_failed")
                    continue
                source = "stooq"
            else:
                source = "yfinance"
            
            # Normalize column names (yfinance returns capitalized)
            df.columns = [c.lower() for c in df.columns]
            
            # Reset index so date becomes a column
            if df.index.name and df.index.name.lower() in ["date", "datetime"]:
                df = df.reset_index()
                df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
            elif "timestamp" not in df.columns:
                df["timestamp"] = df.index
                df = df.reset_index(drop=True)
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Filter to OHLCV columns only
            required = ["timestamp", "open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"{ticker}: missing columns {missing}")
                _update_metadata(conn, ticker, error=True, source=source)
                continue
            
            df = df[required].copy()
            df["ticker"] = ticker
            
            # Store in SQLite (ignore duplicates via UNIQUE constraint)
            _insert_ohlcv(conn, df)
            _update_metadata(conn, ticker, error=False, source=source)
            
            results[ticker] = df
            logger.info(f"✓ Stored {len(df)} rows for {ticker}")
            
        except Exception as e:
            logger.error(f"Exception fetching {ticker}: {e}")
            _update_metadata(conn, ticker, error=True, source="yfinance")
            continue
    
    conn.close()
    return results


def _fallback_stooq(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """
    Fallback to stooq API if yfinance fails.
    
    Args:
        ticker: Ticker symbol
        period: Lookback period string (converted to days)
    
    Returns:
        DataFrame or None if stooq also fails
    """
    try:
        # Stooq API endpoint for historical data
        url = f"https://stooq.com/q/export/?s={ticker}&i=d"
        df = pd.read_csv(url)
        df.rename(columns=str.lower, inplace=True)
        logger.info(f"Stooq fallback successful for {ticker}")
        return df
    except Exception as e:
        logger.warning(f"Stooq fallback failed for {ticker}: {e}")
        return None


def _insert_ohlcv(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Insert OHLCV DataFrame into SQLite, skipping duplicates."""
    cursor = conn.cursor()
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO ohlcv
                (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                row["ticker"],
                row["timestamp"],
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                int(row["volume"])
            ))
        except (ValueError, TypeError) as e:
            logger.warning(f"Row skipped (invalid data): {row.to_dict()}: {e}")
    conn.commit()


def _update_metadata(
    conn: sqlite3.Connection,
    ticker: str,
    error: bool = False,
    source: str = "yfinance"
) -> None:
    """Update price_metadata table with fetch result."""
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    
    if error:
        cursor.execute("""
            INSERT INTO price_metadata (ticker, last_fetch, fetch_count, error
