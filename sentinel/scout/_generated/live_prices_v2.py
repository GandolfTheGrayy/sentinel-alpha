"""
Live price fetcher for Sentinel Scout.

Fetches OHLCV (Open, High, Low, Close, Volume) data from yfinance with
stooq fallback, stores snapshots in SQLite with TimescaleDB-compatible schema,
and exposes a swap-ready interface for plugging in alternative data sources.

Integrated into the daily pipeline to capture real-time price signals for
sentiment correlation and prediction baseline anchoring.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


DB_PATH: str = "sentinel_prices.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize SQLite schema with TimescaleDB-compatible OHLCV table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ohlcv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        created_at TEXT NOT NULL,
        UNIQUE(ticker, timestamp)
    )
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
    ON ohlcv(ticker, timestamp DESC)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_created_at 
    ON ohlcv(created_at DESC)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def fetch_ohlcv_yfinance(
    ticker: str,
    period: str = "1d",
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from yfinance for given ticker and period."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            logger.warning(f"No data returned from yfinance for {ticker}")
            return None
        data.columns = [c.lower() for c in data.columns]
        return data
    except Exception as e:
        logger.error(f"yfinance fetch failed for {ticker}: {e}")
        return None


def fetch_ohlcv_stooq(ticker: str, period: str = "252") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from stooq as fallback (stub for future implementation)."""
    logger.info(f"Stooq fallback not yet implemented for {ticker}")
    return None


def fetch_ohlcv(
    ticker: str,
    period: str = "1d",
    interval: str = "1d",
    use_fallback: bool = True
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with automatic fallback to stooq if yfinance fails."""
    data = fetch_ohlcv_yfinance(ticker, period=period, interval=interval)
    
    if data is None and use_fallback:
        logger.info(f"Falling back to stooq for {ticker}")
        data = fetch_ohlcv_stooq(ticker, period=period)
    
    return data


def store_ohlcv(
    ticker: str,
    data: pd.DataFrame,
    db_path: str = DB_PATH
) -> int:
    """Store OHLCV DataFrame rows into SQLite, returning count inserted."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    inserted = 0
    now = datetime.utcnow().isoformat()
    
    for timestamp, row in data.iterrows():
        ts_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
        
        try:
            cursor.execute("""
            INSERT OR REPLACE INTO ohlcv 
            (ticker, timestamp, open, high, low, close, volume, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                ts_str,
                float(row.get('open', 0)) if pd.notna(row.get('open')) else None,
                float(row.get('high', 0)) if pd.notna(row.get('high')) else None,
                float(row.get('low', 0)) if pd.notna(row.get('low')) else None,
                float(row.get('close', 0)) if pd.notna(row.get('close')) else None,
                int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                now
            ))
            inserted += 1
        except Exception as e:
            logger.error(f"Failed to insert {ticker} {ts_str}: {e}")
    
    conn.commit()
    conn.close()
    logger.info(f"Stored {inserted} rows for {ticker}")
    
    return inserted


def fetch_and_store(
    ticker: str,
    period: str = "1d",
    interval: str = "1d",
    db_path: str = DB_PATH
) -> Tuple[bool, int]:
    """Fetch OHLCV data and store in SQLite; return (success, count)."""
    data = fetch_ohlcv(ticker, period=period, interval=interval)
    
    if data is None or data.empty:
        logger.warning(f"No data to store for {ticker}")
        return False, 0
    
    count = store_ohlcv(ticker, data, db_path=db_path)
    return True, count


def query_ohlcv(
    ticker: str,
    days: int = 30,
    db_path: str = DB_PATH
) -> List[Dict]:
    """Query recent OHLCV records for ticker from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    cursor.execute("""
    SELECT ticker, timestamp, open, high, low, close, volume, created_at
    FROM ohlcv
    WHERE ticker = ? AND timestamp >= ?
    ORDER BY timestamp DESC
    """, (ticker, cutoff))
    
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return rows


def latest_price(ticker: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Get the most recent OHLCV record for a ticker."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT ticker, timestamp, open, high, low, close, volume, created_at
    FROM ohlcv
    WHERE ticker = ?
    ORDER BY timestamp DESC
    LIMIT 1
    """, (ticker,))
    
    row = cursor.fetchone()
    conn.close()
    
    return dict(row) if row else None


def delete_old_records(days: int = 90, db_path: str = DB_PATH) -> int:
    """Delete OHLCV records older than specified days; return count deleted."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    cursor.execute("DELETE FROM ohlcv WHERE created_at < ?", (cutoff,))
    deleted = cursor.rowcount
    
    conn.commit()
    conn.close()
    logger.info(f"Deleted {deleted} records older than {days} days")
    
    return deleted


if __name__ == "__main
