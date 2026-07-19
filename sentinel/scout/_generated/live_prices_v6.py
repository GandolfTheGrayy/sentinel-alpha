"""
Live Price Fetcher for Sentinel Sentiment Engine.

Modular yfinance-based fetcher that pulls OHLCV (Open, High, Low, Close, Volume)
data for equity tickers and stores in SQLite with a swap-ready interface for
TimescaleDB migration. Part of the Scout pillar's data ingestion layer.

Provides both immediate fetch and historical backfill, with configurable retry
logic and fallback to stooq for resilience.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json
import logging

import yfinance as yf
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


logger = logging.getLogger(__name__)


def _ensure_db_path(db_path: str) -> str:
    """Create parent directory for SQLite DB if missing."""
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return db_path


def init_sqlite_schema(db_path: str) -> None:
    """Initialize SQLite schema for OHLCV storage with TimescaleDB-compatible layout."""
    db_path = _ensure_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main OHLCV table: designed to mirror eventual TimescaleDB hypertable structure.
    cursor.execute("""
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
    
    # Index for fast ticker + time range lookups.
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
        ON ohlcv(ticker, timestamp DESC)
    """)
    
    # Metadata table: tracks last fetch per ticker for idempotency.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            last_fetch_timestamp DATETIME NOT NULL,
            status TEXT DEFAULT 'success',
            error_msg TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"SQLite schema initialized at {db_path}")


def _requests_session_with_retry(retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
    """Create requests.Session with exponential backoff for resilience."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_live_price(
    ticker: str,
    db_path: str = "data/ohlcv.db",
    use_cache: bool = True
) -> Optional[Dict[str, float]]:
    """
    Fetch latest OHLCV snapshot for a single ticker; store in SQLite.
    
    Returns dict with 'open', 'high', 'low', 'close', 'volume' or None on failure.
    """
    try:
        # Attempt yfinance fetch with live data.
        data = yf.download(ticker, period="1d", progress=False, timeout=10)
        
        if data.empty:
            logger.warning(f"No data returned for ticker {ticker}, trying stooq fallback")
            return _fetch_fallback_stooq(ticker, db_path)
        
        # Extract latest row (most recent trading day).
        latest = data.iloc[-1]
        ohlcv = {
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": int(latest["Volume"])
        }
        
        # Store in SQLite.
        _store_ohlcv(ticker, datetime.now().date(), ohlcv, db_path)
        return ohlcv
        
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return _fetch_fallback_stooq(ticker, db_path)


def _fetch_fallback_stooq(ticker: str, db_path: str) -> Optional[Dict[str, float]]:
    """Fallback stooq scraper for when yfinance fails."""
    try:
        session = _requests_session_with_retry()
        url = f"https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcvn&h&e=csv"
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            logger.warning(f"Stooq returned empty data for {ticker}")
            return None
        
        # Parse CSV: Symbol,Date,Time,Open,High,Low,Close,Volume,Name
        parts = lines[1].split(",")
        if len(parts) < 8:
            return None
        
        ohlcv = {
            "open": float(parts[3]),
            "high": float(parts[4]),
            "low": float(parts[5]),
            "close": float(parts[6]),
            "volume": int(float(parts[7]))
        }
        
        _store_ohlcv(ticker, datetime.now().date(), ohlcv, db_path)
        return ohlcv
        
    except Exception as e:
        logger.error(f"Stooq fallback also failed for {ticker}: {e}")
        return None


def _store_ohlcv(ticker: str, date: datetime, ohlcv: Dict[str, float], db_path: str) -> None:
    """Insert or ignore OHLCV record in SQLite."""
    db_path = _ensure_db_path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO ohlcv 
            (ticker, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            date.isoformat(),
            ohlcv["open"],
            ohlcv["high"],
            ohlcv["low"],
            ohlcv["close"],
            ohlcv["volume"]
        ))
        conn.commit()
    finally:
        conn.close()


def fetch_historical_range(
    ticker: str,
    start_date: str,
    end_date: str,
    db_path: str = "data/ohlcv.db"
) -> pd.DataFrame:
    """
    Fetch and store multi-day OHLCV data; return as DataFrame.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        start_date: ISO date string (e.g., "2024-01-01")
        end_date: ISO date string (e.g., "2024-01-31")
        db_path: Path to SQLite database
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=10)
