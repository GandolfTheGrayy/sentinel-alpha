"""
Sentinel Scout: Live Price Fetcher with SQLite Storage

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data via yfinance
and stores it in SQLite with a swap-ready interface for TimescaleDB migration.
Designed as a lightweight, portable price ingestion layer for the Sentinel
Sentiment Engine's daily pipeline.

Key responsibilities:
  - Fetch live market data for a given ticker via yfinance
  - Fallback to stooq if yfinance fails
  - Persist OHLCV to SQLite with timestamp indexing
  - Provide query interface compatible with future TimescaleDB swap
  - Handle partial data and retry logic gracefully
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class LivePriceFetcher:
    """
    Fetches and stores live OHLCV data with SQLite persistence and
    TimescaleDB-compatible schema design.
    """

    def __init__(self, db_path: str = "sentinel_prices.db") -> None:
        """Initialize SQLite connection and ensure schema exists."""
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create prices table if it does not exist, with TimescaleDB-compatible design."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # TimescaleDB-ready schema: time-series optimized columns
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
            )
        """)
        
        # Index for fast time-based queries (critical for TimescaleDB migration)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_prices_ticker_time
            ON prices(ticker, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Schema ensured in {self.db_path}")

    def fetch_live(self, ticker: str, period: str = "5d", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data via yfinance; fallback to stooq on failure.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Time period ('5d', '1mo', '1y', etc.)
            interval: Candle interval ('1d', '1h', '15m', etc.)

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume] or None on failure
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if data.empty:
                logger.warning(f"yfinance returned empty data for {ticker}; trying fallback")
                return self._fetch_stooq_fallback(ticker, period)
            logger.info(f"Fetched {len(data)} rows for {ticker} via yfinance")
            return data
        except Exception as e:
            logger.error(f"yfinance fetch failed for {ticker}: {e}; trying fallback")
            return self._fetch_stooq_fallback(ticker, period)

    def _fetch_stooq_fallback(self, ticker: str, period: str = "5d") -> Optional[pd.DataFrame]:
        """
        Fallback price fetcher using stooq API (lightweight, no auth required).

        Args:
            ticker: Stock ticker symbol
            period: Time period for fallback fetch

        Returns:
            DataFrame or None if fallback also fails
        """
        try:
            # Map period to approximate days for stooq
            days_map = {"5d": 5, "1mo": 30, "3mo": 90, "1y": 365}
            days = days_map.get(period, 5)
            
            # stooq URL format (basic daily data)
            url = f"https://stooq.com/q/export.php?s={ticker.lower()}&d={datetime.now().strftime('%Y%m%d')}&i=d"
            data = pd.read_csv(url, sep=",", index_col=0, parse_dates=True)
            
            if data.empty:
                logger.warning(f"stooq fallback also returned empty for {ticker}")
                return None
            
            # Reorder columns to match yfinance output
            data = data[["Open", "High", "Low", "Close", "Volume"]]
            logger.info(f"Fetched {len(data)} rows for {ticker} via stooq fallback")
            return data
        except Exception as e:
            logger.error(f"stooq fallback also failed for {ticker}: {e}")
            return None

    def store_ohlcv(self, ticker: str, data: pd.DataFrame) -> int:
        """
        Persist OHLCV DataFrame to SQLite, skipping duplicates gracefully.

        Args:
            ticker: Stock ticker symbol
            data: DataFrame with index as datetime and columns [Open, High, Low, Close, Volume]

        Returns:
            Number of rows successfully inserted
        """
        if data is None or data.empty:
            logger.warning(f"No data to store for {ticker}")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        inserted = 0
        for timestamp, row in data.iterrows():
            try:
                # Ensure timestamp is timezone-naive for consistency
                if hasattr(timestamp, 'tz_localize'):
                    timestamp = timestamp.tz_localize(None)
                elif hasattr(timestamp, 'replace'):
                    timestamp = timestamp.replace(tzinfo=None)
                
                timestamp_str = timestamp.isoformat()
                cursor.execute("""
                    INSERT OR IGNORE INTO prices
                    (ticker, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    ticker,
                    timestamp_str,
                    float(row.get("Open", 0)),
                    float(row.get("High", 0)),
                    float(row.get("Low", 0)),
                    float(row.get("Close", 0)),
                    int(row.get("Volume", 0))
                ))
                inserted += cursor.rowcount
            except Exception as e:
                logger.error(f"Failed to insert row for {ticker} at {timestamp}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {inserted} new records for {ticker}")
        return inserted

    def get_latest(self, ticker: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """
        Retrieve recent OHLCV data from SQLite for a given ticker.

        Args:
            ticker: Stock ticker symbol
            lookback_days: Number of days of historical data to return

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume] or None
        """
        conn = sqlite3.connect(self.db_path)
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        try:
            data = pd.read_sql_query("""
                SELECT timestamp, open, high, low, close, volume
                FROM prices
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, conn, params=(ticker, cutoff.isoformat()))
            
            if data.empty:
                logger.warning(f"No data found for {ticker} in last {lookback_days} days")
                return None
            
            # Convert timestamp string back to datetime
            data["timestamp"]
