"""
Live price fetcher for Sentinel Scout pillar.

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data via yfinance
with stooq fallback. Stores data in SQLite with schema designed for easy
migration to TimescaleDB. Provides swap-ready interface: caller connects
to either SQLite or TimescaleDB without changing fetch logic.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd


class LivePriceFetcher:
    """Fetches and persists OHLCV data with pluggable storage backend."""

    def __init__(self, db_path: str = "sentinel_prices.db"):
        """Initialize fetcher and ensure schema exists."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create OHLCV table if missing; schema compatible with TimescaleDB migration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
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
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ohlcv(ticker, timestamp DESC)"
        )
        conn.commit()
        conn.close()

    def fetch_latest(self, ticker: str, days: int = 1) -> Optional[Dict]:
        """
        Fetch latest OHLCV for ticker; return most recent close or None.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            days: Lookback window in days
            
        Returns:
            Dict with keys {open, high, low, close, volume, timestamp} or None on error
        """
        try:
            data = yf.download(ticker, period=f"{days}d", progress=False)
            if data.empty:
                return None
            latest = data.iloc[-1]
            return {
                "ticker": ticker,
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": int(latest["Volume"]),
                "timestamp": latest.name.isoformat(),
            }
        except Exception as e:
            print(f"[LivePriceFetcher] Error fetching {ticker}: {e}")
            return None

    def fetch_historical(self, ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV range for ticker.
        
        Args:
            ticker: Stock symbol
            start: ISO date string (YYYY-MM-DD)
            end: ISO date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume] indexed by date, or None
        """
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            return data if not data.empty else None
        except Exception as e:
            print(f"[LivePriceFetcher] Error fetching historical {ticker}: {e}")
            return None

    def store_ohlcv(self, ticker: str, ohlcv_dict: Dict) -> bool:
        """
        Persist OHLCV record to SQLite (insert-or-ignore on duplicate).
        
        Args:
            ticker: Stock symbol
            ohlcv_dict: Dict with keys {open, high, low, close, volume, timestamp}
            
        Returns:
            True if inserted/updated, False on error
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO ohlcv
                (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    ohlcv_dict["timestamp"],
                    ohlcv_dict["open"],
                    ohlcv_dict["high"],
                    ohlcv_dict["low"],
                    ohlcv_dict["close"],
                    ohlcv_dict["volume"],
                ),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[LivePriceFetcher] Error storing OHLCV for {ticker}: {e}")
            return False

    def batch_store(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Bulk insert DataFrame (from yfinance) into SQLite.
        
        Args:
            ticker: Stock symbol
            df: DataFrame with columns [Open, High, Low, Close, Volume]
            
        Returns:
            Count of rows inserted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            inserted = 0
            for idx, row in df.iterrows():
                try:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO ohlcv
                        (ticker, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ticker,
                            idx.isoformat(),
                            float(row["Open"]),
                            float(row["High"]),
                            float(row["Low"]),
                            float(row["Close"]),
                            int(row["Volume"]),
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            conn.close()
            return inserted
        except Exception as e:
            print(f"[LivePriceFetcher] Error batch-storing for {ticker}: {e}")
            return 0

    def query_range(self, ticker: str, start: str, end: str) -> List[Dict]:
        """
        Query stored OHLCV records for ticker within date range.
        
        Args:
            ticker: Stock symbol
            start: ISO date string (YYYY-MM-DD)
            end: ISO date string (YYYY-MM-DD)
            
        Returns:
            List of dicts with keys {ticker, timestamp, open, high, low, close, volume}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ticker, timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (ticker, start, end),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows
        except Exception as e:
            print(f"[LivePriceFetcher] Error querying {ticker}: {e}")
            return []

    def get_latest_price(self, ticker: str) -> Optional[float]:
        """
        Retrieve most recent close price for ticker from storage.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Float close price, or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT close FROM ohlcv
                WHERE ticker = ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (ticker,),
            )
            row = cursor.
