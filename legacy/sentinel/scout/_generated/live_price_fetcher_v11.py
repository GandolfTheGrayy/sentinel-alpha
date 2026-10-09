"""
Sentinel Scout: Live Price Fetcher

Fetches real-time OHLCV (Open, High, Low, Close, Volume) data for equities via yfinance,
persists to SQLite with a swap-ready schema for future TimescaleDB migration. Called by
the main pipeline during each intraday cycle to populate the historian's time-series corpus.

Design: Modular connector pattern. Core fetch() wraps yfinance; store_to_sqlite() persists
with UTC timestamps and idempotent upsert semantics. Schema is minimal and denormalized
to support both SQLite and future TimescaleDB without schema bloat.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas as pd


# ============================================================================
# SQLite Schema & Connection Management
# ============================================================================

_DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel/data/prices.db")


def init_db(db_path: str = _DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite schema for OHLCV storage; idempotent."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Main OHLCV table: ticker, timestamp (UTC), OHLCV, volume.
    # Composite unique index on (ticker, timestamp) for idempotent upsert.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, timestamp)
        )
    """)
    
    # Metadata: last successful fetch per ticker, for backoff/retry logic.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            last_fetch TEXT NOT NULL,
            status TEXT NOT NULL,
            error_msg TEXT,
            row_count INTEGER
        )
    """)
    
    conn.commit()
    return conn


def get_db_connection(db_path: str = _DB_PATH) -> sqlite3.Connection:
    """Acquire a fresh SQLite connection in row_factory mode for dict-like access."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# Live Fetch via yfinance
# ============================================================================

def fetch_ohlcv(ticker: str, period: str = "1d", interval: str = "1m") -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from yfinance; return DataFrame with UTC datetime index and OHLCV columns.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period: Lookback window ("1d", "5d", "1mo", etc.)
        interval: Candle size ("1m", "5m", "1h", "1d")
    
    Returns:
        DataFrame with MultiIndex (ticker, datetime) and [Open, High, Low, Close, Volume],
        or None if fetch fails.
    """
    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            prepost=False,  # Exclude pre/post-market for consistency.
        )
        if data is None or data.empty:
            return None
        
        # Ensure UTC timezone and reset to single index for downstream processing.
        if hasattr(data.index, "tz"):
            data.index = data.index.tz_convert("UTC")
        else:
            data.index = data.index.tz_localize("UTC", ambiguous="raise", nonexistent="shift_forward")
        
        data["ticker"] = ticker
        return data
    except Exception as e:
        print(f"[fetch_ohlcv] Failed to fetch {ticker}: {e}")
        return None


def fetch_batch(tickers: List[str], period: str = "1d", interval: str = "1m") -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple tickers in parallel; return dict of ticker -> DataFrame.
    
    Args:
        tickers: List of stock symbols.
        period: Lookback window.
        interval: Candle size.
    
    Returns:
        Dict {ticker: DataFrame} with timezone-aware UTC datetimes; empty dict on failure.
    """
    result = {}
    for ticker in tickers:
        df = fetch_ohlcv(ticker, period=period, interval=interval)
        if df is not None:
            result[ticker] = df
    return result


# ============================================================================
# Persist to SQLite
# ============================================================================

def store_to_sqlite(
    df: pd.DataFrame,
    ticker: str,
    db_path: str = _DB_PATH,
    overwrite: bool = False
) -> Tuple[int, str]:
    """
    Upsert OHLCV DataFrame into SQLite prices table with idempotent semantics.
    
    Args:
        df: DataFrame with datetime index and [Open, High, Low, Close, Volume].
        ticker: Stock symbol for labeling.
        db_path: Path to SQLite database.
        overwrite: If True, DELETE old rows for this ticker before insert.
    
    Returns:
        Tuple (row_count, status_msg) indicating rows inserted/upserted.
    """
    if df is None or df.empty:
        return 0, "empty"
    
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    try:
        if overwrite:
            cursor.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
        
        fetched_at = datetime.utcnow().isoformat() + "Z"
        rows_inserted = 0
        
        for timestamp, row in df.iterrows():
            # Normalize timestamp to ISO string.
            ts_str = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            
            try:
                cursor.execute("""
                    INSERT INTO prices (ticker, timestamp, open, high, low, close, volume, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticker, timestamp) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        fetched_at = excluded.fetched_at
                """, (
                    ticker,
                    ts_str,
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    int(row["Volume"]),
                    fetched_at
                ))
                rows_inserted += 1
            except (ValueError, TypeError) as e:
                # Skip malformed rows; log and continue.
                print(f"[store_to_sqlite] Skipped row for {ticker} @ {ts_str}: {e}")
                continue
        
        # Update fetch log.
        cursor.execute("""
            INSERT INTO fetch_log (ticker, last_fetch, status, row_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                last_fetch = excluded.last_fetch,
                status = excluded.status,
                row_count = excluded.row_count
        """, (ticker, fetched_at, "ok", rows_inserted))
        
        conn.commit()
        return rows_inserted, "ok"
    
    except sqlite3.Error as e:
        conn.rollback()
        error_msg = f"sql_error: {str
