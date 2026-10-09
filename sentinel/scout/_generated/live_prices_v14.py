"""
Live price fetcher for Sentinel Scout pillar.

Fetches OHLCV (Open, High, Low, Close, Volume) data from yfinance for given tickers,
stores results in SQLite with schema compatible for future TimescaleDB migration.
Provides swap-ready interface: query by ticker/date range, bulk insert, and
fallback to stooq if yfinance fails.

Used by sentinel/pipeline.py to populate the historian's temporal dataset.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf
import pandas as pd


DB_PATH = "sentinel_prices.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Create SQLite schema for OHLCV storage if not exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, date)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_date ON ohlcv(ticker, date)
    """)
    conn.commit()
    conn.close()


def fetch_yfinance(ticker: str, start: str, end: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch OHLCV from yfinance; return None on failure."""
    try:
        data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if data.empty:
            return None
        data.reset_index(inplace=True)
        data.columns = [c.lower() for c in data.columns]
        return data
    except Exception:
        return None


def fetch_stooq_fallback(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fallback fetcher using stooq (requires stooq package); returns None if unavailable."""
    try:
        import stooq
        data = stooq.get(ticker, start_date=start, end_date=end)
        if data is None or data.empty:
            return None
        data.reset_index(inplace=True)
        data.columns = [c.lower() for c in data.columns]
        return data
    except Exception:
        return None


def store_ohlcv(ticker: str, df: pd.DataFrame, db_path: str = DB_PATH) -> int:
    """Insert OHLCV rows into SQLite; return count inserted."""
    if df is None or df.empty:
        return 0
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    inserted = 0
    for _, row in df.iterrows():
        date_str = str(row.get("date", row.get("Date", ""))).split(" ")[0]
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO ohlcv (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date_str,
                float(row.get("open", 0)),
                float(row.get("high", 0)),
                float(row.get("low", 0)),
                float(row.get("close", 0)),
                int(row.get("volume", 0))
            ))
            inserted += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return inserted


def fetch_and_store(ticker: str, days_back: int = 90, db_path: str = DB_PATH) -> dict:
    """
    Fetch OHLCV for ticker over last N days; try yfinance then stooq; store in DB.
    
    Returns dict with keys: ticker, rows_inserted, success, source.
    """
    init_db(db_path)
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    df = fetch_yfinance(ticker, start, end)
    source = "yfinance"
    if df is None:
        df = fetch_stooq_fallback(ticker, start, end)
        source = "stooq"
    
    rows_inserted = store_ohlcv(ticker, df, db_path) if df is not None else 0
    return {
        "ticker": ticker,
        "rows_inserted": rows_inserted,
        "success": rows_inserted > 0,
        "source": source if rows_inserted > 0 else "failed"
    }


def query_ohlcv(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, 
                db_path: str = DB_PATH) -> pd.DataFrame:
    """Query OHLCV rows for ticker in date range; return DataFrame."""
    conn = sqlite3.connect(db_path)
    query = "SELECT ticker, date, open, high, low, close, volume FROM ohlcv WHERE ticker = ?"
    params = [ticker]
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def latest_price(ticker: str, db_path: str = DB_PATH) -> Optional[dict]:
    """Fetch most recent OHLCV row for ticker; return dict or None."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, open, high, low, close, volume FROM ohlcv 
        WHERE ticker = ? 
        ORDER BY date DESC LIMIT 1
    """, (ticker,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "date": row[0],
            "open": row[1],
            "high": row[2],
            "low": row[3],
            "close": row[4],
            "volume": row[5]
        }
    return None
