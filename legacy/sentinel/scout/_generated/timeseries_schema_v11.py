"""
Sentinel Scout — Time-Series SQLite Schema Module.

This module defines and manages the SQLite schema for Sentinel's time-series data:
price history, sentiment signals, prediction records, and resolution outcomes.
It provides schema initialization, connection pooling, and basic CRUD helpers
for the historian and judge pillars to query and store temporal data.

Used by: historian/rag_query.py (lookups), judge/predictor.py (context),
judge/resolver.py (outcome recording), and daily pipeline orchestration.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
import threading


_DB_PATH: str = os.getenv("SENTINEL_DB_PATH", str(Path.home() / ".sentinel" / "timeseries.db"))
_LOCK = threading.Lock()


def _ensure_db_dir() -> None:
    """Ensure the database directory exists."""
    db_dir = Path(_DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a thread-safe SQLite connection to the time-series database."""
    _ensure_db_dir()
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    """Create all required tables if they don't exist; idempotent."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Price history table: intraday/daily OHLCV + source attribution
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume INTEGER,
            source TEXT DEFAULT 'yfinance',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp, source)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_price_ticker_time ON price_history(ticker, timestamp DESC)"
    )
    
    # Sentiment signals table: headlines, Reddit posts, GitHub signals, etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            source TEXT NOT NULL,
            raw_text TEXT,
            score REAL,
            certainty REAL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, signal_type, source, timestamp)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time ON sentiment_signals(ticker, timestamp DESC)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentiment_type ON sentiment_signals(signal_type)"
    )
    
    # Prediction records: what Sentinel predicted and when
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date DATE NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            target_price REAL,
            reasoning TEXT,
            strategy TEXT,
            model_version TEXT DEFAULT 'claude-sonnet-4-6',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, prediction_date)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions(ticker, prediction_date DESC)"
    )
    
    # Resolution table: actual outcomes vs. predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            resolution_date DATE NOT NULL,
            predicted_direction TEXT NOT NULL,
            actual_direction TEXT NOT NULL,
            price_open REAL,
            price_close REAL,
            price_high REAL,
            price_low REAL,
            percent_move REAL,
            hit BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
            UNIQUE(prediction_id)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_resolutions_ticker_date ON resolutions(ticker, resolution_date DESC)"
    )
    
    # Anomalies table: flagged unusual patterns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            severity TEXT DEFAULT 'medium',
            description TEXT,
            evidence TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_anomalies_ticker_time ON anomalies(ticker, timestamp DESC)"
    )
    
    conn.commit()
    conn.close()


def record_price(
    ticker: str,
    timestamp: int,
    close: float,
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[int] = None,
    source: str = "yfinance",
) -> int:
    """Insert or update a price record; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO price_history
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, timestamp, source) DO UPDATE SET
                open=EXCLUDED.open,
                high=EXCLUDED.high,
                low=EXCLUDED.low,
                close=EXCLUDED.close,
                volume=EXCLUDED.volume
        """, (ticker, timestamp, open_, high, low, close, volume, source))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def record_sentiment(
    ticker: str,
    signal_type: str,
    timestamp: int,
    source: str,
    score: float,
    certainty: float,
    raw_text: Optional[str] = None,
    metadata: Optional[str] = None,
) -> int:
    """Insert or update a sentiment signal; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO sentiment_signals
            (ticker, signal_type, timestamp, source, score, certainty, raw_text, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, signal_type, source, timestamp) DO UPDATE SET
                score=EXCLUDED.score,
                certainty=EXCLUDED.certainty,
                raw_text=EXCLUDED.raw_text,
                metadata=EXCLUDED.metadata
        """, (ticker, signal_type, timestamp, source, score, certainty, raw_text, metadata))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def record_prediction(
    ticker: str,
    prediction_date: str,
    direction: str,
    confidence: float,
    target_price: Optional[float] = None,
    reasoning: Optional[str] = None,
    strategy: Optional[str] = None,
) -> int:
    """Insert a prediction record; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO predictions
            (ticker, prediction_date, direction, confidence, target_price, reasoning, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, prediction_date) DO UPDATE SET
                direction=EXCLUDED.direction,
