"""
Sentinel Scout — Time-Series SQLite Schema Module

Provides a durable SQLite schema for storing price history, sentiment signals,
and prediction records. This module is the backbone of Sentinel's data layer,
enabling the Judge and Historian pillars to query historical context and
validate post-mortems against actual market moves.

Tables:
  - price_history: OHLCV data from live_prices.py
  - sentiment_signals: Parsed scores from linguist modules (certainty, drift, etc.)
  - predictions: Per-ticker forecasts from predictor.py (direction, confidence, rationale)
  - prediction_outcomes: Resolved predictions vs. actual market moves (daily post-mortem)
  - rag_context_cache: Embedding-based lookups (ticker + query → top-k SEC/news snippets)

The schema is idempotent: calling ensure_tables() multiple times is safe.
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any


DB_PATH = os.environ.get("SENTINEL_DB_PATH", "sentinel_timeseries.db")


def get_db_connection() -> sqlite3.Connection:
    """Open or create the Sentinel SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables() -> None:
    """Create all required tables if they do not exist; idempotent."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # price_history: OHLCV records (5-min, 1-day, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            interval TEXT DEFAULT '1d',
            source TEXT DEFAULT 'yfinance',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp, interval)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_price_ticker_ts
        ON price_history(ticker, timestamp DESC);
    """)

    # sentiment_signals: Scores from linguist (certainty, drift, regulatory whispers)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            signal_type TEXT NOT NULL,
            value REAL NOT NULL,
            source TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp, signal_type, source)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_ts
        ON sentiment_signals(ticker, timestamp DESC);
    """)

    # predictions: Forecasts from judge/predictor.py
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date INTEGER NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            target_price REAL,
            rationale TEXT NOT NULL,
            model_version TEXT DEFAULT '1.0',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, prediction_date)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
        ON predictions(ticker, prediction_date DESC);
    """)

    # prediction_outcomes: Resolved predictions (post-mortem)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            resolution_date INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            actual_direction TEXT NOT NULL,
            price_move_pct REAL NOT NULL,
            confidence_given REAL NOT NULL,
            correct INTEGER NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id),
            UNIQUE(prediction_id, resolution_date)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_outcomes_ticker_date
        ON prediction_outcomes(ticker, resolution_date DESC);
    """)

    # rag_context_cache: Cached embeddings + lookup results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_context_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            snippet TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            relevance_score REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, query_hash, source, timestamp)
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rag_ticker_query
        ON rag_context_cache(ticker, query_hash);
    """)

    conn.commit()
    conn.close()


def insert_price_record(
    ticker: str,
    timestamp: int,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    interval: str = "1d",
    source: str = "yfinance",
) -> int:
    """Insert a single OHLCV record; return row ID or -1 on duplicate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO price_history
            (ticker, timestamp, open, high, low, close, volume, interval, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, timestamp, open_price, high, low, close, volume, interval, source),
        )
        conn.commit()
        row_id = cursor.lastrowid
        return row_id
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()


def insert_sentiment_signal(
    ticker: str,
    timestamp: int,
    signal_type: str,
    value: float,
    source: str,
    metadata: Optional[str] = None,
) -> int:
    """Insert a sentiment signal (e.g. certainty_score, drift_pct); return row ID or -1 on duplicate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO sentiment_signals
            (ticker, timestamp, signal_type, value, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker, timestamp, signal_type, value, source, metadata),
        )
        conn.commit()
        row_id = cursor.lastrowid
        return row_id
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()


def insert_prediction(
    ticker: str,
    prediction_date: int,
    direction: str,
    confidence: float,
    rationale: str,
    target_price: Optional[float] = None,
    model_version: str = "1.0",
) -> int:
    """Insert a prediction record (direction='UP'|'DOWN'|'HOLD'); return row ID or -1 on duplicate."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO predictions
            (ticker, prediction_date, direction, confidence, target_price, rationale, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, prediction_date, direction, confidence, target_price, rationale, model_version),
        )
        conn.commit()
        row_id = cursor.lastrowid
        return row_id
    except sqlite3.IntegrityError
