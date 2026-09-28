"""
Sentinel Scout: Time-Series SQLite Schema Module

This module defines and manages the core SQLite schema for Sentinel's
time-series data pipeline. It creates tables for:
  - price_history: OHLCV candles for each ticker
  - sentiment_signals: Reddit/HN/news sentiment snapshots
  - prediction_records: Daily predictions + ground truth
  - embedding_cache: Vector embeddings for RAG lookups
  - anomaly_log: Flagged market events for post-mortem analysis

Used by Scout ingestion modules and Historian RAG pipeline for persistent
storage and fast lookups. Automatically initializes schema on first use.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple


DB_PATH = Path("sentinel_timeseries.db")


def init_schema(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize or connect to the time-series database, creating tables if needed."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Price history table: OHLCV + volume + source
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT DEFAULT 'yfinance',
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, date, source)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date)"
    )

    # Sentiment signals table: per-source sentiment snapshots
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT,
            score REAL,
            confidence REAL,
            raw_text TEXT,
            metadata TEXT,
            captured_at TEXT NOT NULL,
            UNIQUE(ticker, date, source, signal_type)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_signals(ticker, date)"
    )

    # Prediction records table: daily predictions + outcomes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            predicted_confidence REAL,
            predicted_price_target REAL,
            reasoning TEXT,
            baseline_strategy TEXT,
            actual_close REAL,
            actual_direction TEXT,
            accuracy_flag TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, prediction_date)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_ticker_date ON prediction_records(ticker, prediction_date)"
    )

    # Embedding cache table: RAG vector embeddings for fast lookup
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            ticker TEXT,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER,
            content_hash TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(source_id, source_type)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_ticker ON embedding_cache(ticker)"
    )

    # Anomaly log table: flagged market events + post-mortem notes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            event_date TEXT NOT NULL,
            anomaly_type TEXT,
            magnitude REAL,
            description TEXT,
            related_signals TEXT,
            flagged_at TEXT NOT NULL,
            resolved BOOLEAN DEFAULT 0,
            resolution_note TEXT,
            UNIQUE(ticker, event_date, anomaly_type)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_anomaly_ticker_date ON anomaly_log(ticker, event_date)"
    )

    conn.commit()
    return conn


def insert_price_candle(
    ticker: str,
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    source: str = "yfinance",
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Insert a single OHLCV candle into price_history."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO price_history
        (ticker, date, open, high, low, close, volume, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, open_, high, low, close, volume, source, datetime.utcnow().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def insert_sentiment_signal(
    ticker: str,
    date: str,
    source: str,
    signal_type: str,
    score: float,
    confidence: float,
    raw_text: Optional[str] = None,
    metadata: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Insert a sentiment signal snapshot into sentiment_signals."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO sentiment_signals
        (ticker, date, source, signal_type, score, confidence, raw_text, metadata, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            date,
            source,
            signal_type,
            score,
            confidence,
            raw_text,
            metadata,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_prediction_record(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_confidence: float,
    predicted_price_target: Optional[float] = None,
    reasoning: Optional[str] = None,
    baseline_strategy: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    """Insert a daily prediction into prediction_records."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO prediction_records
        (ticker, prediction_date, predicted_direction, predicted_confidence,
         predicted_price_target, reasoning, baseline_strategy, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            prediction_date,
            predicted_direction,
            predicted_confidence,
            predicted_price_target,
            reasoning,
            baseline_strategy,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def resolve_prediction(
    ticker: str,
    prediction_date: str,
    actual_close: float,
    actual_direction: str,
    accuracy_flag: str,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """Update a prediction record with ground-truth outcome."""
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE prediction_records
        SET actual_close = ?, actual_
