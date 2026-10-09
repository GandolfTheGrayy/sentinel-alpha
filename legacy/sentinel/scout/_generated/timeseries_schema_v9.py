"""
Sentinel Scout — Time-Series SQLite Schema Module.

This module defines and manages the SQLite schema for persisting price history,
sentiment signals, and prediction records across the Sentinel pipeline.
It provides schema initialization, connection pooling, and basic CRUD operations
for the time-series data backbone that feeds RAG, reasoning, and post-mortem analysis.

Used by: Scout (data ingestion), Historian (RAG context), Judge (post-mortem).
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime


DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel_timeseries.db")


def ensure_db_dir() -> None:
    """Create parent directory for database if it doesn't exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Open and return a direct SQLite connection."""
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_session():
    """Context manager for safe database connections with auto-commit."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_schema() -> None:
    """Initialize all tables if they don't exist."""
    with get_db_session() as conn:
        cursor = conn.cursor()

        # Price history table: raw OHLCV + source metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                date_str TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                source TEXT,
                created_at INTEGER NOT NULL,
                UNIQUE(ticker, timestamp, source)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, timestamp DESC)"
        )

        # Sentiment signals table: per-ticker daily aggregates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date_str TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                signal_type TEXT NOT NULL,
                score REAL NOT NULL,
                confidence REAL,
                source TEXT,
                raw_text TEXT,
                metadata TEXT,
                created_at INTEGER NOT NULL,
                UNIQUE(ticker, date_str, signal_type, source)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_signals(ticker, timestamp DESC)"
        )

        # Prediction records table: daily per-ticker forecasts + outcomes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                prediction_timestamp INTEGER NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                price_target REAL,
                reasoning TEXT,
                horizon_days INTEGER,
                created_at INTEGER NOT NULL,
                outcome_date TEXT,
                outcome_actual_direction TEXT,
                outcome_price_change REAL,
                outcome_accuracy BOOLEAN,
                UNIQUE(ticker, prediction_date)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_ticker_date ON predictions(ticker, prediction_date DESC)"
        )

        # RAG corpus metadata table: embeddings + event references
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_corpus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                source_id TEXT,
                date_str TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                title TEXT,
                summary TEXT,
                embedding_id TEXT,
                relevance_score REAL,
                created_at INTEGER NOT NULL,
                UNIQUE(ticker, doc_type, source_id)
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_corpus_ticker_type ON rag_corpus(ticker, doc_type, timestamp DESC)"
        )

        # Anomaly log table: flagged outliers for Judge review
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                anomaly_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                date_str TEXT NOT NULL,
                severity TEXT,
                description TEXT,
                metrics TEXT,
                resolved BOOLEAN DEFAULT 0,
                created_at INTEGER NOT NULL,
                resolved_at INTEGER
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_anomaly_ticker_date ON anomalies(ticker, timestamp DESC)"
        )


def insert_price(
    ticker: str,
    timestamp: int,
    date_str: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    source: str = "yfinance",
) -> int:
    """Insert or replace a price record; return row id."""
    now = int(datetime.utcnow().timestamp())
    with get_db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO price_history
            (ticker, timestamp, date_str, open, high, low, close, volume, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, timestamp, date_str, open_, high, low, close, volume, source, now),
        )
        return cursor.lastrowid


def insert_sentiment(
    ticker: str,
    date_str: str,
    timestamp: int,
    signal_type: str,
    score: float,
    confidence: Optional[float] = None,
    source: str = "unknown",
    raw_text: Optional[str] = None,
    metadata: Optional[str] = None,
) -> int:
    """Insert a sentiment signal; return row id."""
    now = int(datetime.utcnow().timestamp())
    with get_db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sentiment_signals
            (ticker, date_str, timestamp, signal_type, score, confidence, source, raw_text, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                date_str,
                timestamp,
                signal_type,
                score,
                confidence,
                source,
                raw_text,
                metadata,
                now,
            ),
        )
        return cursor.lastrowid


def insert_prediction(
    ticker: str,
    prediction_date: str,
    prediction_timestamp: int,
    direction: str,
    confidence: float,
    price_target: Optional[float] = None,
    reasoning: Optional[str] = None,
    horizon_days: int = 1,
) -> int:
    """Insert a prediction record; return row id."""
    now = int(datetime.utcnow().timestamp())
    with get_db_session() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO predictions
            (ticker, prediction_date, prediction_timestamp, direction, confidence, price_target, reasoning, horizon_days, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker,
                prediction_date,
                prediction_timestamp,
                direction,
                confidence,
                price_target,
                reasoning,
                horizon_days,
