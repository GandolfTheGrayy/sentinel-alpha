"""
Sentinel time-series SQLite schema module.

Provides a base schema for storing price history, sentiment signals, and
prediction records. Used by Scout ingestion pipelines and Historian RAG
lookups to maintain a persistent, queryable audit trail of market signals
and model predictions.

Functions:
  - init_db(): Create/verify all tables in the target SQLite file.
  - get_connection(): Thread-safe connection factory.
  - reset_db(): Drop and recreate schema (dev/testing only).
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

# Thread-local storage for per-thread DB connections
_thread_local = threading.local()

# Default database path; override via environment or function parameter
DEFAULT_DB_PATH = Path("sentinel_timeseries.db")

SCHEMA_VERSION = 1


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Return a thread-safe SQLite connection, creating it if needed.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not hasattr(_thread_local, "connection"):
        _thread_local.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        _thread_local.connection.row_factory = sqlite3.Row
    return _thread_local.connection


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize or verify the time-series schema; idempotent.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Price history: raw OHLCV + data source
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume INTEGER,
            source TEXT DEFAULT 'yfinance',
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, date, source)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_price_ticker_date
        ON price_history(ticker, date DESC)
    """)

    # Sentiment signals: aggregated scores from news, Reddit, GitHub, etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            score REAL NOT NULL,
            magnitude INTEGER DEFAULT 1,
            metadata TEXT,
            recorded_at TEXT NOT NULL,
            UNIQUE(ticker, date, source, signal_type)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
        ON sentiment_signals(ticker, date DESC)
    """)

    # Linguistic analysis: certainty, drift, regulatory signals per ticker
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS linguistic_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            certainty_score REAL,
            hesitation_indicators INTEGER DEFAULT 0,
            linguistic_drift REAL,
            regulatory_whisper_flag INTEGER DEFAULT 0,
            source_doc_id TEXT,
            analysis_metadata TEXT,
            analyzed_at TEXT NOT NULL,
            UNIQUE(ticker, date, source_doc_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_linguistic_ticker_date
        ON linguistic_analysis(ticker, date DESC)
    """)

    # Prediction records: model outputs, confidence, and ground truth
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            predicted_direction TEXT,
            confidence REAL,
            rationale TEXT,
            signal_summary TEXT,
            model_version TEXT DEFAULT '1.0',
            created_at TEXT NOT NULL,
            UNIQUE(ticker, prediction_date, horizon_days)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
        ON predictions(ticker, prediction_date DESC)
    """)

    # Resolution records: actual vs. predicted outcomes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            resolution_date TEXT NOT NULL,
            actual_price REAL NOT NULL,
            predicted_direction TEXT NOT NULL,
            actual_direction TEXT NOT NULL,
            price_change_pct REAL,
            outcome TEXT,
            resolved_at TEXT NOT NULL,
            FOREIGN KEY(prediction_id) REFERENCES predictions(id),
            UNIQUE(prediction_id)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_resolutions_ticker
        ON resolutions(ticker, resolution_date DESC)
    """)

    # RAG corpus: embedded documents (news, SEC filings, research)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rag_corpus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            ticker TEXT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            embedding_model TEXT DEFAULT 'gemini',
            doc_date TEXT,
            ingested_at TEXT NOT NULL,
            metadata TEXT
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rag_ticker
        ON rag_corpus(ticker)
    """)

    conn.commit()
    conn.close()


def reset_db(db_path: Optional[Path] = None) -> None:
    """
    Drop all tables and reinitialize schema (development/testing only).
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    tables = [
        "resolutions",
        "predictions",
        "linguistic_analysis",
        "sentiment_signals",
        "price_history",
        "rag_corpus",
        "schema_meta",
    ]

    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    conn.commit()
    conn.close()

    # Reinitialize
    init_db(db_path)


if __name__ == "__main__":
    init_db()
    print("Schema initialized at", DEFAULT_DB_PATH)
