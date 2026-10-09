"""
Time-series SQLite schema module for Sentinel Sentiment Engine.

This module defines and initializes the core SQLite database schema used to
persist price history, sentiment signals, prediction records, and post-mortem
calibration data across the Sentinel pipeline. It serves as the single source
of truth for the data model, enabling consistent storage and retrieval of
time-series intelligence across Scout (ingestion), Historian (RAG context),
Judge (prediction), and daily post-mortem workflows.

The schema includes tables for:
  - price_history: OHLCV candles from yfinance/stooq
  - sentiment_signals: Reddit/HN/news sentiment scores with timestamps
  - sec_filings: Metadata and extracted text from SEC filings
  - predictions: Predicted price movements with reasoning and confidence
  - post_mortems: Daily calibration records comparing predicted vs. actual moves
"""

import sqlite3
from pathlib import Path
from typing import Optional
import os


def get_db_path(db_dir: Optional[str] = None) -> str:
    """Return or create the path to the Sentinel time-series database file."""
    if db_dir is None:
        db_dir = os.environ.get("SENTINEL_DB_DIR", str(Path.home() / ".sentinel"))
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    return str(Path(db_dir) / "sentinel.db")


def init_schema(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Initialize or open the Sentinel time-series SQLite database and create all required tables.

    Returns a connection object that can be used to query or insert records.
    Idempotent: calling multiple times on the same database file is safe.
    """
    if db_path is None:
        db_path = get_db_path()
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Price history table: OHLCV candles
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            source TEXT DEFAULT 'yfinance',
            fetched_at TEXT NOT NULL,
            UNIQUE(ticker, date)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date DESC)"
    )

    # Sentiment signals table: Reddit, HN, news sentiment
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            score REAL NOT NULL,
            raw_text TEXT,
            timestamp TEXT NOT NULL,
            url TEXT,
            ingested_at TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time ON sentiment_signals(ticker, timestamp DESC)"
    )

    # SEC filings metadata and text
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cik TEXT NOT NULL,
            form_type TEXT NOT NULL,
            accession_number TEXT UNIQUE NOT NULL,
            filing_date TEXT NOT NULL,
            period_end TEXT,
            extracted_text TEXT,
            url TEXT,
            fetched_at TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_sec_ticker_date ON sec_filings(ticker, filing_date DESC)"
    )

    # Predictions: per-ticker daily predictions with reasoning
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_magnitude REAL,
            confidence REAL NOT NULL,
            reasoning TEXT,
            signals_used TEXT,
            rag_context TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(ticker, prediction_date)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date ON predictions(ticker, prediction_date DESC)"
    )

    # Post-mortems: daily calibration and anomaly detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS post_mortems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            evaluation_date TEXT NOT NULL,
            predicted_direction TEXT,
            predicted_magnitude REAL,
            predicted_confidence REAL,
            actual_close REAL NOT NULL,
            actual_open REAL NOT NULL,
            actual_high REAL NOT NULL,
            actual_low REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            actual_magnitude REAL NOT NULL,
            correct INTEGER NOT NULL,
            magnitude_error REAL,
            anomaly_flags TEXT,
            calibration_notes TEXT,
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_postmortem_ticker_date ON post_mortems(ticker, evaluation_date DESC)"
    )

    # Linguistic signals: certainty/hesitation markers and drift detection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS linguistic_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            certainty_score REAL NOT NULL,
            hesitation_score REAL NOT NULL,
            drift_from_baseline REAL,
            text_sample TEXT,
            analyzed_at TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_linguistic_ticker ON linguistic_signals(ticker, analyzed_at DESC)"
    )

    # Regulatory whispers: SEC filings with anomalous language patterns
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regulatory_whispers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            accession_number TEXT UNIQUE NOT NULL,
            form_type TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            flagged_text TEXT,
            interpretation TEXT,
            detected_at TEXT NOT NULL
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_whispers_ticker ON regulatory_whispers(ticker, detected_at DESC)"
    )

    conn.commit()
    return conn


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open and return an existing Sentinel database connection."""
    if db_path is None:
        db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def reset_database(db_path: Optional[str] = None) -> None:
    """
    Drop all tables and reinitialize the schema.

    Used for testing and reset workflows. Use with extreme caution.
    """
    if db_path is None:
        db_path = get_db_path()
    
    if not Path(db_path).exists():
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop all tables in reverse dependency order
    tables = [
        "post_mortems",
        "predictions",
        "regulatory_whispers",
        "linguistic_signals",
        "sec_filings",
        "sentiment_signals",
        "price_history",
    ]
    
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
    
    conn.commit()
    conn.close()
    
    # Reinitialize
    init_schema(db_path)


if __name__ == "__main__":
    db = init_schema()
    print(f"✓ Sentinel schema initialized at {get_db_path()}")
    db.close()
