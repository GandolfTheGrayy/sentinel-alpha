"""
Sentinel Scout — Time-Series SQLite Schema Module

This module defines and initializes the SQLite schema for Sentinel's time-series
data store. It manages tables for:
  - Price history (OHLCV snapshots from live_prices.py)
  - Sentiment signals (aggregated scores from linguist modules)
  - Prediction records (daily predictions and outcomes from judge modules)
  - RAG corpus metadata (indexed documents for historian lookups)

The schema is designed for fast time-range queries and per-ticker aggregation,
supporting the daily post-mortem and confidence calibration loops. All tables
use UTC timestamps and ticker-based foreign key relationships (logical, not
enforced, to allow flexible ingestion order).

Used by: scout.live_prices (writes), linguist.sample_score (writes),
historian.rag_query (metadata reads), judge.predictor (reads),
judge.resolver (reads/writes outcomes).
"""

import sqlite3
from pathlib import Path
from typing import Optional
import os


DEFAULT_DB_PATH: str = os.getenv("SENTINEL_DB_PATH", "./sentinel_timeseries.db")


def init_schema(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Initialize or connect to the Sentinel time-series database.
    
    Creates all required tables if they don't exist. Returns an open connection
    ready for reads/writes. Caller must close() when done.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Enable foreign keys (logical references only, not enforced)
    conn.execute("PRAGMA foreign_keys = ON")
    
    _create_price_history_table(conn)
    _create_sentiment_signals_table(conn)
    _create_predictions_table(conn)
    _create_rag_corpus_metadata_table(conn)
    _create_prediction_outcomes_table(conn)
    
    conn.commit()
    return conn


def _create_price_history_table(conn: sqlite3.Connection) -> None:
    """Create the OHLCV price history table."""
    conn.execute("""
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
            ingested_at INTEGER NOT NULL,
            UNIQUE(ticker, timestamp),
            INDEX idx_ticker_time (ticker, timestamp DESC)
        )
    """)


def _create_sentiment_signals_table(conn: sqlite3.Connection) -> None:
    """Create the sentiment signals aggregation table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_date INTEGER NOT NULL,
            signal_type TEXT NOT NULL,
            certainty_score REAL NOT NULL,
            hesitation_score REAL NOT NULL,
            source TEXT NOT NULL,
            raw_text TEXT,
            ingested_at INTEGER NOT NULL,
            UNIQUE(ticker, signal_date, signal_type, source),
            INDEX idx_ticker_date (ticker, signal_date DESC),
            INDEX idx_signal_type (signal_type)
        )
    """)


def _create_predictions_table(conn: sqlite3.Connection) -> None:
    """Create the per-ticker daily predictions table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date INTEGER NOT NULL,
            prediction_horizon_days INTEGER DEFAULT 5,
            direction TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            expected_return_pct REAL,
            rationale TEXT,
            baseline_strategy TEXT,
            ingested_at INTEGER NOT NULL,
            UNIQUE(ticker, prediction_date),
            INDEX idx_ticker_date (ticker, prediction_date DESC),
            INDEX idx_direction (direction)
        )
    """)


def _create_rag_corpus_metadata_table(conn: sqlite3.Connection) -> None:
    """Create the RAG corpus metadata table for historian.rag_query indexing."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_corpus_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            ticker TEXT,
            document_id TEXT NOT NULL,
            title TEXT,
            published_at INTEGER,
            ingested_at INTEGER NOT NULL,
            embedding_vector_id TEXT,
            chunk_count INTEGER DEFAULT 1,
            UNIQUE(source_type, document_id),
            INDEX idx_ticker_type (ticker, source_type),
            INDEX idx_published (published_at DESC)
        )
    """)


def _create_prediction_outcomes_table(conn: sqlite3.Connection) -> None:
    """Create the post-mortem outcomes table for resolver.py calibration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_id INTEGER,
            prediction_date INTEGER NOT NULL,
            outcome_date INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_confidence_pct REAL NOT NULL,
            actual_return_pct REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            was_correct INTEGER NOT NULL,
            resolved_at INTEGER NOT NULL,
            UNIQUE(prediction_id, outcome_date),
            INDEX idx_ticker_outcome (ticker, outcome_date DESC),
            INDEX idx_accuracy (was_correct)
        )
    """)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get an open connection to the Sentinel time-series database.
    
    If schema does not exist, initializes it. Caller must close() when done.
    """
    if not Path(db_path).exists():
        return init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def close_connection(conn: sqlite3.Connection) -> None:
    """Safely close a database connection."""
    if conn:
        conn.commit()
        conn.close()


def reset_schema(db_path: str = DEFAULT_DB_PATH, confirm: bool = False) -> None:
    """
    Drop all Sentinel tables and reinitialize schema (destructive).
    
    Requires confirm=True to prevent accidental data loss.
    """
    if not confirm:
        raise ValueError("Reset requires confirm=True to prevent accidental data loss")
    
    conn = sqlite3.connect(db_path)
    for table in [
        "prediction_outcomes",
        "rag_corpus_metadata",
        "predictions",
        "sentiment_signals",
        "price_history"
    ]:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    conn.close()
    
    init_schema(db_path)
