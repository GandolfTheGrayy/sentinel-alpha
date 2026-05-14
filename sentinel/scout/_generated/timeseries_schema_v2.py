"""
Sentinel Scout — Time-Series SQLite Schema Module

Provides a base SQLite schema for persisting price history, sentiment signals,
and prediction records. This module creates and manages tables that feed the
RAG historian and post-mortem judge modules.

Tables:
  - price_history: OHLCV data from live_prices.py
  - sentiment_signals: Aggregated sentiment scores (Reddit, news, GitHub)
  - prediction_records: Claude predictions + baseline strategies
  - signal_metadata: Confidence scores and signal source tracking

Used by: historian/rag_query.py (vector embeddings), judge/postmortem.py (backtest),
         pipeline.py (daily ETL).
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


# Default database path — can be overridden via env var
DEFAULT_DB_PATH = os.getenv(
    "SENTINEL_DB_PATH",
    str(Path(__file__).parent.parent.parent / "data" / "sentinel.db")
)


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open or create the Sentinel time-series database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create all Sentinel schema tables if they don't exist."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    # Price history table — OHLCV data from yfinance/stooq
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS price_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL NOT NULL,
        volume INTEGER,
        source TEXT DEFAULT 'yfinance',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date)
    )
    """)

    # Sentiment signals table — aggregated sentiment from multiple sources
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        sentiment_source TEXT NOT NULL,
        score REAL NOT NULL,
        raw_count INTEGER,
        confidence REAL DEFAULT 0.5,
        note TEXT,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date, sentiment_source)
    )
    """)

    # Prediction records table — Claude predictions + baseline strategies
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prediction_date DATE NOT NULL,
        strategy TEXT NOT NULL,
        direction TEXT,
        confidence REAL,
        target_price REAL,
        reasoning TEXT,
        model_version TEXT DEFAULT 'sonnet-4-6',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, prediction_date, strategy)
    )
    """)

    # Signal metadata table — track confidence, sources, anomalies
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signal_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        date DATE NOT NULL,
        signal_type TEXT NOT NULL,
        linguistic_certainty REAL,
        linguistic_drift REAL,
        regulatory_whisper_flag INTEGER DEFAULT 0,
        anomaly_score REAL DEFAULT 0.0,
        composite_confidence REAL,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, date, signal_type)
    )
    """)

    # Post-mortem resolution table — actual vs predicted, backtest results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS postmortem_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prediction_date DATE NOT NULL,
        actual_close REAL,
        actual_direction TEXT,
        predicted_direction TEXT,
        predicted_confidence REAL,
        strategy TEXT NOT NULL,
        outcome TEXT,
        pnl_pct REAL,
        resolved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, prediction_date, strategy)
    )
    """)

    # Create indices for fast lookups
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_price_ticker_date
    ON price_history(ticker, date DESC)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date
    ON sentiment_signals(ticker, date DESC)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_prediction_ticker_date
    ON prediction_records(ticker, prediction_date DESC)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_signal_metadata_ticker_date
    ON signal_metadata(ticker, date DESC)
    """)

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_postmortem_ticker_date
    ON postmortem_records(ticker, prediction_date DESC)
    """)

    conn.commit()
    conn.close()


def insert_price_record(
    ticker: str,
    date: str,
    open_: Optional[float],
    high: Optional[float],
    low: Optional[float],
    close: float,
    volume: Optional[int],
    source: str = "yfinance",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Insert or replace a price history record."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO price_history
    (ticker, date, open, high, low, close, volume, source)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, date, open_, high, low, close, volume, source))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


def insert_sentiment_signal(
    ticker: str,
    date: str,
    sentiment_source: str,
    score: float,
    raw_count: Optional[int] = None,
    confidence: float = 0.5,
    note: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Insert a sentiment signal record (Reddit, news, GitHub, etc)."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO sentiment_signals
    (ticker, date, sentiment_source, score, raw_count, confidence, note)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ticker, date, sentiment_source, score, raw_count, confidence, note))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


def insert_prediction_record(
    ticker: str,
    prediction_date: str,
    strategy: str,
    direction: Optional[str],
    confidence: Optional[float],
    target_price: Optional[float] = None,
    reasoning: Optional[str] = None,
    model_version: str = "sonnet-4-6",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Insert a prediction record from Claude or baseline strategy."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO prediction_records
    (ticker, prediction_date, strategy, direction, confidence, target_price, reasoning, model_version)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, prediction_date, strategy, direction, confidence, target_price, reasoning, model_version))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last
