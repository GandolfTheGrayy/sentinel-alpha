"""
Sentinel Scout — Time-Series SQLite Schema Module

Manages persistent storage for price history, sentiment signals, and prediction
records. Provides schema initialization, connection pooling, and CRUD helpers
for the core Sentinel data flow.

Used by Scout ingesters (live_prices, sec_filings, news) and Judge post-mortems
to store and retrieve historical signals for RAG and backtesting.
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime


DB_PATH = os.getenv("SENTINEL_DB_PATH", str(Path.home() / ".sentinel" / "timeseries.db"))


def ensure_db_dir() -> None:
    """Create database directory if it does not exist."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row factory enabled."""
    ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    """Initialize all Sentinel schema tables (idempotent)."""
    ensure_db_dir()
    conn = get_connection()
    cursor = conn.cursor()

    # Price history table — daily/intraday snapshots from yfinance/stooq.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL NOT NULL,
            volume INTEGER,
            source TEXT DEFAULT 'yfinance',
            UNIQUE(ticker, timestamp),
            INDEX idx_ticker_time (ticker, timestamp DESC)
        )
    """)

    # Sentiment signals table — aggregated sentiment from news, Reddit, etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_date DATETIME NOT NULL,
            source TEXT NOT NULL,
            score REAL NOT NULL,
            headline TEXT,
            url TEXT,
            confidence REAL DEFAULT 0.5,
            ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, signal_date, source, headline),
            INDEX idx_ticker_signal (ticker, signal_date DESC)
        )
    """)

    # Regulatory signals — SEC filings, earnings dates, etc.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS regulatory_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            event_date DATETIME NOT NULL,
            event_type TEXT NOT NULL,
            filing_url TEXT,
            summary TEXT,
            extracted_text TEXT,
            embedding_id TEXT,
            ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, event_date, event_type),
            INDEX idx_ticker_event (ticker, event_date DESC)
        )
    """)

    # Prediction records table — daily predictions and confidence scores.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date DATETIME NOT NULL,
            direction TEXT NOT NULL,
            confidence REAL NOT NULL,
            predicted_move_pct REAL,
            reasoning TEXT,
            strategy_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ticker_pred (ticker, prediction_date DESC)
        )
    """)

    # Actuals & post-mortems — resolved predictions vs. real market moves.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_date DATETIME NOT NULL,
            actual_date DATETIME NOT NULL,
            actual_move_pct REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            was_correct INTEGER DEFAULT 0,
            resolved_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            postmortem_notes TEXT,
            FOREIGN KEY (prediction_id) REFERENCES prediction_records(id),
            UNIQUE(prediction_id),
            INDEX idx_ticker_actual (ticker, actual_date DESC)
        )
    """)

    # Model calibration table — heuristic refinement and anomaly flags.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            calibration_date DATETIME NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            anomaly_flag INTEGER DEFAULT 0,
            notes TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(calibration_date, metric_name),
            INDEX idx_calibration (calibration_date DESC)
        )
    """)

    conn.commit()
    conn.close()


def insert_price(
    ticker: str, timestamp: datetime, close: float, open_: Optional[float] = None,
    high: Optional[float] = None, low: Optional[float] = None,
    volume: Optional[int] = None, source: str = "yfinance"
) -> int:
    """Insert or upsert a single price record; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO price_history
            (ticker, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, timestamp) DO UPDATE SET
                open=excluded.open, high=excluded.high, low=excluded.low,
                close=excluded.close, volume=excluded.volume, source=excluded.source
        """, (ticker, timestamp, open_, high, low, close, volume, source))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_sentiment_signal(
    ticker: str, signal_date: datetime, source: str, score: float,
    headline: Optional[str] = None, url: Optional[str] = None,
    confidence: float = 0.5
) -> int:
    """Insert a sentiment signal; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO sentiment_signals
            (ticker, signal_date, source, score, headline, url, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, signal_date, source, headline) DO UPDATE SET
                score=excluded.score, confidence=excluded.confidence
        """, (ticker, signal_date, source, score, headline, url, confidence))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_regulatory_signal(
    ticker: str, event_date: datetime, event_type: str,
    filing_url: Optional[str] = None, summary: Optional[str] = None,
    extracted_text: Optional[str] = None, embedding_id: Optional[str] = None
) -> int:
    """Insert a regulatory event; return row ID."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO regulatory_signals
            (ticker, event_date, event_type, filing_url, summary, extracted_text, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, event_date, event_type) DO UPDATE SET
                filing_url=excluded.filing_url, summary=excluded.summary,
                extracted_text=excluded.extracted_text
        """, (ticker, event_date, event_type, filing_url, summary, extracted_text, embedding_id))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def insert_prediction(
    ticker: str, prediction_date: datetime, direction: str, confidence: float,
    predicted_move_pct: Optional[float] = None,
