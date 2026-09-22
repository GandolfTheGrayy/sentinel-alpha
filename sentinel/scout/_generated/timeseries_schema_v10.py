"""
Sentinel Scout — Time-Series SQLite Schema Module

This module defines and manages the persistent SQLite database schema for Sentinel,
storing price history, sentiment signals, and prediction records. It provides table
creation, connection pooling, and schema migrations for the core data pipeline.

Used by: scout (ingest), historian (lookup), judge (post-mortem analysis).
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime


# Default database location
DEFAULT_DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel_data.db")


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create or connect to Sentinel SQLite database and initialize schema."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create all required tables if they do not exist."""
    cursor = conn.cursor()

    # Price history table
    cursor.execute(
        """
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
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp),
            INDEX idx_ticker_ts (ticker, timestamp)
        )
        """
    )

    # Sentiment signals table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            score REAL NOT NULL,
            raw_text TEXT,
            metadata TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ticker_source_ts (ticker, source, timestamp)
        )
        """
    )

    # Predictions table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date DATETIME NOT NULL,
            prediction_direction TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            rationale TEXT,
            signals_used TEXT,
            model_version TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ticker_date (ticker, prediction_date)
        )
        """
    )

    # Outcomes table (predicted vs. actual)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_date DATETIME NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            actual_direction TEXT,
            price_at_prediction REAL,
            price_at_resolution REAL,
            resolved_at DATETIME,
            accuracy INTEGER,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id),
            INDEX idx_ticker_resolved (ticker, resolved_at)
        )
        """
    )

    # SEC filings metadata table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sec_filings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            cik TEXT,
            accession_number TEXT UNIQUE,
            filing_type TEXT NOT NULL,
            filing_date DATETIME NOT NULL,
            report_period DATETIME,
            url TEXT,
            raw_text TEXT,
            processed INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_ticker_type_date (ticker, filing_type, filing_date)
        )
        """
    )

    conn.commit()


def insert_price(
    conn: sqlite3.Connection,
    ticker: str,
    timestamp: datetime,
    close: float,
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[int] = None,
    source: str = "yfinance",
) -> int:
    """Insert or update a price record; return row id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO price_history
        (ticker, timestamp, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, timestamp, open_, high, low, close, volume, source),
    )
    conn.commit()
    return cursor.lastrowid


def insert_sentiment_signal(
    conn: sqlite3.Connection,
    ticker: str,
    timestamp: datetime,
    source: str,
    signal_type: str,
    score: float,
    raw_text: Optional[str] = None,
    metadata: Optional[str] = None,
) -> int:
    """Insert a sentiment signal record; return row id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO sentiment_signals
        (ticker, timestamp, source, signal_type, score, raw_text, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, timestamp, source, signal_type, score, raw_text, metadata),
    )
    conn.commit()
    return cursor.lastrowid


def insert_prediction(
    conn: sqlite3.Connection,
    ticker: str,
    prediction_date: datetime,
    prediction_direction: str,
    confidence_score: float,
    rationale: Optional[str] = None,
    signals_used: Optional[str] = None,
    model_version: str = "v1.0",
) -> int:
    """Insert a prediction record; return row id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions
        (ticker, prediction_date, prediction_direction, confidence_score,
         rationale, signals_used, model_version)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            prediction_date,
            prediction_direction,
            confidence_score,
            rationale,
            signals_used,
            model_version,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_outcome(
    conn: sqlite3.Connection,
    prediction_id: int,
    ticker: str,
    prediction_date: datetime,
    predicted_direction: str,
    predicted_confidence: float,
    actual_direction: Optional[str] = None,
    price_at_prediction: Optional[float] = None,
    price_at_resolution: Optional[float] = None,
    resolved_at: Optional[datetime] = None,
    accuracy: Optional[int] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert an outcome record; return row id."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO outcomes
        (prediction_id, ticker, prediction_date, predicted_direction,
         predicted_confidence, actual_direction, price_at_prediction,
         price_at_resolution, resolved_at, accuracy, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            ticker,
            prediction_date,
            predicted_direction,
            predicted_confidence,
            actual_direction,
            price_at_prediction,
            price_at_resolution,
            resolved_at,
            accuracy,
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_sec_filing(
    conn: sqlite3.Connection,
    ticker: str,
    filing_type: str,
