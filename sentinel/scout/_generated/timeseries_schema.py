"""
Sentinel Scout — Time-Series Database Schema Module.

This module defines and manages the SQLite schema for storing price history,
sentiment signals, and prediction records. It provides initialization, migration,
and query-building utilities for the core time-series tables that feed the
Historian and Judge agents.

Tables:
  - price_history: OHLCV data from yfinance (symbol, timestamp, open, high, low, close, volume)
  - sentiment_signals: Unified sentiment scores from Reddit, HN, GitHub (symbol, timestamp, source, score, metadata)
  - prediction_records: Daily predictions vs. actual outcomes for backtesting (symbol, date, predicted_move, actual_move, confidence)
  - linguistic_analysis: Cached LLM reasoning outputs (symbol, timestamp, signal_id, analysis_text, certainty_score)
"""

import sqlite3
import os
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


DB_DIR = Path(__file__).parent.parent.parent / "data"
DEFAULT_DB_PATH = DB_DIR / "sentinel.db"


def ensure_db_dir() -> Path:
    """Ensure data directory exists; return path."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return DB_DIR


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open a connection to the Sentinel time-series database."""
    path = db_path or str(DEFAULT_DB_PATH)
    return sqlite3.connect(path)


def init_schema(db_path: Optional[str] = None) -> None:
    """Initialize all tables if they do not exist."""
    ensure_db_dir()
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # Price history table: OHLCV snapshots from yfinance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_price_symbol_time
        ON price_history(symbol, timestamp DESC)
    """)
    
    # Sentiment signals table: unified scores from all Scout sources
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            source TEXT NOT NULL,
            score REAL NOT NULL,
            raw_text TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timestamp, source)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time
        ON sentiment_signals(symbol, timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sentiment_source
        ON sentiment_signals(source)
    """)
    
    # Prediction records table: daily predictions and outcomes for backtesting
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            prediction_date INTEGER NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_magnitude REAL NOT NULL,
            confidence_score REAL NOT NULL,
            actual_direction TEXT,
            actual_magnitude REAL,
            price_open REAL,
            price_close REAL,
            volume INTEGER,
            reasoning TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, prediction_date)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date
        ON prediction_records(symbol, prediction_date DESC)
    """)
    
    # Linguistic analysis cache: reasoning outputs from Linguist agent
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS linguistic_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            signal_id INTEGER,
            analysis_type TEXT NOT NULL,
            analysis_text TEXT NOT NULL,
            certainty_score REAL NOT NULL,
            drift_detected INTEGER DEFAULT 0,
            regulatory_flags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(signal_id) REFERENCES sentiment_signals(id),
            UNIQUE(symbol, timestamp, analysis_type)
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_linguistic_symbol_time
        ON linguistic_analysis(symbol, timestamp DESC)
    """)
    
    conn.commit()
    conn.close()


def insert_price(symbol: str, timestamp: int, open_: float, high: float,
                 low: float, close: float, volume: int, db_path: Optional[str] = None) -> int:
    """Insert a price record; return row id or raise on conflict."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO price_history (symbol, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (symbol, timestamp, open_, high, low, close, volume))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def insert_sentiment_signal(symbol: str, timestamp: int, source: str, score: float,
                           raw_text: Optional[str] = None, metadata: Optional[str] = None,
                           db_path: Optional[str] = None) -> int:
    """Insert a sentiment signal record; return row id."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO sentiment_signals (symbol, timestamp, source, score, raw_text, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, timestamp, source, score, raw_text, metadata))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def insert_prediction(symbol: str, prediction_date: int, predicted_direction: str,
                     predicted_magnitude: float, confidence_score: float,
                     reasoning: Optional[str] = None, db_path: Optional[str] = None) -> int:
    """Insert a prediction record; return row id."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_records
        (symbol, prediction_date, predicted_direction, predicted_magnitude, confidence_score, reasoning)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, prediction_date, predicted_direction, predicted_magnitude, confidence_score, reasoning))
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def update_prediction_outcome(prediction_id: int, actual_direction: str, actual_magnitude: float,
                             price_open: float, price_close: float, volume: int,
                             db_path: Optional[str] = None) -> None:
    """Update a prediction record with actual market outcome."""
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE prediction_records
        SET actual_direction = ?, actual_magnitude = ?, price_open = ?, price_close = ?, volume = ?
        WHERE id = ?
    """, (actual_direction, actual_magnitude, price_open, price_close, volume, prediction_id))
    conn.commit()
    conn.close()


def insert_linguistic_analysis(symbol: str, timestamp: int, analysis_type: str,
                              analysis_text: str, certainty_score: float,
                              signal_id: Optional[int] = None, drift_detected: int = 0,
                              regulatory_flags: Optional[str] =
