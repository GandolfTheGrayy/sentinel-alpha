"""
Sentinel Scout — Time-Series SQLite Schema Module

Provides schema initialization and management for storing price history,
sentiment signals, and prediction records in a persistent SQLite database.
Integrates with other Scout modules (live_prices, news, sec_filings) to
populate tables and with Judge modules (predictor, resolver) to log outcomes.

Tables:
  - price_history: OHLCV candles per ticker
  - sentiment_signals: aggregated sentiment scores (news, Reddit, GitHub)
  - prediction_records: daily predictions with metadata
  - prediction_outcomes: actual vs. predicted moves for post-mortem analysis
"""

import sqlite3
from pathlib import Path
from typing import Optional
import json


DEFAULT_DB_PATH = Path("sentinel_timeseries.db")


def init_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Initialize SQLite database with Sentinel schema; returns open connection."""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Price history table: OHLCV data for each ticker
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
            source TEXT,
            fetched_at TEXT,
            UNIQUE(ticker, date)
        );
    """)

    # Sentiment signals table: aggregated sentiment per ticker per date
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            news_sentiment REAL,
            news_count INTEGER,
            reddit_sentiment REAL,
            reddit_volume INTEGER,
            github_velocity REAL,
            sec_event_type TEXT,
            sec_urgency_level TEXT,
            composite_score REAL,
            computed_at TEXT,
            UNIQUE(ticker, date)
        );
    """)

    # Prediction records table: daily per-ticker predictions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            prediction_horizon_days INTEGER DEFAULT 5,
            predicted_direction TEXT,
            confidence_score REAL,
            price_target REAL,
            reasoning_summary TEXT,
            model_version TEXT,
            baseline_strategy TEXT,
            created_at TEXT,
            UNIQUE(ticker, prediction_date)
        );
    """)

    # Prediction outcomes table: actual results for post-mortem
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            actual_direction TEXT,
            actual_pct_move REAL,
            actual_close_price REAL,
            outcome_date TEXT NOT NULL,
            resolved_at TEXT,
            is_correct BOOLEAN,
            FOREIGN KEY (prediction_id) REFERENCES prediction_records(id),
            UNIQUE(prediction_id)
        );
    """)

    # Indices for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_date ON sentiment_signals(ticker, date DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_ticker_date ON prediction_records(ticker, prediction_date DESC);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome_prediction ON prediction_outcomes(prediction_id);")

    conn.commit()
    return conn


def insert_price_candle(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    source: str = "yfinance",
    fetched_at: Optional[str] = None,
) -> int:
    """Insert or replace a price candle; returns row id."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO price_history
        (ticker, date, open, high, low, close, volume, source, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, date, open_, high, low, close, volume, source, fetched_at))
    conn.commit()
    return cursor.lastrowid


def insert_sentiment_signal(
    conn: sqlite3.Connection,
    ticker: str,
    date: str,
    news_sentiment: Optional[float] = None,
    news_count: Optional[int] = None,
    reddit_sentiment: Optional[float] = None,
    reddit_volume: Optional[int] = None,
    github_velocity: Optional[float] = None,
    sec_event_type: Optional[str] = None,
    sec_urgency_level: Optional[str] = None,
    composite_score: Optional[float] = None,
    computed_at: Optional[str] = None,
) -> int:
    """Insert or replace a sentiment signal record; returns row id."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO sentiment_signals
        (ticker, date, news_sentiment, news_count, reddit_sentiment, reddit_volume,
         github_velocity, sec_event_type, sec_urgency_level, composite_score, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, date, news_sentiment, news_count, reddit_sentiment, reddit_volume,
          github_velocity, sec_event_type, sec_urgency_level, composite_score, computed_at))
    conn.commit()
    return cursor.lastrowid


def insert_prediction_record(
    conn: sqlite3.Connection,
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    confidence_score: float,
    prediction_horizon_days: int = 5,
    price_target: Optional[float] = None,
    reasoning_summary: Optional[str] = None,
    model_version: str = "v1",
    baseline_strategy: Optional[str] = None,
    created_at: Optional[str] = None,
) -> int:
    """Insert a prediction record; returns row id."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO prediction_records
        (ticker, prediction_date, prediction_horizon_days, predicted_direction,
         confidence_score, price_target, reasoning_summary, model_version,
         baseline_strategy, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, prediction_date, prediction_horizon_days, predicted_direction,
          confidence_score, price_target, reasoning_summary, model_version,
          baseline_strategy, created_at))
    conn.commit()
    return cursor.lastrowid


def insert_prediction_outcome(
    conn: sqlite3.Connection,
    prediction_id: int,
    ticker: str,
    prediction_date: str,
    actual_direction: str,
    actual_pct_move: float,
    actual_close_price: float,
    outcome_date: str,
    is_correct: bool,
    resolved_at: Optional[str] = None,
) -> int:
    """Insert a prediction outcome record for post-mortem; returns row id."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO prediction_outcomes
        (prediction_id, ticker, prediction_date, actual_direction,
         actual_pct_move, actual_close_price, outcome_date, is_correct, resolved_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
