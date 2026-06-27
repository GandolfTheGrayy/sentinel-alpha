"""
Sentinel Scout — Time-Series SQLite Schema Module

This module defines and manages the SQLite database schema for Sentinel's
core time-series data: historical price snapshots, sentiment signal records,
prediction logs, and post-mortem calibration records. It provides schema
initialization, table creation, and helper functions for data insertion
and querying across all pillars.

Used by Scout (data ingestion), Historian (RAG context enrichment), and
Judge (prediction logging & post-mortem analysis).
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


DB_PATH = Path(__file__).parent.parent.parent / "data" / "sentinel.db"


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database and create all schema tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _create_schema(conn)
    conn.commit()
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    """Internal: Create all required tables in the database."""
    cursor = conn.cursor()
    
    # Price history table — raw OHLCV snapshots from yfinance/stooq
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
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ticker, timestamp)
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_ts ON price_history(ticker, timestamp DESC)")
    
    # News & SEC filings table — ingestion from Scout (news.py, sec_filings.py)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        source TEXT,
        headline TEXT NOT NULL,
        body TEXT,
        url TEXT UNIQUE,
        published_at DATETIME,
        ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        article_type TEXT DEFAULT 'news'
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news_articles(ticker, published_at DESC)")
    
    # Sentiment signals table — output from Linguist (sample_score.py)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        signal_source TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        sentiment_score REAL NOT NULL,
        certainty REAL,
        raw_text TEXT,
        reference_url TEXT,
        timestamp DATETIME NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_ts ON sentiment_signals(ticker, timestamp DESC)")
    
    # RAG context cache — historical lookups from Historian (rag_query.py)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rag_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        context_type TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding_id TEXT,
        relevance_score REAL,
        retrieved_at DATETIME NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rag_ticker ON rag_context(ticker, retrieved_at DESC)")
    
    # Predictions table — logged by Judge (predictor.py)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prediction_date DATETIME NOT NULL,
        predicted_direction TEXT NOT NULL,
        predicted_confidence REAL NOT NULL,
        target_price REAL,
        reasoning TEXT,
        baseline_strategy TEXT,
        sentiment_inputs TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_ticker_date ON predictions(ticker, prediction_date DESC)")
    
    # Post-mortem records — logged by Judge (resolver.py, postmortem.py)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS post_mortems (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        prediction_id INTEGER REFERENCES predictions(id),
        prediction_date DATETIME NOT NULL,
        actual_close REAL NOT NULL,
        actual_direction TEXT NOT NULL,
        predicted_direction TEXT NOT NULL,
        was_correct BOOLEAN NOT NULL,
        price_delta REAL,
        confidence_error REAL,
        anomaly_flag BOOLEAN DEFAULT 0,
        anomaly_reason TEXT,
        heuristic_update TEXT,
        analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_postmortem_ticker ON post_mortems(ticker, prediction_date DESC)")
    
    conn.commit()


def insert_price(
    ticker: str,
    timestamp: datetime,
    close: float,
    open: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: Optional[int] = None,
    db_path: Path = DB_PATH
) -> int:
    """Insert a price snapshot into price_history table; return row ID."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO price_history
        (ticker, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, timestamp.isoformat(), open, high, low, close, volume))
        conn.commit()
        row_id = cursor.lastrowid
        return row_id
    except sqlite3.IntegrityError:
        return -1  # Duplicate
    finally:
        conn.close()


def insert_news_article(
    ticker: str,
    headline: str,
    source: str,
    published_at: datetime,
    body: Optional[str] = None,
    url: Optional[str] = None,
    article_type: str = "news",
    db_path: Path = DB_PATH
) -> int:
    """Insert a news article into news_articles table; return row ID."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO news_articles
        (ticker, source, headline, body, url, published_at, article_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, source, headline, body, url, published_at.isoformat(), article_type))
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return -1
    finally:
        conn.close()


def insert_sentiment_signal(
    ticker: str,
    signal_source: str,
    signal_type: str,
    sentiment_score: float,
    timestamp: datetime,
    certainty: Optional[float] = None,
    raw_text: Optional[str] = None,
    reference_url: Optional[str] = None,
    db_path: Path = DB_PATH
) -> int:
    """Insert a sentiment signal into sentiment_signals table; return row ID."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO sentiment_signals
    (ticker, signal_source, signal_type, sentiment_score, certainty, raw_text, reference_url, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ticker, signal
