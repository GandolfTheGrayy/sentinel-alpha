"""
Data normalizer for Sentinel Scout.

Maps heterogeneous outputs from live_prices, news, sec_filings, Reddit/HN sentiment
scrapers into a unified SignalRecord schema, persisted in SQLite. Acts as the
canonical signal intake layer before downstream Linguist and Historian processing.

Responsibilities:
  - Define SignalRecord: ticker, source, signal_type, raw_text, metadata, ts
  - Provide normalize_* functions for each scraper type
  - Manage SQLite schema initialization and upsert logic
  - Expose query helpers for retrieval by ticker/source/date range
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class SignalSourceType(str, Enum):
    """Enumeration of canonical signal sources in Sentinel Scout."""
    PRICE_LIVE = "price_live"
    NEWS_HEADLINE = "news_headline"
    SEC_FILING = "sec_filing"
    REDDIT = "reddit"
    HACKERNEWS = "hackernews"
    GITHUB = "github_developer"


class SignalType(str, Enum):
    """Enumeration of signal interpretation categories."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    ANOMALY = "anomaly"
    REGULATORY = "regulatory"
    TECHNICAL = "technical"


@dataclass
class SignalRecord:
    """Unified schema for all sentiment and price signals in Sentinel."""
    ticker: str
    source: SignalSourceType
    signal_type: SignalType
    raw_text: str
    confidence: float
    metadata: dict[str, Any]
    timestamp: datetime
    id: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert SignalRecord to dict, serializing enums and datetime."""
        d = asdict(self)
        d["source"] = self.source.value
        d["signal_type"] = self.signal_type.value
        d["timestamp"] = self.timestamp.isoformat()
        d["metadata"] = json.dumps(self.metadata)
        return d


class SignalDatabase:
    """SQLite persistence layer for normalized signal records."""

    def __init__(self, db_path: str = "sentinel_signals.db"):
        """Initialize or open SignalDatabase at db_path."""
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create signal_records table if not present."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, source, signal_type, timestamp)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_ts
            ON signal_records (ticker, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_ts
            ON signal_records (source, timestamp DESC)
        """)
        self.conn.commit()

    def upsert_signal(self, record: SignalRecord) -> int:
        """Insert or replace a signal record; return its id."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO signal_records
            (ticker, source, signal_type, raw_text, confidence, metadata, timestamp, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.ticker,
            record.source.value,
            record.signal_type.value,
            record.raw_text,
            record.confidence,
            json.dumps(record.metadata),
            record.timestamp.isoformat(),
            now
        ))
        self.conn.commit()
        return cursor.lastrowid

    def query_by_ticker(self, ticker: str, limit: int = 100) -> list[SignalRecord]:
        """Retrieve signals for a ticker, ordered by recency."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, ticker, source, signal_type, raw_text, confidence, metadata, timestamp
            FROM signal_records
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (ticker, limit))
        rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    def query_by_source(self, source: SignalSourceType, limit: int = 100) -> list[SignalRecord]:
        """Retrieve signals from a specific source, ordered by recency."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, ticker, source, signal_type, raw_text, confidence, metadata, timestamp
            FROM signal_records
            WHERE source = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (source.value, limit))
        rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    def query_by_date_range(self, start: datetime, end: datetime, ticker: Optional[str] = None) -> list[SignalRecord]:
        """Retrieve signals within a date range, optionally filtered by ticker."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        if ticker:
            cursor.execute("""
                SELECT id, ticker, source, signal_type, raw_text, confidence, metadata, timestamp
                FROM signal_records
                WHERE ticker = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (ticker, start.isoformat(), end.isoformat()))
        else:
            cursor.execute("""
                SELECT id, ticker, source, signal_type, raw_text, confidence, metadata, timestamp
                FROM signal_records
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            """, (start.isoformat(), end.isoformat()))
        rows = cursor.fetchall()
        return [_row_to_record(row) for row in rows]

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def _row_to_record(row: tuple) -> SignalRecord:
    """Convert a database row tuple to SignalRecord."""
    (record_id, ticker, source, signal_type, raw_text, confidence, metadata_json, timestamp_str) = row
    return SignalRecord(
        id=record_id,
        ticker=ticker,
        source=SignalSourceType(source),
        signal_type=SignalType(signal_type),
        raw_text=raw_text,
        confidence=confidence,
        metadata=json.loads(metadata_json),
        timestamp=datetime.fromisoformat(timestamp_str)
    )


def normalize_price_record(ticker: str, price: float, change_pct: float, timestamp: datetime) -> SignalRecord:
    """Normalize a live price update into a SignalRecord."""
    signal_type = SignalType.TECHNICAL
    if change_pct > 2.0:
        signal_type = SignalType.BULLISH
    elif change_pct < -2.0:
        signal_type = SignalType.BEAR
