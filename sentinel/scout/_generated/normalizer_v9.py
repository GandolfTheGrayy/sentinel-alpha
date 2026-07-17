"""
Sentinel Scout Normalizer — unified signal record schema and SQLite persistence.

This module provides the SignalRecord dataclass and database layer that normalize
outputs from all Scout scrapers (live_prices, news, sec_filings, reddit, github)
into a single queryable schema. All heterogeneous signals flow through here before
being passed to the Linguist and Historian pillars.

Schema:
  - signal_id: unique identifier (ticker + source + timestamp hash)
  - ticker: stock symbol
  - source: scraper origin (price, news, sec, reddit, github)
  - signal_type: granular category (price_movement, headline, 8k_filing, etc.)
  - value: normalized numeric or text payload
  - confidence: 0.0–1.0 scraper-supplied confidence
  - timestamp: UTC when signal was captured
  - raw_metadata: JSON blob for source-specific fields
"""

import sqlite3
import json
import hashlib
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import os


@dataclass
class SignalRecord:
    """
    Unified signal record schema for all Scout data sources.
    """
    signal_id: str
    ticker: str
    source: str  # 'price', 'news', 'sec', 'reddit', 'github'
    signal_type: str  # 'price_movement', 'headline', '8k_filing', etc.
    value: str  # normalized to string; caller interprets type
    confidence: float  # 0.0–1.0
    timestamp: str  # ISO 8601 UTC
    raw_metadata: str = field(default_factory=lambda: "{}")  # JSON string

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SignalRecord":
        """Construct record from dictionary."""
        return SignalRecord(**d)


class SignalDatabase:
    """SQLite persistence layer for normalized signals."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize database connection.
        Uses SENTINEL_DB_PATH env var or default to ./sentinel.db.
        """
        if db_path is None:
            db_path = os.environ.get("SENTINEL_DB_PATH", "./sentinel.db")
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create signals table if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                raw_metadata TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
            ON signals (ticker, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_type
            ON signals (source, signal_type)
        """)
        conn.commit()
        conn.close()

    def insert_signal(self, record: SignalRecord) -> bool:
        """
        Insert a signal record; return True if successful, False if duplicate.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO signals
                (signal_id, ticker, source, signal_type, value, confidence, timestamp, raw_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.signal_id,
                record.ticker,
                record.source,
                record.signal_type,
                record.value,
                record.confidence,
                record.timestamp,
                record.raw_metadata
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def insert_batch(self, records: List[SignalRecord]) -> int:
        """
        Insert multiple signals; return count inserted.
        Silently skips duplicates.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        inserted = 0
        for record in records:
            try:
                cursor.execute("""
                    INSERT INTO signals
                    (signal_id, ticker, source, signal_type, value, confidence, timestamp, raw_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.signal_id,
                    record.ticker,
                    record.source,
                    record.signal_type,
                    record.value,
                    record.confidence,
                    record.timestamp,
                    record.raw_metadata
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        conn.close()
        return inserted

    def query_by_ticker(self, ticker: str, limit: int = 100) -> List[SignalRecord]:
        """
        Fetch recent signals for a ticker (ordered by timestamp DESC).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE ticker = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (ticker, limit))
        rows = cursor.fetchall()
        conn.close()
        return [SignalRecord(**dict(row)) for row in rows]

    def query_by_source(
        self,
        source: str,
        ticker: Optional[str] = None,
        limit: int = 100
    ) -> List[SignalRecord]:
        """
        Fetch signals by source, optionally filtered by ticker.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if ticker:
            cursor.execute("""
                SELECT * FROM signals
                WHERE source = ? AND ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (source, ticker, limit))
        else:
            cursor.execute("""
                SELECT * FROM signals
                WHERE source = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (source, limit))
        rows = cursor.fetchall()
        conn.close()
        return [SignalRecord(**dict(row)) for row in rows]

    def query_time_range(
        self,
        ticker: str,
        start_ts: str,
        end_ts: str
    ) -> List[SignalRecord]:
        """
        Fetch signals for ticker within ISO 8601 timestamp range (inclusive).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE ticker = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp DESC
        """, (ticker, start_ts, end_ts))
        rows = cursor.fetchall()
        conn.close()
        return [SignalRecord(**dict(row)) for row in rows]

    def get_signal(self, signal_id: str) -> Optional[SignalRecord]:
        """Fetch a single signal by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM signals WHERE signal_id = ?", (signal_id,))
        row = cursor.fetchone()
        conn.close()
        return SignalRecord(**dict(row)) if row else None

    def delete_older_than(self, days: int) -> int:
