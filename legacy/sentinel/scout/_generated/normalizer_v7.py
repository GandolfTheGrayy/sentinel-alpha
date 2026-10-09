"""
Sentinel Scout Normalizer — unified signal record schema and persistence.

This module provides a canonical SignalRecord dataclass and SQLite persistence
layer that normalizes outputs from all Scout scrapers (live_prices, news,
sec_filings, reddit, github) into a unified schema. Each signal is tagged with
source, timestamp, ticker, and confidence metadata, enabling downstream
Linguist/Historian modules to reason over a consistent record format.

Normalizer runs after each Scout module completes, inserting records into
sentinel.db for RAG ingestion and post-mortem analysis.
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import os


@dataclass
class SignalRecord:
    """
    Canonical unified schema for all sentiment/price/regulatory signals.
    
    Fields:
      - id: auto-increment primary key
      - ticker: stock symbol (e.g., 'AAPL')
      - source: scraper origin ('live_price', 'news', 'sec_filing', 'reddit', 'github')
      - signal_type: category within source ('headline', 'price_jump', '8-K', 'dev_activity')
      - raw_text: original extracted text or JSON payload
      - timestamp: when signal was collected (ISO 8601)
      - event_date: when event occurred (optional, for filings/news)
      - confidence: [0.0, 1.0] normalizer's trust in the signal
      - metadata: JSON blob for source-specific fields (e.g., price, sentiment_score)
      - inserted_at: when record was normalized into DB
    """
    ticker: str
    source: str
    signal_type: str
    raw_text: str
    timestamp: str
    confidence: float
    source_id: Optional[str] = None
    event_date: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    inserted_at: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalRecord to dict, JSON-serializing metadata."""
        d = asdict(self)
        if d.get("metadata"):
            d["metadata"] = json.dumps(d["metadata"])
        return d


class SignalDB:
    """SQLite persistence layer for normalized signal records."""

    def __init__(self, db_path: str = "sentinel.db"):
        """Initialize DB connection and ensure schema exists."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create signals table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_date TEXT,
                    confidence REAL NOT NULL,
                    source_id TEXT,
                    metadata TEXT,
                    inserted_at TEXT NOT NULL,
                    UNIQUE(ticker, source, source_id, timestamp)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_ts
                ON signals(ticker, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON signals(source)
            """)
            conn.commit()

    def insert(self, record: SignalRecord) -> int:
        """
        Insert a single SignalRecord; return inserted row ID.
        
        Handles upsert via UNIQUE constraint on (ticker, source, source_id, timestamp).
        """
        if record.inserted_at is None:
            record.inserted_at = datetime.utcnow().isoformat() + "Z"
        
        metadata_json = json.dumps(record.metadata) if record.metadata else None
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO signals
                    (ticker, source, signal_type, raw_text, timestamp, event_date,
                     confidence, source_id, metadata, inserted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.ticker,
                    record.source,
                    record.signal_type,
                    record.raw_text,
                    record.timestamp,
                    record.event_date,
                    record.confidence,
                    record.source_id,
                    metadata_json,
                    record.inserted_at
                ))
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Duplicate (ticker, source, source_id, timestamp) — skip silently
                return -1

    def insert_batch(self, records: List[SignalRecord]) -> List[int]:
        """Insert multiple records; return list of inserted row IDs (or -1 for duplicates)."""
        return [self.insert(r) for r in records]

    def query_recent(self, ticker: str, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch recent signals for a ticker within N hours.
        
        Returns list of dicts with metadata parsed back from JSON.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM signals
                WHERE ticker = ?
                  AND datetime(timestamp) >= datetime('now', '-{hours} hours')
                ORDER BY timestamp DESC
                LIMIT ?
            """, (ticker, limit))
            rows = cursor.fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    def query_by_source(self, ticker: str, source: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch signals of a specific source type for a ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM signals
                WHERE ticker = ? AND source = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (ticker, source, limit))
            rows = cursor.fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    def query_all_tickers(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch recent signals across all tickers for corpus building."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM signals
                ORDER BY inserted_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        
        result = []
        for row in rows:
            d = dict(row)
            if d.get("metadata"):
                d["metadata"] = json.loads(d["metadata"])
            result.append(d)
        return result

    def delete_older_than_days(self, days: int) -> int:
        """Delete signals older than N days; return count deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM signals
                WHERE datetime(timestamp) < datetime('now', ? || ' days')
            """, (f"-{days}",))
            conn.commit()
            return cursor.rowcount
