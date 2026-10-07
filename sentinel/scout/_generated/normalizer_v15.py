"""
Sentinel Scout Normalizer — unified signal schema and SQLite persistence.

This module provides a canonical SignalRecord dataclass and SQLite-backed
storage for all scout outputs (live prices, news, SEC filings, sentiment).
It bridges heterogeneous scraper outputs into a single queryable schema,
enabling downstream Linguist and Historian modules to operate on normalized data.

Fits into Sentinel pipeline as the Scout pillar's persistence layer:
  scout/live_prices.py → normalizer.store_price_signal()
  scout/news.py → normalizer.store_news_signal()
  scout/sec_filings.py → normalizer.store_filing_signal()
  scout/sentiment.py → normalizer.store_sentiment_signal()
"""

import sqlite3
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import os


DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel.db")


@dataclass
class SignalRecord:
    """
    Canonical schema for all scout signals.
    
    Fields:
      signal_id: UUID or auto-incremented primary key
      ticker: Stock symbol (e.g., "AAPL")
      signal_type: "price" | "news" | "filing" | "sentiment"
      source: Scraper origin (e.g., "yfinance", "sec_edgar", "reddit")
      timestamp: UTC datetime of signal capture
      value: Numeric or categorical value (price, sentiment score, etc.)
      raw_text: Full text context (headline, filing snippet, post body)
      metadata: JSON blob for signal-specific fields
      confidence: 0–1 float; higher = more trustworthy
      normalized: True if processed by Linguist/Historian
    """
    ticker: str
    signal_type: str  # "price" | "news" | "filing" | "sentiment"
    source: str
    timestamp: datetime
    value: Optional[float]
    raw_text: Optional[str]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    normalized: bool = False
    signal_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert SignalRecord to JSON-serializable dict."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        d["metadata"] = json.dumps(d["metadata"])
        return d


class SignalStore:
    """SQLite-backed storage for normalized scout signals."""

    def __init__(self, db_path: str = DB_PATH):
        """Initialize or connect to SQLite database."""
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL,
                raw_text TEXT,
                confidence REAL DEFAULT 1.0,
                metadata TEXT,
                normalized INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
            ON signals(ticker, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_type
            ON signals(signal_type)
        """)
        conn.commit()
        conn.close()

    def store(self, record: SignalRecord) -> int:
        """
        Persist a SignalRecord to SQLite; return signal_id.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals
            (ticker, signal_type, source, timestamp, value, raw_text, confidence, metadata, normalized)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.ticker,
            record.signal_type,
            record.source,
            record.timestamp.isoformat(),
            record.value,
            record.raw_text,
            record.confidence,
            json.dumps(record.metadata),
            int(record.normalized),
        ))
        conn.commit()
        signal_id = cursor.lastrowid
        conn.close()
        return signal_id

    def fetch_by_ticker(self, ticker: str, limit: int = 100) -> List[SignalRecord]:
        """Retrieve all signals for a ticker, ordered by recency."""
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
        return [self._row_to_record(row) for row in rows]

    def fetch_by_type(self, signal_type: str, limit: int = 100) -> List[SignalRecord]:
        """Retrieve all signals of a given type, ordered by recency."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE signal_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (signal_type, limit))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows]

    def fetch_recent(self, hours: int = 24, limit: int = 1000) -> List[SignalRecord]:
        """Retrieve signals from the last N hours."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE timestamp > datetime('now', ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"-{hours} hours", limit))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows]

    def fetch_unnormalized(self, limit: int = 100) -> List[SignalRecord]:
        """Retrieve signals pending Linguist/Historian processing."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM signals
            WHERE normalized = 0
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_record(row) for row in rows]

    def mark_normalized(self, signal_id: int) -> None:
        """Mark a signal as processed by downstream modules."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET normalized = 1 WHERE signal_id = ?", (signal_id,))
        conn.commit()
        conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> SignalRecord:
        """Convert SQLite row to SignalRecord dataclass."""
        return SignalRecord(
            signal_id=row["signal_id"],
            ticker=row["ticker"],
            signal_type=row["signal_type"],
            source=row["source"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            value=row["value"],
            raw_text=row["raw_text"],
            confidence=row["confidence"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            normalized=bool(row["normalized"]),
        )

    def delete_older_than(self,
