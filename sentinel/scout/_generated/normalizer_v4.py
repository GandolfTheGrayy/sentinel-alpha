"""
Sentinel Scout Normalizer — unified signal record schema and persistence.

This module provides a canonical SignalRecord dataclass and SQLite-backed
storage layer that normalizes outputs from all scrapers (live_prices, news,
sec_filings, reddit sentiment, GitHub signals) into a single queryable schema.
The normalizer acts as the single source of truth for raw signals before
they reach the Linguist and Historian pillars.

Role in Sentinel:
  - Accept heterogeneous signal dicts from Scout scrapers.
  - Validate and coerce into SignalRecord instances.
  - Persist to SQLite with idempotent upsert semantics.
  - Expose query interface for downstream reasoning (Linguist, Historian).
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import os


class SignalType(str, Enum):
    """Enumeration of signal types recognized by Sentinel."""
    PRICE = "price"
    NEWS = "news"
    SEC_FILING = "sec_filing"
    REDDIT_SENTIMENT = "reddit_sentiment"
    GITHUB_ACTIVITY = "github_activity"
    EARNINGS_CALENDAR = "earnings_calendar"


class SignalSentiment(str, Enum):
    """Sentiment polarity for qualitative signals."""
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    UNKNOWN = "unknown"


@dataclass
class SignalRecord:
    """
    Canonical signal record: normalized output from any Scout scraper.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL").
        signal_type: Category from SignalType enum.
        timestamp: ISO 8601 UTC datetime when signal was observed.
        raw_value: Numeric or string value (price, sentiment score, headline text).
        sentiment: Polarity for qualitative signals (bullish/neutral/bearish).
        source: Origin identifier (e.g., "yfinance", "sec_edgar", "reddit").
        source_id: Unique identifier within source (e.g., filing accession, post URL).
        metadata: JSON-serializable dict for scraper-specific fields.
        confidence: Float [0, 1] estimate of signal reliability.
        created_at: ISO 8601 UTC timestamp when record was inserted into Sentinel.
    """
    ticker: str
    signal_type: str
    timestamp: str
    raw_value: Any
    sentiment: str
    source: str
    source_id: str
    metadata: Dict[str, Any]
    confidence: float
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary, JSON-serializing metadata and raw_value."""
        d = asdict(self)
        d["metadata"] = json.dumps(d["metadata"])
        d["raw_value"] = json.dumps(d["raw_value"]) if not isinstance(d["raw_value"], str) else d["raw_value"]
        return d


class SignalNormalizer:
    """SQLite-backed normalizer for heterogeneous Scout scraper outputs."""

    def __init__(self, db_path: str = "sentinel_signals.db"):
        """
        Initialize normalizer and ensure SQLite schema exists.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create signal_records table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_value TEXT,
                sentiment TEXT,
                source TEXT NOT NULL,
                source_id TEXT NOT NULL,
                metadata TEXT,
                confidence REAL,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, signal_type, source, source_id, timestamp)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_type
            ON signal_records(ticker, signal_type, created_at DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_id
            ON signal_records(source, source_id)
        """)
        conn.commit()
        conn.close()

    def normalize(self, raw_signal: Dict[str, Any]) -> SignalRecord:
        """
        Coerce raw scraper output dict into a validated SignalRecord.
        
        Args:
            raw_signal: Dict with keys: ticker, signal_type, timestamp, raw_value,
                       sentiment (optional), source, source_id, metadata (optional),
                       confidence (optional).
        
        Returns:
            SignalRecord instance with sensible defaults for missing fields.
        
        Raises:
            ValueError: If required fields are missing.
        """
        required = {"ticker", "signal_type", "timestamp", "source", "source_id"}
        if not required.issubset(raw_signal.keys()):
            missing = required - set(raw_signal.keys())
            raise ValueError(f"Missing required fields: {missing}")

        return SignalRecord(
            ticker=raw_signal["ticker"].upper(),
            signal_type=raw_signal["signal_type"],
            timestamp=raw_signal["timestamp"],
            raw_value=raw_signal.get("raw_value"),
            sentiment=raw_signal.get("sentiment", SignalSentiment.UNKNOWN.value),
            source=raw_signal["source"],
            source_id=raw_signal["source_id"],
            metadata=raw_signal.get("metadata", {}),
            confidence=raw_signal.get("confidence", 0.5),
            created_at=datetime.utcnow().isoformat() + "Z"
        )

    def store(self, record: SignalRecord) -> bool:
        """
        Upsert signal record into SQLite (idempotent by source + source_id).
        
        Args:
            record: SignalRecord to persist.
        
        Returns:
            True if inserted or updated; False on constraint violation.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            record_dict = record.to_dict()
            placeholders = ", ".join(["?"] * len(record_dict))
            columns = ", ".join(record_dict.keys())
            cursor.execute(f"""
                INSERT OR REPLACE INTO signal_records ({columns})
                VALUES ({placeholders})
            """, tuple(record_dict.values()))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def store_batch(self, records: List[SignalRecord]) -> int:
        """
        Upsert multiple records in a single transaction.
        
        Args:
            records: List of SignalRecord instances.
        
        Returns:
            Number of records successfully inserted or updated.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        inserted = 0
        try:
            for record in records:
                record_dict = record.to_dict()
                placeholders = ", ".join(["?"] * len(record_dict))
                columns = ", ".join(record_dict.keys())
                try:
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO signal_records ({columns})
                        VALUES ({placeholders})
                    """, tuple(record_dict.values()))
                    inserted += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        finally:
            conn.close()
        return inserted

    def query_by_ticker(self, ticker: str, signal_type: Optional[str] = None,
                        limit: int = 100) -> List[SignalRecord]:
        """
        Fetch recent signals for a ticker, optionally filtered by signal_type.
        
        Args:
            ticker: Stock
