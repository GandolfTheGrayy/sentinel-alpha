"""
Sentinel Scout Normalizer — unified signal schema mapper.

This module normalizes heterogeneous outputs from Scout scrapers (live prices,
news, SEC filings, Reddit/HN sentiment) into a standardized SignalRecord schema
stored in SQLite. Acts as the single source of truth for all ingested signals,
enabling consistent downstream analysis by Linguist and Historian.

Role in Sentinel:
  - Consumes raw outputs from sentinel/scout/* scrapers
  - Validates, deduplicates, and stores in SQLite schema
  - Provides query interface for Historian RAG and Judge post-mortem
  - Tracks signal provenance (source, timestamp, confidence)
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
import os


class SignalSourceType(Enum):
    """Enumeration of all signal sources that feed Sentinel."""
    LIVE_PRICE = "live_price"
    NEWS_HEADLINE = "news_headline"
    SEC_FILING = "sec_filing"
    REDDIT_SENTIMENT = "reddit_sentiment"
    HACKERNEWS_SENTIMENT = "hackernews_sentiment"
    GITHUB_DEVELOPER = "github_developer"


class SignalRecord:
    """
    Unified schema for all ingested signals.
    
    Attributes:
        signal_id: UUID or unique key (auto-generated if None)
        ticker: Stock ticker symbol (e.g., "AAPL")
        signal_type: One of SignalSourceType enum values
        timestamp: ISO 8601 datetime when signal was ingested
        raw_value: Original unparsed value (JSON-serializable)
        normalized_value: Cleaned/extracted value (e.g., price float, sentiment score)
        confidence: Float 0.0–1.0 reflecting signal reliability
        metadata: Dict for source-specific fields (filing_type, headline_url, etc.)
        created_at: ISO 8601 when record was written to DB
    """
    
    def __init__(
        self,
        ticker: str,
        signal_type: str,
        timestamp: str,
        raw_value: Any,
        normalized_value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        signal_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.signal_id = signal_id or self._generate_id(ticker, signal_type, timestamp)
        self.ticker = ticker
        self.signal_type = signal_type
        self.timestamp = timestamp
        self.raw_value = raw_value
        self.normalized_value = normalized_value
        self.confidence = confidence
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.utcnow().isoformat()
    
    @staticmethod
    def _generate_id(ticker: str, signal_type: str, timestamp: str) -> str:
        """Generate a deterministic signal ID from ticker, type, and timestamp."""
        return f"{ticker}_{signal_type}_{timestamp}".replace(" ", "_").replace(":", "")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize SignalRecord to a dictionary."""
        return {
            "signal_id": self.signal_id,
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "confidence": self.confidence,
            "metadata": json.dumps(self.metadata) if self.metadata else "{}",
            "created_at": self.created_at,
        }


class SignalNormalizer:
    """
    Manages SQLite storage and querying of normalized SignalRecords.
    
    Enforces schema consistency, deduplication by signal_id, and provides
    lookups for downstream Historian and Judge modules.
    """
    
    def __init__(self, db_path: str = "sentinel.db"):
        """
        Initialize normalizer and ensure schema exists.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create signals table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                normalized_value TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                UNIQUE(ticker, signal_type, timestamp)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
            ON signals(ticker, timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_type
            ON signals(signal_type)
        """)
        conn.commit()
        conn.close()
    
    def insert_record(self, record: SignalRecord) -> bool:
        """
        Insert or skip a normalized signal record.
        
        Args:
            record: SignalRecord instance to store.
        
        Returns:
            True if inserted, False if duplicate (UNIQUE constraint).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            row = record.to_dict()
            cursor.execute("""
                INSERT INTO signals (
                    signal_id, ticker, signal_type, timestamp,
                    raw_value, normalized_value, confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["signal_id"],
                row["ticker"],
                row["signal_type"],
                row["timestamp"],
                json.dumps(row["raw_value"]),
                json.dumps(row["normalized_value"]),
                row["confidence"],
                row["metadata"],
                row["created_at"],
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
    
    def insert_batch(self, records: List[SignalRecord]) -> Dict[str, int]:
        """
        Insert multiple records and return summary stats.
        
        Args:
            records: List of SignalRecord instances.
        
        Returns:
            Dict with 'inserted' and 'skipped' counts.
        """
        inserted, skipped = 0, 0
        for record in records:
            if self.insert_record(record):
                inserted += 1
            else:
                skipped += 1
        return {"inserted": inserted, "skipped": skipped}
    
    def query_by_ticker(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Retrieve all signals for a given ticker, ordered by timestamp descending.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            List of signal dicts with metadata parsed from JSON.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT signal_id, ticker, signal_type, timestamp,
                   raw_value, normalized_value, confidence, metadata, created_at
            FROM signals
            WHERE ticker = ?
            ORDER BY timestamp DESC
        """, (ticker,))
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                "signal_id": row[0],
                "ticker": row[1],
                "signal_type": row[2],
                "timestamp": row[3],
                "raw_value": json.loads(row[4]),
                "normalized_value": json.loads(row[5]),
                "confidence": row[6],
                "metadata": json.loads(row
