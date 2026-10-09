"""
Sentinel Scout Normalizer — unified signal record schema and storage.

This module provides a canonical SignalRecord dataclass and SQLite persistence
layer that normalizes outputs from all scrapers (live_prices, news, sec_filings,
Reddit sentiment, GitHub signals, etc.) into a single, queryable format.

The normalizer bridges the Scout pillar's diverse data sources and feeds
consistent, timestamped records into the Historian (RAG) and Linguist (reasoning)
pillars for downstream analysis.
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class SignalRecord:
    """
    Canonical signal record unifying all Scout data sources.
    
    Fields map to: live prices, news headlines, SEC filings, social sentiment,
    developer health metrics, and any future signals. All timestamps are ISO 8601.
    """
    ticker: str
    signal_type: str  # "price", "news", "sec_filing", "reddit", "github", "earnings_calendar"
    timestamp: str  # ISO 8601
    source: str  # "yfinance", "newsapi", "sec_edgar", "reddit", "github", "calendar"
    
    # Core signal payload (varies by type)
    value: Optional[float] = None  # price, sentiment score, dev velocity
    text: Optional[str] = None  # headline, filing summary, comment
    
    # Metadata for traceability and weighting
    confidence: Optional[float] = None  # 0.0–1.0; higher = more reliable
    url: Optional[str] = None  # source link or SEC filing ID
    raw_metadata: Optional[str] = None  # JSON blob for source-specific fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary, JSON-serializing raw_metadata."""
        d = asdict(self)
        if self.raw_metadata and isinstance(self.raw_metadata, dict):
            d["raw_metadata"] = json.dumps(self.raw_metadata)
        return d


class SignalNormalizer:
    """SQLite-backed normalizer for unified signal storage and retrieval."""
    
    def __init__(self, db_path: str = "sentinel_signals.db"):
        """Initialize SQLite connection and create schema if needed."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create signal_records table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                value REAL,
                text TEXT,
                confidence REAL,
                url TEXT,
                raw_metadata TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, signal_type, timestamp, source)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
            ON signal_records(ticker, timestamp DESC)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_type_source 
            ON signal_records(signal_type, source)
        """)
        self.conn.commit()
    
    def insert_record(self, record: SignalRecord) -> int:
        """
        Insert a signal record; return row ID or raise on duplicate.
        
        Duplicates (same ticker, signal_type, timestamp, source) are rejected
        to prevent double-counting. Use upsert_record() to overwrite.
        """
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO signal_records 
                (ticker, signal_type, timestamp, source, value, text, 
                 confidence, url, raw_metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.ticker,
                record.signal_type,
                record.timestamp,
                record.source,
                record.value,
                record.text,
                record.confidence,
                record.url,
                json.dumps(record.raw_metadata) if record.raw_metadata else None,
                datetime.utcnow().isoformat()
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError as e:
            raise ValueError(f"Duplicate signal or constraint violation: {e}")
    
    def upsert_record(self, record: SignalRecord) -> int:
        """
        Insert or replace a signal record; always succeeds (overwrites duplicates).
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO signal_records 
            (ticker, signal_type, timestamp, source, value, text, 
             confidence, url, raw_metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.ticker,
            record.signal_type,
            record.timestamp,
            record.source,
            record.value,
            record.text,
            record.confidence,
            record.url,
            json.dumps(record.raw_metadata) if record.raw_metadata else None,
            datetime.utcnow().isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_batch(self, records: List[SignalRecord]) -> int:
        """
        Bulk insert records; skip duplicates silently.
        
        Returns count of successfully inserted rows.
        """
        cursor = self.conn.cursor()
        inserted = 0
        for record in records:
            try:
                cursor.execute("""
                    INSERT INTO signal_records 
                    (ticker, signal_type, timestamp, source, value, text, 
                     confidence, url, raw_metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.ticker,
                    record.signal_type,
                    record.timestamp,
                    record.source,
                    record.value,
                    record.text,
                    record.confidence,
                    record.url,
                    json.dumps(record.raw_metadata) if record.raw_metadata else None,
                    datetime.utcnow().isoformat()
                ))
                inserted += 1
            except sqlite3.IntegrityError:
                pass  # Skip duplicate
        self.conn.commit()
        return inserted
    
    def query_by_ticker(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch all signal records for a ticker, most recent first.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM signal_records 
            WHERE ticker = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (ticker, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def query_by_type(self, signal_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch all records of a specific signal type, most recent first.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM signal_records 
            WHERE signal_type = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (signal_type, limit))
        return [dict(row) for row in cursor.fetchall()]
    
    def query_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch all records from a specific source, most recent first.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
