"""
Sentinel Scout Normalizer — Unified Signal Record Schema & SQLite Storage

This module provides a data normalization layer that maps outputs from all
Scout scrapers (live_prices, news, sec_filings, reddit, github) into a
unified SignalRecord schema. Records are persisted to SQLite for downstream
consumption by Linguist, Historian, and Judge pillars.

The normalizer ensures:
  1. Consistent timestamp handling (UTC, ISO 8601)
  2. Ticker symbol canonicalization
  3. Source attribution and confidence scoring
  4. Deduplication via content hash
  5. Schema versioning for backward compatibility
"""

import sqlite3
import hashlib
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import threading


# ============================================================================
# Schema Definition
# ============================================================================

@dataclass
class SignalRecord:
    """
    Unified schema for all sentiment and price signals.
    
    Fields:
      ticker: Canonicalized stock symbol (e.g. "AAPL")
      timestamp: UTC datetime when signal was generated/observed
      source: Origin (e.g. "yfinance", "sec_edgar", "reddit", "hacker_news")
      signal_type: Category of signal (e.g. "price", "filing", "sentiment")
      value: Primary numeric or categorical value
      metadata: Source-specific extra fields (JSON string)
      content_hash: SHA256 of normalized content for deduplication
      confidence: Float [0, 1] indicating signal reliability
      version: Schema version (for migrations)
    """
    ticker: str
    timestamp: str  # ISO 8601 UTC
    source: str
    signal_type: str
    value: str  # Stored as string; parsing is caller's responsibility
    metadata: str  # JSON string
    content_hash: str
    confidence: float = 1.0
    version: int = 1
    id: Optional[int] = None  # Auto-increment PK, None until inserted


# ============================================================================
# Normalizer Class
# ============================================================================

class SignalNormalizer:
    """
    Multi-threaded normalizer that accepts heterogeneous scout outputs and
    persists unified SignalRecords to SQLite.
    """

    def __init__(self, db_path: str = "sentinel_signals.db"):
        """
        Initialize normalizer with SQLite backend.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self.lock = threading.RLock()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Return a thread-local database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Main signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    content_hash TEXT UNIQUE NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON signals (ticker, timestamp DESC)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON signals (source)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_hash
                ON signals (content_hash)
            """)
            
            # Deduplication log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dedup_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT UNIQUE NOT NULL,
                    duplicate_count INTEGER DEFAULT 1,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()

    def _compute_content_hash(self, ticker: str, source: str, 
                              signal_type: str, value: str) -> str:
        """
        Compute SHA256 hash of normalized content for deduplication.
        
        Args:
            ticker: Stock symbol.
            source: Data source.
            signal_type: Type of signal.
            value: Signal value.
            
        Returns:
            Hex-encoded SHA256 hash.
        """
        content = f"{ticker}|{source}|{signal_type}|{value}".lower()
        return hashlib.sha256(content.encode()).hexdigest()

    def _canonicalize_ticker(self, ticker: str) -> str:
        """
        Normalize ticker symbol.
        
        Args:
            ticker: Raw ticker string.
            
        Returns:
            Uppercase, whitespace-trimmed ticker.
        """
        return ticker.strip().upper()

    def _normalize_timestamp(self, ts: Any) -> str:
        """
        Convert timestamp to ISO 8601 UTC string.
        
        Args:
            ts: datetime, timestamp string, or numeric Unix timestamp.
            
        Returns:
            ISO 8601 UTC string.
        """
        if isinstance(ts, str):
            # Try to parse common formats
            for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", 
                        "%Y-%m-%d", "%d/%m/%Y"]:
                try:
                    dt = datetime.strptime(ts, fmt)
                    return dt.isoformat() + "Z"
                except ValueError:
                    continue
            # If no format matches, assume ISO and return as-is
            return ts if ts.endswith("Z") else ts + "Z"
        elif isinstance(ts, (int, float)):
            # Unix timestamp
            return datetime.utcfromtimestamp(ts).isoformat() + "Z"
        elif isinstance(ts, datetime):
            return ts.isoformat() + "Z"
        else:
            return datetime.utcnow().isoformat() + "Z"

    def normalize_price_signal(self, ticker: str, price: float, 
                              timestamp: Any, source: str = "yfinance",
                              confidence: float = 1.0) -> SignalRecord:
        """
        Normalize a price signal from yfinance or fallback sources.
        
        Args:
            ticker: Stock symbol.
            price: Current price.
            timestamp: When price was observed.
            source: Data source (default "yfinance").
            confidence: Signal confidence [0, 1].
            
        Returns:
            Normalized SignalRecord.
        """
        ticker = self._canonicalize_ticker(ticker)
        ts = self._normalize_timestamp(timestamp)
        value = str(round(price, 2))
        content_hash = self._compute_content_hash(ticker, source, "price", value)
        
        return SignalRecord(
            ticker=ticker,
            timestamp=ts,
            source=source,
            signal_type="price",
            value=value,
            metadata="{}",
            content_hash=content_hash,
            confidence=confidence,
            version=1
        )

    def normalize_news_signal(self, ticker: str, headline: str, url: str,
                             timestamp: Any, sentiment: Optional[str] = None,
                             source: str = "news_api",
                             confidence: float = 0.8) -> SignalRecord:
        """
        Normalize a news/headline signal.
        
        Args:
            ticker: Stock symbol.
            headline: Article headline/title.
            url: Source URL.
            timestamp: Publication date.
            sentiment: Optional sentiment label ("positive", "negative", "neutral
