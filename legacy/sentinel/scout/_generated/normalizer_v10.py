"""
Sentinel Scout Normalizer — unified signal schema & SQLite persistence.

This module accepts heterogeneous outputs from scout scrapers (live_prices, news,
sec_filings, reddit sentiment, etc.) and normalizes them into a canonical
SignalRecord dataclass. It manages SQLite storage, upserts, and schema migration
to ensure all downstream Linguist/Historian/Judge modules work with consistent,
typed data structures.

Role in Sentinel: Scout pillar backbone. Called by scout orchestrators after
each scraper run. Feeds normalized records to Historian (RAG indexing) and
Judge (prediction pipeline).
"""

import sqlite3
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Enumeration of all scrapers that feed into normalizer."""
    LIVE_PRICE = "live_price"
    NEWS_HEADLINE = "news_headline"
    SEC_FILING = "sec_filing"
    REDDIT_SENTIMENT = "reddit_sentiment"
    GITHUB_DEVELOPER = "github_developer"
    EARNINGS_CALENDAR = "earnings_calendar"
    ANALYST_CALL = "analyst_call"


class SignalSentiment(Enum):
    """Normalized sentiment polarity across all sources."""
    BEARISH = -1
    NEUTRAL = 0
    BULLISH = 1


@dataclasses.dataclass
class SignalRecord:
    """
    Canonical normalized signal record.
    
    All scout scrapers must produce dictionaries that can be coerced into
    this schema. Primary key is (ticker, source, event_id).
    """
    ticker: str
    source: SignalSource
    event_id: str
    timestamp: datetime
    sentiment: Optional[SignalSentiment]
    confidence: float
    raw_text: str
    metadata: Dict[str, Any]
    processed_at: datetime = dataclasses.field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize SignalRecord to JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "source": self.source.value,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "sentiment": self.sentiment.value if self.sentiment else None,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "metadata": json.dumps(self.metadata),
            "processed_at": self.processed_at.isoformat(),
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SignalRecord":
        """Deserialize from dict (reverse of to_dict)."""
        return SignalRecord(
            ticker=data["ticker"],
            source=SignalSource(data["source"]),
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            sentiment=SignalSentiment(data["sentiment"]) if data.get("sentiment") is not None else None,
            confidence=data["confidence"],
            raw_text=data["raw_text"],
            metadata=json.loads(data["metadata"]) if isinstance(data["metadata"], str) else data["metadata"],
            processed_at=datetime.fromisoformat(data["processed_at"]),
        )


class SignalNormalizer:
    """
    Main normalizer class: converts scraper outputs to SignalRecord,
    persists to SQLite, and provides query/upsert interface.
    """
    
    def __init__(self, db_path: str = "sentinel.db"):
        """Initialize normalizer with SQLite backend."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Create or migrate signal_records table if needed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_records (
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sentiment INTEGER,
                    confidence REAL NOT NULL,
                    raw_text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (ticker, source, event_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON signal_records(ticker, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON signal_records(source)
            """)
            conn.commit()
            logger.info(f"Schema initialized: {self.db_path}")
    
    def normalize_live_price(
        self,
        ticker: str,
        price: float,
        volume: int,
        timestamp: datetime,
    ) -> SignalRecord:
        """
        Normalize live price data from yfinance/stooq.
        
        No inherent sentiment, but encoded as neutral with high confidence.
        """
        return SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.LIVE_PRICE,
            event_id=f"{ticker}_{timestamp.timestamp()}",
            timestamp=timestamp,
            sentiment=SignalSentiment.NEUTRAL,
            confidence=0.95,
            raw_text=f"Price: ${price:.2f}, Volume: {volume}",
            metadata={
                "price": price,
                "volume": volume,
            },
        )
    
    def normalize_news_headline(
        self,
        ticker: str,
        headline: str,
        source: str,
        url: str,
        timestamp: datetime,
        sentiment: Optional[int] = None,
        confidence: float = 0.7,
    ) -> SignalRecord:
        """
        Normalize news headline with optional sentiment from Linguist.
        
        sentiment: -1 (bearish), 0 (neutral), 1 (bullish), None (unannotated).
        """
        return SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.NEWS_HEADLINE,
            event_id=f"{source}_{url.replace('/', '_')[:50]}",
            timestamp=timestamp,
            sentiment=SignalSentiment(sentiment) if sentiment is not None else None,
            confidence=confidence,
            raw_text=headline,
            metadata={
                "news_source": source,
                "url": url,
            },
        )
    
    def normalize_sec_filing(
        self,
        ticker: str,
        filing_type: str,
        accession_number: str,
        filing_date: datetime,
        raw_text: str,
        sentiment: Optional[int] = None,
        confidence: float = 0.8,
    ) -> SignalRecord:
        """
        Normalize SEC EDGAR filing (8-K, 10-Q, etc.).
        
        filing_type: e.g., '8-K', '10-Q', '10-K'.
        """
        return SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.SEC_FILING,
            event_id=accession_number,
            timestamp=filing_date,
            sentiment=SignalSentiment(sentiment) if sentiment is not None else None,
            confidence=confidence,
            raw_text=raw_text,
            metadata={
                "filing_type": filing_type,
                "accession_number": accession_number,
            },
        )
    
    def normalize_reddit_sentiment(
        self,
        ticker: str,
        post_id: str,
        title: str,
        body: str,
        timestamp: datetime,
        sentiment: int,
        confidence: float,
        subreddit: str = "stocks",
    ) -> SignalRecord:
        """Normalize Reddit post sentiment (from scout or Linguist)."""
        return SignalRecord(
            ticker=
