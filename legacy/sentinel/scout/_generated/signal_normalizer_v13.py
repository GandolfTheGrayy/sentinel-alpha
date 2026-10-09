"""
Sentinel Scout — Signal Normalizer

Unifies outputs from all scrapers (live_prices, news, sec_filings, reddit, github)
into a canonical SignalRecord schema persisted in SQLite. Handles schema migrations,
deduplication, and confidence weighting across heterogeneous data sources.

Role in Sentinel:
  - Consumes raw outputs from scout/* scrapers
  - Normalizes into SignalRecord (ticker, source, signal_type, value, confidence, timestamp)
  - Stores in `signals.db` for downstream RAG and prediction pipelines
  - Provides query interface for historian/* and judge/* modules
"""

import sqlite3
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import os


class SignalSource(Enum):
    """Enumeration of data sources feeding Sentinel."""
    LIVE_PRICE = "live_price"
    NEWS_HEADLINE = "news_headline"
    SEC_FILING = "sec_filing"
    REDDIT = "reddit"
    GITHUB = "github"
    MANUAL = "manual"


class SignalType(Enum):
    """Enumeration of normalized signal types."""
    PRICE_CHANGE = "price_change"
    VOLUME_SPIKE = "volume_spike"
    SENTIMENT = "sentiment"
    REGULATORY = "regulatory"
    DEVELOPER_ACTIVITY = "developer_activity"
    EARNINGS_SURPRISE = "earnings_surprise"
    INSIDER_TRADE = "insider_trade"
    ANALYST_NOTE = "analyst_note"


@dataclass
class SignalRecord:
    """
    Canonical normalized record across all data sources.
    
    Attributes:
        ticker: Stock symbol (e.g., 'AAPL')
        source: SignalSource enum
        signal_type: SignalType enum
        value: Normalized numeric or categorical value
        confidence: Float [0.0, 1.0] indicating data quality/certainty
        timestamp: ISO 8601 datetime when signal was generated
        raw_data: JSON blob of unstructured metadata
        record_id: Unique identifier for deduplication (auto-assigned)
    """
    ticker: str
    source: str
    signal_type: str
    value: Any
    confidence: float
    timestamp: str
    raw_data: Optional[str] = None
    record_id: Optional[str] = None


class SignalNormalizer:
    """Normalizer responsible for schema enforcement, dedup, and persistence."""
    
    def __init__(self, db_path: str = "signals.db") -> None:
        """
        Initialize normalizer with SQLite backend.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create or migrate SignalRecord table if needed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
        )
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            cursor.execute("""
                CREATE TABLE signals (
                    record_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    source TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_data TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX idx_ticker_timestamp ON signals(ticker, timestamp DESC)"
            )
            cursor.execute(
                "CREATE INDEX idx_source ON signals(source)"
            )
            conn.commit()
        
        conn.close()
    
    def normalize_live_price(
        self,
        ticker: str,
        current_price: float,
        previous_price: float,
        volume: int,
        timestamp: str
    ) -> SignalRecord:
        """
        Normalize live price feed into SignalRecord.
        
        Args:
            ticker: Stock symbol
            current_price: Current trading price
            previous_price: Previous closing price
            volume: Trading volume
            timestamp: ISO 8601 datetime
        
        Returns:
            SignalRecord for price change and volume
        """
        price_change_pct = ((current_price - previous_price) / previous_price) * 100
        
        # High volume spikes warrant higher confidence
        confidence = 0.9 if abs(price_change_pct) > 2 else 0.7
        
        record = SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.LIVE_PRICE.value,
            signal_type=SignalType.PRICE_CHANGE.value,
            value=round(price_change_pct, 2),
            confidence=confidence,
            timestamp=timestamp,
            raw_data=json.dumps({
                "current_price": current_price,
                "previous_price": previous_price,
                "volume": volume
            })
        )
        return record
    
    def normalize_news_headline(
        self,
        ticker: str,
        headline: str,
        source_url: str,
        sentiment_score: float,
        timestamp: str
    ) -> SignalRecord:
        """
        Normalize news headline with embedded sentiment into SignalRecord.
        
        Args:
            ticker: Stock symbol
            headline: Article headline text
            source_url: Origin URL
            sentiment_score: Float in [-1, 1], negative=bearish, positive=bullish
            timestamp: ISO 8601 datetime
        
        Returns:
            SignalRecord for news sentiment
        """
        # Clamp sentiment to [-1, 1]
        clamped_sentiment = max(-1.0, min(1.0, sentiment_score))
        # Map to confidence: extreme scores are more confident
        confidence = 0.6 + (abs(clamped_sentiment) * 0.3)
        
        record = SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.NEWS_HEADLINE.value,
            signal_type=SignalType.SENTIMENT.value,
            value=round(clamped_sentiment, 2),
            confidence=confidence,
            timestamp=timestamp,
            raw_data=json.dumps({
                "headline": headline,
                "source_url": source_url
            })
        )
        return record
    
    def normalize_sec_filing(
        self,
        ticker: str,
        filing_type: str,
        key_findings: Dict[str, Any],
        regulatory_risk: float,
        timestamp: str
    ) -> SignalRecord:
        """
        Normalize SEC EDGAR filing into SignalRecord.
        
        Args:
            ticker: Stock symbol
            filing_type: '8-K', '10-Q', '10-K', etc.
            key_findings: Dict of extracted facts (e.g., revenue, warnings)
            regulatory_risk: Float [0, 1] risk assessment
            timestamp: ISO 8601 datetime
        
        Returns:
            SignalRecord for regulatory signal
        """
        confidence = 0.85 if filing_type in ["8-K", "10-Q"] else 0.75
        
        record = SignalRecord(
            ticker=ticker.upper(),
            source=SignalSource.SEC_FILING.value,
            signal_type=SignalType.REGULATORY.value,
            value=regulatory_risk,
            confidence=confidence,
            timestamp=timestamp,
            raw_data=json.dumps({
                "filing_type": filing_type,
                "key_findings": key_findings
            })
        )
        return record
    
    def normalize_reddit_sentiment(
        self,
        ticker: str,
        subreddit: str,
        mention_count: int,
        sentiment_avg: float,
        timestamp: str
    ) -> SignalRecord:
        """
        Normalize Reddit mention stream into SignalRecord.
