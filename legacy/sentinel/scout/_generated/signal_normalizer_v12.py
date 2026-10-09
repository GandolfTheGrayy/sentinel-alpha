"""
Sentinel Scout Signal Normalizer — unified data pipeline for all scrapers.

This module ingests heterogeneous outputs from live_prices.py, news.py,
sec_filings.py, and future sentiment sources, normalizing them into a
canonical SignalRecord schema. Records are persisted to SQLite for:
  - historical lookup by Historian (RAG backbone)
  - drift detection by Linguist (tone shift analysis)
  - post-mortem correlation by Judge (prediction calibration)

Architecture:
  1. SignalRecord (dataclass) — immutable canonical form
  2. SignalNormalizer (class) — ingests raw scraper outputs, validates, stores
  3. SQLite schema — indexed by (ticker, signal_type, timestamp)
  4. Query API — fetch signals by ticker/date range for downstream consumers

Design principle: schema-first, source-agnostic. If a new scraper is added,
only register it here; downstream (Linguist, Judge) need not change.
"""

import dataclasses
import datetime
import json
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class SignalType(str, Enum):
    """Enumeration of all signal sources in Sentinel."""
    LIVE_PRICE = "live_price"
    NEWS_HEADLINE = "news_headline"
    SEC_8K = "sec_8k"
    SEC_10Q = "sec_10q"
    REDDIT_SENTIMENT = "reddit_sentiment"
    GITHUB_ACTIVITY = "github_activity"


@dataclasses.dataclass(frozen=True)
class SignalRecord:
    """
    Canonical signal record — immutable, schema-first.

    Attributes:
        ticker: stock symbol (e.g., "AAPL")
        signal_type: source type (SignalType enum)
        timestamp: UTC datetime when signal was generated/observed
        raw_data: dict of source-specific fields (e.g., headline text, price, volume)
        confidence: float [0.0, 1.0] — credibility score from source
        metadata: dict of optional normalized fields (e.g., sentiment_label, urgency)
    """
    ticker: str
    signal_type: SignalType
    timestamp: datetime.datetime
    raw_data: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate after frozen instantiation."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError(f"ticker must be non-empty string, got {self.ticker}")


class SignalNormalizer:
    """
    Unified ingestion and storage for all Sentinel signal sources.

    Responsibilities:
      - Accept raw outputs from Scout scrapers (live_prices, news, sec_filings, etc.)
      - Normalize to SignalRecord schema
      - Validate and deduplicate
      - Persist to SQLite with indexing
      - Query interface for Historian, Linguist, Judge
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize normalizer with SQLite backend.

        Args:
            db_path: path to SQLite database. Defaults to sentinel/data/signals.db
        """
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "data" / "signals.db")
        
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create SQLite schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_data TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(ticker, signal_type, timestamp, raw_data)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON signals(ticker, timestamp DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_type
                ON signals(signal_type)
            """)
            conn.commit()

    def ingest_live_price(
        self,
        ticker: str,
        price: float,
        volume: int,
        timestamp: datetime.datetime,
        confidence: float = 0.95,
    ) -> SignalRecord:
        """
        Normalize live price feed (from yfinance/stooq).

        Args:
            ticker: stock symbol
            price: current price
            volume: trading volume
            timestamp: observation time (UTC)
            confidence: source reliability [0, 1]

        Returns:
            SignalRecord persisted to SQLite
        """
        record = SignalRecord(
            ticker=ticker.upper(),
            signal_type=SignalType.LIVE_PRICE,
            timestamp=timestamp,
            raw_data={"price": price, "volume": volume},
            confidence=confidence,
            metadata={"price_usd": price, "volume_shares": volume},
        )
        self.store(record)
        return record

    def ingest_news_headline(
        self,
        ticker: str,
        headline: str,
        url: str,
        source: str,
        timestamp: datetime.datetime,
        confidence: float = 0.80,
    ) -> SignalRecord:
        """
        Normalize news headline (from news.py scraper).

        Args:
            ticker: stock symbol
            headline: headline text
            url: source URL
            source: news outlet (e.g., "Reuters", "Bloomberg")
            timestamp: publication time (UTC)
            confidence: source credibility [0, 1]

        Returns:
            SignalRecord persisted to SQLite
        """
        record = SignalRecord(
            ticker=ticker.upper(),
            signal_type=SignalType.NEWS_HEADLINE,
            timestamp=timestamp,
            raw_data={"headline": headline, "url": url, "source": source},
            confidence=confidence,
            metadata={"headline_char_count": len(headline), "source_tier": source},
        )
        self.store(record)
        return record

    def ingest_sec_8k(
        self,
        ticker: str,
        filing_date: datetime.datetime,
        item_type: str,
        text_excerpt: str,
        url: str,
        confidence: float = 0.98,
    ) -> SignalRecord:
        """
        Normalize SEC 8-K current report (material events).

        Args:
            ticker: stock symbol
            filing_date: date filed with SEC
            item_type: Item number (e.g., "Item 8.01", "Item 5.02")
            text_excerpt: relevant text snippet (first 500 chars)
            url: SEC EDGAR URL
            confidence: high; SEC filings are authoritative

        Returns:
            SignalRecord persisted to SQLite
        """
        record = SignalRecord(
            ticker=ticker.upper(),
            signal_type=SignalType.SEC_8K,
            timestamp=filing_date,
            raw_data={
                "item_type": item_type,
                "text_excerpt": text_excerpt,
                "url": url,
            },
            confidence=confidence,
            metadata={"filing_type": "8-K", "item": item_type},
        )
        self.store(record)
        return record

    def ingest_sec_10q(
        self,
        ticker: str,
        filing_date: datetime.datetime,
        quarter: str,
        revenue: Optional[float],
        net_income: Optional[float],
