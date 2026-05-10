"""
Sentinel Scout Normalizer — unified data pipeline for heterogeneous signals.

This module consumes raw outputs from Scout scrapers (live_prices, news, sec_filings, etc.)
and normalizes them into a canonical SignalRecord schema, persisted in SQLite.
The normalizer ensures downstream Linguist and Historian modules receive consistent,
validated data regardless of source variability.

Schema:
  - signal_id (UUID): unique identifier
  - ticker (str): stock symbol
  - source (str): scraper origin (live_price, news, sec_filing, reddit, github)
  - signal_type (str): category (price, headline, filing_event, sentiment, activity)
  - raw_data (JSON): original unmodified payload
  - normalized_text (str): canonical prose for LLM ingestion
  - timestamp (ISO8601): event time (or ingestion time if unknown)
  - ingestion_time (ISO8601): when Sentinel processed it
  - confidence (float): 0.0–1.0, source credibility + data freshness
  - metadata (JSON): source-specific extras (url, filing_type, upvotes, etc.)
"""

import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SignalRecord:
    """Canonical signal record for normalized Sentinel data."""
    signal_id: str
    ticker: str
    source: str
    signal_type: str
    raw_data: Dict[str, Any]
    normalized_text: str
    timestamp: str
    ingestion_time: str
    confidence: float
    metadata: Dict[str, Any]


class SignalNormalizer:
    """
    Consumes heterogeneous scraper outputs and normalizes them into SignalRecord schema.
    Manages SQLite persistence and schema validation.
    """

    def __init__(self, db_path: str = "sentinel.db") -> None:
        """Initialize normalizer and ensure SQLite schema exists."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create signals table if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                raw_data TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                ingestion_time TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_ticker (ticker),
                INDEX idx_source (source),
                INDEX idx_timestamp (timestamp)
            )
        """)
        conn.commit()
        conn.close()

    def _iso_now(self) -> str:
        """Return current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def normalize_live_price(
        self,
        ticker: str,
        price: float,
        source: str = "live_price",
        confidence: float = 0.95,
        timestamp: Optional[str] = None,
    ) -> SignalRecord:
        """
        Normalize a live price tick into SignalRecord.
        
        Args:
            ticker: stock symbol
            price: closing/current price
            source: data origin (e.g., "yfinance", "stooq")
            confidence: credibility score
            timestamp: event time (defaults to now)
        
        Returns:
            SignalRecord ready for persistence
        """
        ts = timestamp or self._iso_now()
        raw = {"ticker": ticker, "price": price, "source": source}
        normalized_text = f"{ticker} price update: ${price:.2f}"
        
        return SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker=ticker,
            source=source,
            signal_type="price",
            raw_data=raw,
            normalized_text=normalized_text,
            timestamp=ts,
            ingestion_time=self._iso_now(),
            confidence=confidence,
            metadata={"price_currency": "USD"},
        )

    def normalize_news_headline(
        self,
        ticker: str,
        headline: str,
        url: str,
        source: str = "news",
        confidence: float = 0.85,
        timestamp: Optional[str] = None,
    ) -> SignalRecord:
        """
        Normalize a news headline into SignalRecord.
        
        Args:
            ticker: affected stock symbol
            headline: article title/summary
            url: source URL
            source: scraper origin (e.g., "bloomberg", "reuters")
            confidence: relevance credibility
            timestamp: publication time (defaults to now)
        
        Returns:
            SignalRecord ready for persistence
        """
        ts = timestamp or self._iso_now()
        raw = {"ticker": ticker, "headline": headline, "url": url, "source": source}
        normalized_text = f"{ticker} news: {headline}"
        
        return SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker=ticker,
            source=source,
            signal_type="headline",
            raw_data=raw,
            normalized_text=normalized_text,
            timestamp=ts,
            ingestion_time=self._iso_now(),
            confidence=confidence,
            metadata={"url": url},
        )

    def normalize_sec_filing(
        self,
        ticker: str,
        filing_type: str,
        filing_date: str,
        accession_number: str,
        summary: str,
        url: str,
        confidence: float = 0.90,
    ) -> SignalRecord:
        """
        Normalize an SEC EDGAR filing into SignalRecord.
        
        Args:
            ticker: issuer symbol
            filing_type: "8-K", "10-Q", "10-K", etc.
            filing_date: YYYY-MM-DD
            accession_number: SEC accession ID
            summary: extracted key text or form section
            url: EDGAR URL
            confidence: regulatory authenticity score
        
        Returns:
            SignalRecord ready for persistence
        """
        raw = {
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "accession_number": accession_number,
            "summary": summary,
            "url": url,
        }
        normalized_text = f"{ticker} SEC {filing_type} ({filing_date}): {summary[:200]}"
        
        return SignalRecord(
            signal_id=str(uuid.uuid4()),
            ticker=ticker,
            source="sec_filing",
            signal_type="filing_event",
            raw_data=raw,
            normalized_text=normalized_text,
            timestamp=filing_date,
            ingestion_time=self._iso_now(),
            confidence=confidence,
            metadata={
                "filing_type": filing_type,
                "accession_number": accession_number,
                "url": url,
            },
        )

    def normalize_reddit_post(
        self,
        ticker: str,
        title: str,
        body: str,
        upvotes: int,
        subreddit: str,
        created_utc: int,
        confidence: float = 0.65,
    ) -> SignalRecord:
        """
        Normalize a Reddit post into SignalRecord.
        
        Args:
            ticker: mentioned stock symbol
            title: post title
            body: post content
            upvotes: score
            subreddit: source community
            created_utc: Unix timestamp
            confidence: retail-sentiment reliability (lower than institutional)
        
        Returns:
            SignalRecord ready for persistence
        """
        ts = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()
        raw = {
            "ticker": ticker,
            "title": title,
            "body": body,
            "upvotes": upvotes,
