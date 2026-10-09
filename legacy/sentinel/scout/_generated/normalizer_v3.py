"""
Sentinel Scout Normalizer — Data integration layer that maps heterogeneous
scraper outputs (live prices, SEC filings, news, sentiment) into a unified
SignalRecord schema and persists to SQLite.

This module bridges all Scout data sources (live_prices, sec_filings, news,
Reddit sentiment, GitHub signals) into a canonical event stream. Each record
is timestamped, tagged with source and confidence, and stored in a queryable
SQLite table for downstream Linguist and Historian consumption.

Role in Sentinel:
  - Accepts raw dicts from live_prices.py, sec_filings.py, news.py, etc.
  - Validates and normalizes into SignalRecord dataclass
  - Assigns confidence scores based on source reliability
  - Persists to sentinel.db for RAG historization and Judge replay
  - Provides query APIs for Historian RAG pipeline
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMA & DATACLASS
# ============================================================================

@dataclass
class SignalRecord:
    """
    Canonical signal record schema for all Sentinel data sources.
    
    Attributes:
        ticker: Stock symbol (e.g., 'AAPL')
        timestamp: ISO 8601 UTC timestamp of signal origin
        source: Scraper origin ('live_price', 'sec_8k', 'sec_10q', 'news',
                                'reddit', 'github')
        signal_type: Category of signal ('price', 'regulatory', 'sentiment',
                                         'developer_health')
        headline: Brief headline or title (for news/filings)
        body: Raw text content (filing excerpt, article text, sentiment)
        metadata: JSON dict with source-specific fields (url, filing_id, etc.)
        confidence: Float [0.0, 1.0] based on source reliability
        raw_data: Entire original dict for audit trail
    """
    ticker: str
    timestamp: str
    source: str
    signal_type: str
    headline: Optional[str] = None
    body: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    confidence: float = 0.5
    raw_data: Optional[Dict[str, Any]] = None


# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_signals_db(db_path: str = "sentinel.db") -> None:
    """Initialize SQLite schema for signal records if not exists."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            headline TEXT,
            body TEXT,
            metadata TEXT,
            confidence REAL NOT NULL,
            raw_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, timestamp, source, signal_type)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
        ON signals(ticker, timestamp DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_source
        ON signals(source)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_signal_type
        ON signals(signal_type)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"Initialized signals database at {db_path}")


# ============================================================================
# NORMALIZATION: ROUTE BY SOURCE
# ============================================================================

def normalize_live_price(ticker: str, price_dict: Dict[str, Any]) -> SignalRecord:
    """Normalize yfinance/stooq live price into SignalRecord."""
    price_dict = price_dict or {}
    current_price = price_dict.get("current", 0.0)
    high = price_dict.get("high", None)
    low = price_dict.get("low", None)
    volume = price_dict.get("volume", None)
    
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return SignalRecord(
        ticker=ticker.upper(),
        timestamp=timestamp,
        source="live_price",
        signal_type="price",
        headline=f"{ticker} @ ${current_price:.2f}",
        body=f"Current: ${current_price:.2f} | High: ${high} | Low: ${low} | Vol: {volume}",
        metadata={
            "price": current_price,
            "high": high,
            "low": low,
            "volume": volume,
        },
        confidence=0.95,
        raw_data=price_dict,
    )


def normalize_sec_filing(ticker: str, filing_dict: Dict[str, Any]) -> SignalRecord:
    """Normalize SEC EDGAR filing (8-K, 10-Q) into SignalRecord."""
    filing_dict = filing_dict or {}
    filing_type = filing_dict.get("type", "UNKNOWN")
    filing_date = filing_dict.get("date", datetime.now(timezone.utc).isoformat())
    accession = filing_dict.get("accession_number", "")
    text = filing_dict.get("text", "")
    
    return SignalRecord(
        ticker=ticker.upper(),
        timestamp=filing_date,
        source=f"sec_{filing_type.lower()}",
        signal_type="regulatory",
        headline=f"SEC {filing_type} Filing",
        body=text[:500] if text else "",
        metadata={
            "filing_type": filing_type,
            "accession_number": accession,
            "full_text_length": len(text) if text else 0,
        },
        confidence=0.98,
        raw_data=filing_dict,
    )


def normalize_news(ticker: str, news_dict: Dict[str, Any]) -> SignalRecord:
    """Normalize news headline/article into SignalRecord."""
    news_dict = news_dict or {}
    title = news_dict.get("title", "Untitled")
    url = news_dict.get("url", "")
    published = news_dict.get("published", datetime.now(timezone.utc).isoformat())
    body = news_dict.get("text", "")
    source_name = news_dict.get("source", "unknown_news")
    
    return SignalRecord(
        ticker=ticker.upper(),
        timestamp=published,
        source="news",
        signal_type="sentiment",
        headline=title,
        body=body[:500] if body else "",
        metadata={
            "url": url,
            "source": source_name,
        },
        confidence=0.70,
        raw_data=news_dict,
    )


def normalize_reddit(ticker: str, reddit_dict: Dict[str, Any]) -> SignalRecord:
    """Normalize Reddit post/comment into SignalRecord."""
    reddit_dict = reddit_dict or {}
    title = reddit_dict.get("title", reddit_dict.get("body", "Reddit mention"))
    body = reddit_dict.get("body", "")
    post_url = reddit_dict.get("url", "")
    created = reddit_dict.get("created_utc", datetime.now(timezone.utc).isoformat())
    subreddit = reddit_dict.get("subreddit", "unknown")
    score = reddit_dict.get("score", 0)
    
    return SignalRecord(
        ticker=ticker.upper(),
        timestamp=created,
        source="reddit",
        signal_type="sentiment",
        headline=title[:100],
        body=body[:500] if body else "",
        metadata={
            "subreddit": subreddit,
            "score": score,
            "url": post_url,
        },
        confidence=0.50,
        raw_data=reddit_dict,
    )


def normalize_github(ticker: str, github_dict: Dict[str
