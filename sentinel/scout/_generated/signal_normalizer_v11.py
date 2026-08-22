"""
Sentinel Scout Signal Normalizer — unified schema ingestion layer.

Maps heterogeneous outputs from live_prices, news, sec_filings scrapers
into a canonical SignalRecord stored in SQLite. Ensures all upstream
collectors feed a consistent schema for downstream Linguist and Historian
analysis. Handles type coercion, timestamp normalization, and deduplication.
"""

import sqlite3
import json
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SignalRecord:
    """Canonical schema for all market signals across Sentinel."""
    
    ticker: str
    signal_type: str  # "price", "news", "sec_filing", "reddit", "github"
    source: str  # "yfinance", "newsapi", "sec_edgar", "praw", "github_api"
    timestamp: str  # ISO 8601
    value: float  # numeric signal: price, sentiment score, event weight
    text_body: Optional[str]  # headline, filing excerpt, post body
    metadata: str  # JSON string: {urls, event_codes, sentiment_label, etc}
    confidence: float  # [0.0, 1.0] — scraper's confidence in this signal
    processed: bool  # True if linguist/historian has consumed it


def init_signal_db(db_path: str) -> None:
    """Initialize SQLite schema for SignalRecord storage."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            source TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL,
            text_body TEXT,
            metadata TEXT NOT NULL,
            confidence REAL NOT NULL,
            processed BOOLEAN DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, signal_type, source, timestamp, value)
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticker_time 
        ON signals(ticker, timestamp DESC)
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed 
        ON signals(processed, created_at DESC)
    """)
    conn.commit()
    conn.close()


def normalize_price_signal(
    ticker: str,
    price: float,
    timestamp: datetime,
    source: str = "yfinance",
    metadata: Optional[dict] = None
) -> SignalRecord:
    """Normalize live price feed into SignalRecord."""
    meta = metadata or {}
    meta.setdefault("price_source", source)
    
    return SignalRecord(
        ticker=ticker.upper(),
        signal_type="price",
        source=source,
        timestamp=timestamp.isoformat(),
        value=float(price),
        text_body=None,
        metadata=json.dumps(meta),
        confidence=0.95,
        processed=False
    )


def normalize_news_signal(
    ticker: str,
    headline: str,
    url: str,
    timestamp: datetime,
    sentiment_label: Optional[str] = None,
    sentiment_score: float = 0.5,
    source: str = "newsapi"
) -> SignalRecord:
    """Normalize news headline into SignalRecord."""
    meta = {
        "url": url,
        "sentiment_label": sentiment_label or "neutral",
        "raw_sentiment": sentiment_score
    }
    
    return SignalRecord(
        ticker=ticker.upper(),
        signal_type="news",
        source=source,
        timestamp=timestamp.isoformat(),
        value=float(sentiment_score),
        text_body=headline[:500],
        metadata=json.dumps(meta),
        confidence=0.80,
        processed=False
    )


def normalize_sec_filing_signal(
    ticker: str,
    filing_type: str,
    accession_number: str,
    filing_date: datetime,
    excerpt: Optional[str] = None,
    event_codes: Optional[list] = None,
    source: str = "sec_edgar"
) -> SignalRecord:
    """Normalize SEC EDGAR 8-K/10-Q filing into SignalRecord."""
    event_codes = event_codes or []
    
    meta = {
        "filing_type": filing_type,
        "accession_number": accession_number,
        "event_codes": event_codes,
        "url": f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={ticker}&accession_number={accession_number}"
    }
    
    event_weight = len(event_codes) * 0.25 + 0.5
    event_weight = min(event_weight, 1.0)
    
    return SignalRecord(
        ticker=ticker.upper(),
        signal_type="sec_filing",
        source=source,
        timestamp=filing_date.isoformat(),
        value=float(event_weight),
        text_body=excerpt[:1000] if excerpt else None,
        metadata=json.dumps(meta),
        confidence=0.99,
        processed=False
    )


def normalize_reddit_signal(
    ticker: str,
    post_title: str,
    post_body: Optional[str],
    post_url: str,
    timestamp: datetime,
    upvote_ratio: float = 0.5,
    num_comments: int = 0,
    source: str = "praw"
) -> SignalRecord:
    """Normalize Reddit post sentiment into SignalRecord."""
    combined_text = post_title
    if post_body:
        combined_text += "\n" + post_body[:300]
    
    engagement_signal = min(upvote_ratio * (1.0 + 0.001 * num_comments), 1.0)
    
    meta = {
        "url": post_url,
        "upvote_ratio": upvote_ratio,
        "num_comments": num_comments,
        "platform": "reddit"
    }
    
    return SignalRecord(
        ticker=ticker.upper(),
        signal_type="reddit",
        source=source,
        timestamp=timestamp.isoformat(),
        value=float(engagement_signal),
        text_body=combined_text[:500],
        metadata=json.dumps(meta),
        confidence=0.70,
        processed=False
    )


def normalize_github_signal(
    ticker: str,
    repo_name: str,
    metric_type: str,
    metric_value: float,
    timestamp: datetime,
    metadata: Optional[dict] = None,
    source: str = "github_api"
) -> SignalRecord:
    """Normalize GitHub developer health metric into SignalRecord."""
    meta = metadata or {}
    meta.update({
        "repo_name": repo_name,
        "metric_type": metric_type
    })
    
    normalized_value = min(max(metric_value / 100.0, 0.0), 1.0)
    
    return SignalRecord(
        ticker=ticker.upper(),
        signal_type="github",
        source=source,
        timestamp=timestamp.isoformat(),
        value=float(normalized_value),
        text_body=f"{metric_type}: {metric_value}",
        metadata=json.dumps(meta),
        confidence=0.75,
        processed=False
    )


def insert_signal(db_path: str, record: SignalRecord) -> bool:
    """Insert normalized SignalRecord into SQLite, skip on duplicate."""
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO signals 
            (ticker, signal_type, source, timestamp, value, text_body, metadata, confidence, processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.ticker,
            record.signal_type,
            record.source,
            record.timestamp,
            record.value,
