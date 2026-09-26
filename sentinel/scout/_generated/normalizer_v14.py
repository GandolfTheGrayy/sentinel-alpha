"""
Sentinel Scout Normalizer — Unified Signal Record Schema & Storage

This module standardizes heterogeneous outputs from Scout scrapers (live prices,
news, SEC filings, Reddit sentiment, GitHub signals) into a single SignalRecord
schema persisted in SQLite. The normalizer acts as the central hub converting
raw scraper outputs into queryable, timestamped signal tuples that feed the
Linguist and Historian pillars.

Role in Sentinel:
  - Receives dicts/objects from live_prices.py, news.py, sec_filings.py, etc.
  - Maps each to SignalRecord (ticker, signal_type, value, source, timestamp)
  - Stores in SQLite for historical replay and RAG corpus building
  - Provides query interface for Historian RAG and Judge post-mortem
"""

import sqlite3
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Signal type enumeration — covers all Scout data sources
SignalType = Literal[
    "price_close",
    "price_volume",
    "news_headline",
    "sentiment_score",
    "sec_filing_8k",
    "sec_filing_10q",
    "reddit_post",
    "github_commit",
    "regulatory_alert",
]


class SignalRecord:
    """Immutable normalized signal record."""

    def __init__(
        self,
        ticker: str,
        signal_type: SignalType,
        value: float | str | Dict[str, Any],
        source: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a normalized signal record.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            signal_type: Category of signal (e.g., "price_close", "news_headline")
            value: Numeric or string payload (price, sentiment score, headline text)
            source: Origin scraper or data provider (e.g., "yfinance", "reddit")
            timestamp: UTC datetime when signal was observed
            metadata: Optional dict for extra context (URL, filing ID, author, etc.)
        """
        self.ticker = ticker.upper()
        self.signal_type = signal_type
        self.value = value
        self.source = source
        self.timestamp = timestamp if timestamp.tzinfo else timestamp.replace(
            tzinfo=timezone.utc
        )
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize record to dictionary for storage."""
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type,
            "value": self.value,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "SignalRecord":
        """Deserialize record from SQLite row."""
        return cls(
            ticker=row["ticker"],
            signal_type=row["signal_type"],
            value=row["value"],
            source=row["source"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def __repr__(self) -> str:
        return (
            f"SignalRecord(ticker={self.ticker}, type={self.signal_type}, "
            f"value={self.value}, source={self.source}, ts={self.timestamp.isoformat()})"
        )


class SignalNormalizer:
    """Normalize scraper outputs and persist to SQLite."""

    def __init__(self, db_path: str | Path = "sentinel/data/signals.db"):
        """
        Initialize normalizer with SQLite backend.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create signals table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, signal_type, source, timestamp)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker_ts ON signals(ticker, timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_type_ts ON signals(signal_type, timestamp)"
            )
            conn.commit()

    def ingest_price_record(
        self, ticker: str, close: float, volume: int, timestamp: datetime
    ) -> SignalRecord:
        """
        Normalize yfinance/stooq price record.

        Args:
            ticker: Stock ticker
            close: Closing price
            volume: Trading volume
            timestamp: Price observation time

        Returns:
            Stored SignalRecord for price_close
        """
        record = SignalRecord(
            ticker=ticker,
            signal_type="price_close",
            value=close,
            source="yfinance",
            timestamp=timestamp,
            metadata={"volume": volume},
        )
        self.store(record)

        volume_record = SignalRecord(
            ticker=ticker,
            signal_type="price_volume",
            value=volume,
            source="yfinance",
            timestamp=timestamp,
        )
        self.store(volume_record)

        return record

    def ingest_news_record(
        self, ticker: str, headline: str, url: str, timestamp: datetime
    ) -> SignalRecord:
        """
        Normalize news scraper headline record.

        Args:
            ticker: Stock ticker
            headline: News headline text
            url: Source URL
            timestamp: Publication time

        Returns:
            Stored SignalRecord for news_headline
        """
        record = SignalRecord(
            ticker=ticker,
            signal_type="news_headline",
            value=headline,
            source="news",
            timestamp=timestamp,
            metadata={"url": url},
        )
        self.store(record)
        return record

    def ingest_sec_filing_record(
        self,
        ticker: str,
        filing_type: Literal["8-K", "10-Q", "10-K"],
        content: str,
        filing_url: str,
        timestamp: datetime,
    ) -> SignalRecord:
        """
        Normalize SEC EDGAR filing record.

        Args:
            ticker: Company ticker
            filing_type: SEC form type (8-K, 10-Q, 10-K)
            content: Filing body text or summary
            filing_url: EDGAR link
            timestamp: Filing date

        Returns:
            Stored SignalRecord for sec_filing_*
        """
        signal_type_map = {"8-K": "sec_filing_8k", "10-Q": "sec_filing_10q", "10-K": "sec_filing_10q"}
        record = SignalRecord(
            ticker=ticker,
            signal_type=signal_type_map.get(filing_type, "sec_filing_8k"),
            value=content[:500],  # Truncate to first 500 chars
            source="sec_edgar",
            timestamp=timestamp,
            metadata={"filing_type": filing_type, "full_url": filing_url, "content_length": len(content)},
        )
        self.store(record)
        return record

    def ingest_sentiment_record(
        self,
        ticker: str,
        sentiment_score: float,
        source: str,
        timestamp: datetime,
        text_sample: Optional[str] = None,
    ) ->
