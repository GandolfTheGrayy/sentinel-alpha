"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up a persistent ChromaDB instance with typed collections
for market events, SEC filings, and sentiment signals. It provides a single
typed wrapper (ChromaClient) that all Historian and Judge modules use to
query historical context and embeddings.

Part of the Sentinel Sentiment Engine's historian pillar — enables RAG
lookup of similar past events, regulatory whispers, and earnings-driven
sentiment patterns.
"""

import os
import logging
from typing import Optional, TypedDict
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingResult(TypedDict):
    """Single embedding result from a ChromaDB query."""
    id: str
    document: str
    metadata: dict
    distance: float


class ChromaClient:
    """Typed wrapper around ChromaDB collections for Sentinel historian."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """
        Initialize ChromaDB client with persistent storage and collections.
        
        Args:
            persist_dir: Path to persistent storage; defaults to ./sentinel_chromadb
        """
        if persist_dir is None:
            persist_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "chromadb_data"
            )
        
        os.makedirs(persist_dir, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )
        
        self.client = chromadb.Client(settings)
        self._ensure_collections()
        logger.info(f"ChromaDB initialized at {persist_dir}")

    def _ensure_collections(self) -> None:
        """Create or retrieve standard collections if they don't exist."""
        try:
            self.market_events = self.client.get_or_create_collection(
                name="market_events",
                metadata={"description": "Historical market events, earnings, crashes, rallies"},
            )
            logger.debug("Loaded/created 'market_events' collection")
        except Exception as e:
            logger.error(f"Failed to ensure market_events collection: {e}")
            raise

        try:
            self.sec_filings = self.client.get_or_create_collection(
                name="sec_filings",
                metadata={"description": "SEC 8-K, 10-Q, 10-K filings with extracted risk/opportunity signals"},
            )
            logger.debug("Loaded/created 'sec_filings' collection")
        except Exception as e:
            logger.error(f"Failed to ensure sec_filings collection: {e}")
            raise

        try:
            self.sentiment_signals = self.client.get_or_create_collection(
                name="sentiment_signals",
                metadata={"description": "Reddit, HN, social media sentiment patterns linked to tickers"},
            )
            logger.debug("Loaded/created 'sentiment_signals' collection")
        except Exception as e:
            logger.error(f"Failed to ensure sentiment_signals collection: {e}")
            raise

    def add_market_event(
        self,
        event_id: str,
        text: str,
        ticker: str,
        event_type: str,
        date: str,
    ) -> None:
        """
        Add a historical market event (earnings, crash, FDA approval, etc.).
        
        Args:
            event_id: Unique identifier (e.g., "AAPL_earnings_2023-01-31")
            text: Full event description for embedding
            ticker: Stock ticker symbol
            event_type: Category (earnings, crash, upgrade, recall, etc.)
            date: ISO date string (YYYY-MM-DD)
        """
        self.market_events.add(
            ids=[event_id],
            documents=[text],
            metadatas=[{
                "ticker": ticker,
                "event_type": event_type,
                "date": date,
            }],
        )
        logger.debug(f"Added market_event: {event_id}")

    def add_sec_filing(
        self,
        filing_id: str,
        text: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        url: str = "",
    ) -> None:
        """
        Add a parsed SEC filing for historical context lookup.
        
        Args:
            filing_id: Unique identifier (e.g., "AAPL_10-Q_2023-Q1")
            text: Extracted risk/opportunity sections from filing
            ticker: Stock ticker symbol
            form_type: SEC form type (8-K, 10-Q, 10-K, etc.)
            filing_date: ISO date string
            url: Optional link to SEC EDGAR
        """
        self.sec_filings.add(
            ids=[filing_id],
            documents=[text],
            metadatas=[{
                "ticker": ticker,
                "form_type": form_type,
                "filing_date": filing_date,
                "url": url,
            }],
        )
        logger.debug(f"Added sec_filing: {filing_id}")

    def add_sentiment_signal(
        self,
        signal_id: str,
        text: str,
        ticker: str,
        source: str,
        signal_date: str,
    ) -> None:
        """
        Add a sentiment signal (Reddit thread, HN comment, social media thread).
        
        Args:
            signal_id: Unique identifier (e.g., "TSLA_reddit_2024-01-15_xyz123")
            text: Full text of the sentiment signal
            ticker: Stock ticker symbol
            source: Origin (reddit, hackernews, twitter, stocktwits, etc.)
            signal_date: ISO date string
        """
        self.sentiment_signals.add(
            ids=[signal_id],
            documents=[text],
            metadatas=[{
                "ticker": ticker,
                "source": source,
                "signal_date": signal_date,
            }],
        )
        logger.debug(f"Added sentiment_signal: {signal_id}")

    def query_market_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> list[EmbeddingResult]:
        """
        Semantic search for similar historical market events.
        
        Args:
            query_text: Natural language query (e.g., "iPhone shortage impact")
            ticker: Optional filter by ticker
            n_results: Number of results to return
        
        Returns:
            List of EmbeddingResult dicts with id, document, metadata, distance
        """
        where = {"ticker": ticker} if ticker else None
        result = self.market_events.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )
        
        return self._format_query_results(result)

    def query_sec_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        form_type: Optional[str] = None,
        n_results: int = 5,
    ) -> list[EmbeddingResult]:
        """
        Semantic search for relevant SEC filings.
        
        Args:
            query_text: Natural language query (e.g., "supply chain risk")
            ticker: Optional filter by ticker
            form_type: Optional filter by form type (8-K, 10-Q, etc.)
            n_results: Number of results to return
        
        Returns:
            List of EmbeddingResult dicts
        """
        where = {}
        if ticker:
            where["ticker"] = ticker
        if form_type:
            where["form_type"] = form_type
        
        where_clause = where if where else None
        result = self.sec_filings.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_clause,
        )
        
        return self._format_query_results(result)
