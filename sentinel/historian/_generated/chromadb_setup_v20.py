"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with collections for
market events, SEC filings, and sentiment signals. It provides a typed client
interface for embedding-based RAG queries used by the Historian pillar to
cross-reference sentiment signals with historical market context.

Role in Sentinel:
- Initializes persistent ChromaDB collections on startup
- Provides typed wrapper methods for upserting and querying vectors
- Integrates with sentinel/historian/rag_query.py for RAG synthesis
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings


class ChromaDBClient:
    """Typed wrapper around ChromaDB collections for Sentinel historian queries."""

    def __init__(self, db_path: str = "sentinel_chroma") -> None:
        """
        Initialize ChromaDB client and create/load collections.

        Args:
            db_path: Directory path for persistent ChromaDB storage.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)

        # Initialize or retrieve collections
        self._init_collections()

    def _init_collections(self) -> None:
        """Create or retrieve ChromaDB collections for market events and filings."""
        # Collection for SEC filings (10-K, 10-Q, 8-K)
        self.filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC filings with embedding vectors"},
        )

        # Collection for market events (earnings, guidance, regulatory actions)
        self.events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events and their outcomes"},
        )

        # Collection for sentiment signals (Reddit, HackerNews, news headlines)
        self.sentiment_collection = self.client.get_or_create_collection(
            name="sentiment_signals",
            metadata={"description": "Niche sentiment signals cross-indexed by ticker"},
        )

    def upsert_filing(
        self,
        doc_id: str,
        text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Upsert a SEC filing document with embedding into filings collection.

        Args:
            doc_id: Unique identifier for the filing.
            text: Full text of the filing.
            ticker: Stock ticker symbol.
            filing_type: Type of filing (8-K, 10-Q, 10-K, etc.).
            filing_date: ISO date string of filing date.
            metadata: Optional additional metadata dict.
        """
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
            }
        )
        self.filings_collection.upsert(
            ids=[doc_id], documents=[text], metadatas=[metadata]
        )

    def upsert_event(
        self,
        doc_id: str,
        description: str,
        ticker: str,
        event_type: str,
        event_date: str,
        outcome: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Upsert a market event (e.g., earnings miss, regulatory approval).

        Args:
            doc_id: Unique identifier for the event.
            description: Human-readable description of the event.
            ticker: Stock ticker symbol.
            event_type: Category (earnings, guidance, acquisition, etc.).
            event_date: ISO date string of event date.
            outcome: Price movement or market outcome following the event.
            metadata: Optional additional metadata dict.
        """
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "ticker": ticker,
                "event_type": event_type,
                "event_date": event_date,
            }
        )
        if outcome:
            metadata["outcome"] = outcome
        self.events_collection.upsert(
            ids=[doc_id], documents=[description], metadatas=[metadata]
        )

    def upsert_sentiment(
        self,
        doc_id: str,
        text: str,
        ticker: str,
        source: str,
        timestamp: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Upsert a sentiment signal (Reddit, HN, news headline, etc.).

        Args:
            doc_id: Unique identifier for the signal.
            text: Content text of the signal.
            ticker: Stock ticker symbol.
            source: Source identifier (reddit, hackernews, news, etc.).
            timestamp: ISO timestamp of signal creation.
            metadata: Optional additional metadata dict.
        """
        if metadata is None:
            metadata = {}
        metadata.update({"ticker": ticker, "source": source, "timestamp": timestamp})
        self.sentiment_collection.upsert(
            ids=[doc_id], documents=[text], metadatas=[metadata]
        )

    def query_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Semantic search for SEC filings by text and optional filters.

        Args:
            query_text: Query string to embed and search.
            ticker: Optional ticker filter.
            filing_type: Optional filing type filter (8-K, 10-Q, etc.).
            top_k: Number of results to return.

        Returns:
            ChromaDB query result dict with ids, documents, distances, metadatas.
        """
        where_filter = {}
        if ticker:
            where_filter["ticker"] = ticker
        if filing_type:
            where_filter["filing_type"] = filing_type

        result = self.filings_collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter if where_filter else None,
        )
        return result

    def query_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Semantic search for market events by description and optional filters.

        Args:
            query_text: Query string to embed and search.
            ticker: Optional ticker filter.
            event_type: Optional event type filter (earnings, guidance, etc.).
            top_k: Number of results to return.

        Returns:
            ChromaDB query result dict with ids, documents, distances, metadatas.
        """
        where_filter = {}
        if ticker:
            where_filter["ticker"] = ticker
        if event_type:
            where_filter["event_type"] = event_type

        result = self.events_collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter if where_filter else None,
        )
        return result

    def query_sentiment(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """
        Semantic search for sentiment signals by text and optional filters.

        Args:
            query_text: Query string to embed and search.
            ticker: Optional ticker filter.
            source: Optional source filter (reddit,
