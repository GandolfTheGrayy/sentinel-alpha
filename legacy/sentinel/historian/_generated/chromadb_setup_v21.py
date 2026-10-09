"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up a local ChromaDB instance with collections for market events
and SEC filings, enabling RAG queries across historical sentiment and regulatory data.
Provides a typed ClientWrapper for safe, schema-aware vector operations.
"""

import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings


class ClientWrapper:
    """Typed wrapper around ChromaDB client for Sentinel RAG operations."""

    def __init__(self, db_path: str) -> None:
        """Initialize ChromaDB client and collections.
        
        Args:
            db_path: Path to persistent ChromaDB directory.
        """
        self.db_path = db_path
        self._ensure_db_dir()
        
        # Configure ChromaDB for persistent storage
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        
        # Initialize or get collections
        self.market_events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, earnings, regulatory filings"},
        )
        self.sec_filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC 8-K, 10-Q, 10-K document embeddings and metadata"},
        )
        self.news_sentiment_collection = self.client.get_or_create_collection(
            name="news_sentiment",
            metadata={"description": "News headlines and sentiment scores indexed by ticker and date"},
        )

    def _ensure_db_dir(self) -> None:
        """Create ChromaDB directory if it does not exist."""
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

    def add_market_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        date: str,
        text: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a market event to the vector database.
        
        Args:
            event_id: Unique event identifier.
            ticker: Stock ticker symbol.
            event_type: Type of event (e.g., "earnings", "acquisition", "regulatory_action").
            date: ISO 8601 date string.
            text: Event description or summary.
            embedding: Optional pre-computed embedding vector.
            metadata: Additional metadata dict.
        """
        doc_metadata = metadata or {}
        doc_metadata.update({"ticker": ticker, "event_type": event_type, "date": date})
        
        self.market_events_collection.add(
            ids=[event_id],
            documents=[text],
            embeddings=[embedding] if embedding else None,
            metadatas=[doc_metadata],
        )

    def add_sec_filing(
        self,
        filing_id: str,
        ticker: str,
        form_type: str,
        date: str,
        text: str,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a SEC filing to the vector database.
        
        Args:
            filing_id: Unique filing identifier (e.g., CIK-accession).
            ticker: Stock ticker symbol.
            form_type: SEC form type (e.g., "8-K", "10-Q", "10-K").
            date: ISO 8601 filing date.
            text: Filing text or summary.
            embedding: Optional pre-computed embedding vector.
            metadata: Additional metadata dict.
        """
        doc_metadata = metadata or {}
        doc_metadata.update({"ticker": ticker, "form_type": form_type, "date": date})
        
        self.sec_filings_collection.add(
            ids=[filing_id],
            documents=[text],
            embeddings=[embedding] if embedding else None,
            metadatas=[doc_metadata],
        )

    def add_news_sentiment(
        self,
        news_id: str,
        ticker: str,
        date: str,
        headline: str,
        sentiment_score: float,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Add a news headline with sentiment to the vector database.
        
        Args:
            news_id: Unique news identifier.
            ticker: Stock ticker symbol.
            date: ISO 8601 date of article.
            headline: News headline or summary text.
            sentiment_score: Sentiment score (typically -1.0 to 1.0).
            embedding: Optional pre-computed embedding vector.
            metadata: Additional metadata dict.
        """
        doc_metadata = metadata or {}
        doc_metadata.update({
            "ticker": ticker,
            "date": date,
            "sentiment_score": sentiment_score,
        })
        
        self.news_sentiment_collection.add(
            ids=[news_id],
            documents=[headline],
            embeddings=[embedding] if embedding else None,
            metadatas=[doc_metadata],
        )

    def query_market_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve market events by semantic similarity.
        
        Args:
            query_text: Natural language query.
            ticker: Optional ticker filter.
            event_type: Optional event type filter.
            limit: Maximum number of results.
        
        Returns:
            Dict with ids, documents, distances, metadatas keys.
        """
        where_filter = None
        if ticker or event_type:
            where_filter = {}
            if ticker:
                where_filter["ticker"] = ticker
            if event_type:
                where_filter["event_type"] = event_type
        
        return self.market_events_collection.query(
            query_texts=[query_text],
            where=where_filter,
            n_results=limit,
        )

    def query_sec_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        form_type: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve SEC filings by semantic similarity.
        
        Args:
            query_text: Natural language query.
            ticker: Optional ticker filter.
            form_type: Optional form type filter (e.g., "8-K").
            limit: Maximum number of results.
        
        Returns:
            Dict with ids, documents, distances, metadatas keys.
        """
        where_filter = None
        if ticker or form_type:
            where_filter = {}
            if ticker:
                where_filter["ticker"] = ticker
            if form_type:
                where_filter["form_type"] = form_type
        
        return self.sec_filings_collection.query(
            query_texts=[query_text],
            where=where_filter,
            n_results=limit,
        )

    def query_news_sentiment(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Retrieve news articles by semantic similarity.
        
        Args:
            query_text: Natural language query.
            ticker: Optional ticker filter.
            limit: Maximum number of results.
        
        Returns:
            Dict with ids, documents, distances, metadatas keys.
        """
        where_filter = {"ticker": ticker} if ticker else None
        
        return self.news_sentiment_collection.query(
            query_texts=[query_text],
            where=where_filter,
            n_results=limit,
        )

    def
