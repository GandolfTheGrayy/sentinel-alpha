"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with typed collections
for market events, SEC filings, and sentiment signals. It provides a type-safe
client wrapper that the RAG pipeline uses to embed and retrieve historical context.

Part of sentinel/historian/ — the retrieval-augmented generation backbone.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


class ChromaDBClient:
    """Typed wrapper around ChromaDB collections for Sentinel's historian."""

    def __init__(self, db_path: str = "./data/chromadb") -> None:
        """Initialize ChromaDB client with persistent storage and collections.
        
        Args:
            db_path: Root directory for ChromaDB persistence; created if missing.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self._init_collections()

    def _init_collections(self) -> None:
        """Create or retrieve typed collections for market events, filings, and sentiment."""
        self.events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, earnings, M&A, regulatory changes"}
        )
        self.filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC 8-K, 10-Q, 10-K excerpts and summaries"}
        )
        self.sentiment_collection = self.client.get_or_create_collection(
            name="sentiment_signals",
            metadata={"description": "Reddit, HN, news sentiment snapshots per ticker/date"}
        )
        self.dev_health_collection = self.client.get_or_create_collection(
            name="dev_health_signals",
            metadata={"description": "GitHub activity, commit velocity, issue churn"}
        )

    def insert_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        text: str,
        date: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Insert a market event into the events collection.
        
        Args:
            event_id: Unique identifier for the event (e.g., "AAPL_earnings_2024-01-30").
            ticker: Stock ticker symbol.
            event_type: Type of event (e.g., "earnings", "acquisition", "regulatory").
            text: Full text/description of the event.
            date: ISO 8601 date string (YYYY-MM-DD).
            metadata: Optional extra metadata dict (impact_score, source, etc.).
        """
        meta = metadata or {}
        meta.update({"ticker": ticker, "event_type": event_type, "date": date})
        self.events_collection.add(
            ids=[event_id],
            documents=[text],
            metadatas=[meta],
        )

    def insert_filing(
        self,
        filing_id: str,
        ticker: str,
        filing_type: str,
        text: str,
        date: str,
        url: Optional[str] = None,
    ) -> None:
        """Insert a SEC filing excerpt into the filings collection.
        
        Args:
            filing_id: Unique filing identifier (e.g., "AAPL_8K_2024-01-15").
            ticker: Stock ticker symbol.
            filing_type: Filing type (e.g., "8-K", "10-Q", "10-K").
            text: Extracted text or summary from the filing.
            date: Filing date (YYYY-MM-DD).
            url: Optional URL to the SEC EDGAR page.
        """
        meta = {"ticker": ticker, "filing_type": filing_type, "date": date}
        if url:
            meta["url"] = url
        self.filings_collection.add(
            ids=[filing_id],
            documents=[text],
            metadatas=[meta],
        )

    def insert_sentiment(
        self,
        signal_id: str,
        ticker: str,
        source: str,
        text: str,
        date: str,
        score: float,
    ) -> None:
        """Insert a sentiment signal (Reddit, HN, news) into the sentiment collection.
        
        Args:
            signal_id: Unique signal identifier (e.g., "AAPL_reddit_2024-01-15_001").
            ticker: Stock ticker symbol.
            source: Source of sentiment (e.g., "reddit", "hackernews", "news").
            text: Raw text snippet from the source.
            date: Signal date (YYYY-MM-DD).
            score: Sentiment score, typically in [-1, 1] range.
        """
        meta = {"ticker": ticker, "source": source, "date": date, "score": score}
        self.sentiment_collection.add(
            ids=[signal_id],
            documents=[text],
            metadatas=[meta],
        )

    def insert_dev_health(
        self,
        signal_id: str,
        repo: str,
        metric: str,
        value: float,
        date: str,
    ) -> None:
        """Insert a developer health signal (GitHub activity) into the dev_health collection.
        
        Args:
            signal_id: Unique signal identifier (e.g., "openai/gpt-4_commits_2024-01-15").
            repo: Repository identifier (owner/name).
            metric: Metric type (e.g., "commit_velocity", "issue_churn", "pr_merge_rate").
            value: Numeric value of the metric.
            date: Signal date (YYYY-MM-DD).
        """
        meta = {"repo": repo, "metric": metric, "date": date, "value": value}
        self.dev_health_collection.add(
            ids=[signal_id],
            documents=[f"{repo} {metric}: {value}"],
            metadatas=[meta],
        )

    def query_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """Query market events collection by semantic similarity.
        
        Args:
            query_text: Semantic query string.
            ticker: Optional ticker filter.
            n_results: Number of results to return.
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances' keys.
        """
        where_filter = {"ticker": {"$eq": ticker}} if ticker else None
        return self.events_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
        )

    def query_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """Query SEC filings collection by semantic similarity.
        
        Args:
            query_text: Semantic query string.
            ticker: Optional ticker filter.
            filing_type: Optional filing type filter (e.g., "8-K").
            n_results: Number of results to return.
            
        Returns:
            Dict with 'ids', 'documents', 'metadatas', 'distances' keys.
        """
        where_filter = {}
        if ticker:
            where_filter["ticker"] = {"$eq": ticker}
        if filing_type:
            where_filter["filing_type"] = {"$eq": filing_type}
        
        where = where_filter if where_filter else None
        return self.filings_collection.query(
            query_texts=[query_text],
