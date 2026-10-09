"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up the local ChromaDB instance with collections for market events
and SEC filings, providing a type-safe interface for embedding storage and retrieval.
Integrates with the Historian pillar's RAG pipeline for historical event lookup
and confidence score weighting.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict, Any

import chromadb
from chromadb.config import Settings


class EventRecord(TypedDict):
    """Type definition for market event records."""
    id: str
    ticker: str
    event_type: str
    date: str
    headline: str
    embedding: list[float]


class FilingRecord(TypedDict):
    """Type definition for SEC filing records."""
    id: str
    ticker: str
    filing_type: str
    date: str
    content: str
    embedding: list[float]


class SentinelChromaClient:
    """Typed wrapper around ChromaDB client for Sentinel vector storage."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize ChromaDB client with Sentinel schema.

        Args:
            db_path: Path to persist ChromaDB. Defaults to ./data/chroma.db.
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "chroma.db"
            )

        db_dir = os.path.dirname(db_path)
        if db_dir:
            Path(db_dir).mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_dir,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        """Create or retrieve collections for events and filings."""
        self.events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events and news"},
        )
        self.filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC EDGAR filings and regulatory documents"},
        )

    def add_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        date: str,
        headline: str,
        embedding: list[float],
    ) -> None:
        """Add a market event to the events collection.

        Args:
            event_id: Unique identifier for the event.
            ticker: Stock ticker symbol.
            event_type: Category (e.g., 'earnings', 'acquisition', 'recall').
            date: Event date in YYYY-MM-DD format.
            headline: Event headline or title.
            embedding: Vector embedding (typically 768 or 1536 dims).
        """
        self.events_collection.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[headline],
            metadatas=[
                {
                    "ticker": ticker,
                    "event_type": event_type,
                    "date": date,
                }
            ],
        )

    def add_filing(
        self,
        filing_id: str,
        ticker: str,
        filing_type: str,
        date: str,
        content: str,
        embedding: list[float],
    ) -> None:
        """Add a SEC filing to the filings collection.

        Args:
            filing_id: Unique identifier (typically CIK + accession number).
            ticker: Stock ticker symbol.
            filing_type: Form type (e.g., '8-K', '10-Q', '10-K').
            date: Filing date in YYYY-MM-DD format.
            content: Extracted filing text (typically truncated for embedding).
            embedding: Vector embedding.
        """
        self.filings_collection.add(
            ids=[filing_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[
                {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "date": date,
                }
            ],
        )

    def query_events(
        self,
        query_embedding: list[float],
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Query market events by embedding similarity.

        Args:
            query_embedding: Query vector embedding.
            ticker: Optional ticker filter.
            n_results: Number of results to return.

        Returns:
            Dict with 'ids', 'documents', 'distances', 'metadatas'.
        """
        where_filter = {"ticker": ticker} if ticker else None
        return self.events_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
        )

    def query_filings(
        self,
        query_embedding: list[float],
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Query SEC filings by embedding similarity.

        Args:
            query_embedding: Query vector embedding.
            ticker: Optional ticker filter.
            filing_type: Optional filing type filter (e.g., '8-K').
            n_results: Number of results to return.

        Returns:
            Dict with 'ids', 'documents', 'distances', 'metadatas'.
        """
        where_filter = {}
        if ticker:
            where_filter["ticker"] = ticker
        if filing_type:
            where_filter["filing_type"] = filing_type

        return self.filings_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter or None,
        )

    def persist(self) -> None:
        """Persist ChromaDB to disk."""
        self.client.persist()

    def get_event_count(self) -> int:
        """Return total count of market events in collection."""
        return self.events_collection.count()

    def get_filing_count(self) -> int:
        """Return total count of SEC filings in collection."""
        return self.filings_collection.count()

    def delete_event(self, event_id: str) -> None:
        """Delete a market event by ID."""
        self.events_collection.delete(ids=[event_id])

    def delete_filing(self, filing_id: str) -> None:
        """Delete a SEC filing by ID."""
        self.filings_collection.delete(ids=[filing_id])


def initialize_chromadb(
    db_path: Optional[str] = None,
) -> SentinelChromaClient:
    """Factory function to initialize a Sentinel ChromaDB client.

    Args:
        db_path: Optional custom path for ChromaDB persistence.

    Returns:
        Initialized SentinelChromaClient instance.
    """
    return SentinelChromaClient(db_path=db_path)
