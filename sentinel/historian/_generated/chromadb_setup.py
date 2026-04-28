"""
ChromaDB Vector Database Setup for Sentinel Historian.

This module initializes and manages the local ChromaDB vector database instance,
defines typed collections for market events and SEC filings, and provides a
wrapper client for RAG queries. Serves as the persistent knowledge store for
historical event lookup and confidence score weighting across Sentinel pipelines.
"""

import os
import json
from pathlib import Path
from typing import Optional, Any
import chromadb
from chromadb.config import Settings


class ChromaDBClient:
    """Typed wrapper around ChromaDB for Sentinel Historian operations."""

    def __init__(self, db_path: str = "data/chromadb") -> None:
        """Initialize ChromaDB client with persistent local storage."""
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
        """Initialize or retrieve standard collections for market events and filings."""
        self.market_events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, price swings, sentiment reversals"}
        )
        
        self.sec_filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "Parsed SEC 8-K, 10-Q, 10-K documents with regulatory signals"}
        )
        
        self.sentiment_signals_collection = self.client.get_or_create_collection(
            name="sentiment_signals",
            metadata={"description": "Reddit, HN, GitHub sentiment snapshots with timestamps"}
        )

    def add_market_event(
        self,
        event_id: str,
        text: str,
        ticker: str,
        event_date: str,
        price_move_pct: float,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Add a historical market event to the vector store."""
        meta = metadata or {}
        meta.update({
            "ticker": ticker,
            "event_date": event_date,
            "price_move_pct": price_move_pct
        })
        self.market_events_collection.add(
            ids=[event_id],
            documents=[text],
            metadatas=[meta]
        )

    def add_sec_filing(
        self,
        filing_id: str,
        text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
        cik: str,
        regulatory_flags: Optional[list[str]] = None
    ) -> None:
        """Add a parsed SEC filing to the vector store with regulatory signal flags."""
        meta = {
            "ticker": ticker,
            "filing_type": filing_type,
            "filing_date": filing_date,
            "cik": cik,
            "regulatory_flags": json.dumps(regulatory_flags or [])
        }
        self.sec_filings_collection.add(
            ids=[filing_id],
            documents=[text],
            metadatas=[meta]
        )

    def add_sentiment_signal(
        self,
        signal_id: str,
        text: str,
        source: str,
        ticker: str,
        signal_date: str,
        sentiment_score: float,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Add a sentiment signal snapshot (Reddit, HN, GitHub) to the vector store."""
        meta = metadata or {}
        meta.update({
            "source": source,
            "ticker": ticker,
            "signal_date": signal_date,
            "sentiment_score": sentiment_score
        })
        self.sentiment_signals_collection.add(
            ids=[signal_id],
            documents=[text],
            metadatas=[meta]
        )

    def query_market_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = 5
    ) -> dict[str, Any]:
        """Retrieve similar historical market events via vector similarity."""
        where_filter = None
        if ticker:
            where_filter = {"ticker": {"$eq": ticker}}
        
        results = self.market_events_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results

    def query_sec_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        n_results: int = 5
    ) -> dict[str, Any]:
        """Retrieve similar SEC filings via vector similarity with optional filters."""
        where_filter = None
        if ticker or filing_type:
            where_filter = {}
            if ticker:
                where_filter["ticker"] = {"$eq": ticker}
            if filing_type:
                where_filter["filing_type"] = {"$eq": filing_type}
        
        results = self.sec_filings_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results

    def query_sentiment_signals(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        n_results: int = 5
    ) -> dict[str, Any]:
        """Retrieve similar sentiment signals via vector similarity."""
        where_filter = None
        if ticker or source:
            where_filter = {}
            if ticker:
                where_filter["ticker"] = {"$eq": ticker}
            if source:
                where_filter["source"] = {"$eq": source}
        
        results = self.sentiment_signals_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )
        return results

    def get_collection_stats(self) -> dict[str, int]:
        """Return document counts for all collections."""
        return {
            "market_events": self.market_events_collection.count(),
            "sec_filings": self.sec_filings_collection.count(),
            "sentiment_signals": self.sentiment_signals_collection.count()
        }

    def reset_collections(self) -> None:
        """Delete and reinitialize all collections (use with caution)."""
        self.client.delete_collection(name="market_events")
        self.client.delete_collection(name="sec_filings")
        self.client.delete_collection(name="sentiment_signals")
        self._init_collections()

    def persist(self) -> None:
        """Explicitly persist database to disk."""
        self.client.persist()


def init_chromadb(db_path: str = "data/chromadb") -> ChromaDBClient:
    """Factory function to initialize and return a ChromaDB client."""
    return ChromaDBClient(db_path=db_path)


if __name__ == "__main__":
    client = init_chromadb()
    print("ChromaDB initialized successfully.")
    print("Collections:", client.get_collection_stats())
