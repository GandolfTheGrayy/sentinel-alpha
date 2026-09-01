"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up a local ChromaDB instance with typed collections for
market events, SEC filings, and sentiment signals. It provides a unified
client interface for the Historian pillar to store and retrieve embeddings
cross-referenced with historical data for RAG-based prediction context.

Part of the Historian pillar — enables efficient semantic search over
ingested financial documents and market events.
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, Dict, List, Any

import chromadb
from chromadb.config import Settings


class SentinelChromaClient:
    """Typed wrapper around ChromaDB collections for Sentinel's historian."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize ChromaDB client and collections."""
        if db_path is None:
            db_path = os.getenv("SENTINEL_CHROMA_PATH", "./data/chroma")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        
        self.client = chromadb.Client(settings)
        self._ensure_collections()
    
    def _ensure_collections(self) -> None:
        """Create or retrieve collections for market events and SEC filings."""
        self.market_events = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, news, and sentiment signals"},
        )
        
        self.sec_filings = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC 8-K, 10-Q, 10-K filings with extracted entities"},
        )
        
        self.earnings_calendar = self.client.get_or_create_collection(
            name="earnings_calendar",
            metadata={"description": "Earnings dates, guidance, and surprise deltas"},
        )
    
    def add_market_event(
        self,
        event_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a market event (news, Reddit post, etc.) to the market_events collection."""
        self.market_events.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
    
    def add_sec_filing(
        self,
        filing_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a parsed SEC filing to the sec_filings collection."""
        self.sec_filings.add(
            ids=[filing_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
    
    def add_earnings_event(
        self,
        event_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add an earnings event to the earnings_calendar collection."""
        self.earnings_calendar.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
    
    def query_market_events(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query market_events collection by embedding similarity."""
        return self.market_events.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
    
    def query_sec_filings(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query sec_filings collection by embedding similarity."""
        return self.sec_filings.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
    
    def query_earnings(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Query earnings_calendar collection by embedding similarity."""
        return self.earnings_calendar.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return counts and metadata for all collections."""
        return {
            "market_events": {
                "count": self.market_events.count(),
                "metadata": self.market_events.metadata,
            },
            "sec_filings": {
                "count": self.sec_filings.count(),
                "metadata": self.sec_filings.metadata,
            },
            "earnings_calendar": {
                "count": self.earnings_calendar.count(),
                "metadata": self.earnings_calendar.metadata,
            },
        }
    
    def persist(self) -> None:
        """Explicitly persist ChromaDB to disk."""
        self.client.persist()
    
    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection by name (for testing/reset)."""
        self.client.delete_collection(collection_name)


def initialize_sentinel_chroma(db_path: Optional[str] = None) -> SentinelChromaClient:
    """Factory function to initialize and return a ready Sentinel ChromaDB client."""
    return SentinelChromaClient(db_path=db_path)


if __name__ == "__main__":
    chroma = initialize_sentinel_chroma()
    stats = chroma.get_collection_stats()
    print("ChromaDB collections initialized:")
    print(json.dumps(stats, indent=2))
