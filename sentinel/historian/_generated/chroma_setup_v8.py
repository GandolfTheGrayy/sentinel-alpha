"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with collections for
market events, SEC filings, and sentiment signals. It provides a typed async-safe
client wrapper that Historian uses for RAG embedding storage and retrieval.

Fits into Sentinel pipeline: Historian pillar uses this to persist embeddings
and support semantic search across historical market data and regulatory filings.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Any
import chromadb
from chromadb.config import Settings


# Module-level constants
DB_DIR = Path(os.getenv("SENTINEL_DB_DIR", "./data/chroma"))
COLLECTION_MARKET_EVENTS = "market_events"
COLLECTION_SEC_FILINGS = "sec_filings"
COLLECTION_SENTIMENT_SIGNALS = "sentiment_signals"


def initialize_chroma_db() -> chromadb.Client:
    """Initialize and return a ChromaDB client with persistent local storage."""
    db_dir = DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(db_dir),
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def ensure_collections(client: chromadb.Client) -> dict[str, chromadb.Collection]:
    """Ensure all required collections exist; return dict of collection objects."""
    collections = {}
    
    for collection_name in [
        COLLECTION_MARKET_EVENTS,
        COLLECTION_SEC_FILINGS,
        COLLECTION_SENTIMENT_SIGNALS,
    ]:
        try:
            collection = client.get_collection(name=collection_name)
        except ValueError:
            # Collection does not exist; create it
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        collections[collection_name] = collection
    
    return collections


class ChromaClientWrapper:
    """Typed wrapper around ChromaDB client for Historian operations."""

    def __init__(self, client: Optional[chromadb.Client] = None):
        """Initialize wrapper with existing client or create new one."""
        self.client = client or initialize_chroma_db()
        self.collections = ensure_collections(self.client)

    def add_event(
        self,
        event_id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Add a market event embedding to the market_events collection."""
        collection = self.collections[COLLECTION_MARKET_EVENTS]
        collection.add(
            ids=[event_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
        )

    def add_filing(
        self,
        filing_id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Add a SEC filing embedding to the sec_filings collection."""
        collection = self.collections[COLLECTION_SEC_FILINGS]
        collection.add(
            ids=[filing_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
        )

    def add_sentiment_signal(
        self,
        signal_id: str,
        text: str,
        metadata: dict[str, Any],
        embedding: Optional[list[float]] = None,
    ) -> None:
        """Add a sentiment signal embedding to the sentiment_signals collection."""
        collection = self.collections[COLLECTION_SENTIMENT_SIGNALS]
        collection.add(
            ids=[signal_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding] if embedding else None,
        )

    def query_events(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Semantically search market_events collection; return matches with scores."""
        collection = self.collections[COLLECTION_MARKET_EVENTS]
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def query_filings(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Semantically search sec_filings collection; return matches with scores."""
        collection = self.collections[COLLECTION_SEC_FILINGS]
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def query_sentiment(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Semantically search sentiment_signals collection; return matches with scores."""
        collection = self.collections[COLLECTION_SENTIMENT_SIGNALS]
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def get_collection_count(self, collection_name: str) -> int:
        """Return the number of items in a named collection."""
        if collection_name not in self.collections:
            return 0
        collection = self.collections[collection_name]
        return collection.count()

    def delete_collection(self, collection_name: str) -> None:
        """Delete a named collection (for testing/reset)."""
        if collection_name in self.collections:
            self.client.delete_collection(name=collection_name)
            del self.collections[collection_name]

    def persist(self) -> None:
        """Explicitly persist all changes to disk."""
        self.client.persist()


def get_global_chroma_client() -> ChromaClientWrapper:
    """Singleton accessor for the global ChromaDB client wrapper."""
    if not hasattr(get_global_chroma_client, "_instance"):
        get_global_chroma_client._instance = ChromaClientWrapper()
    return get_global_chroma_client._instance


if __name__ == "__main__":
    # Quick smoke test
    wrapper = ChromaClientWrapper()
    print(f"Market events count: {wrapper.get_collection_count(COLLECTION_MARKET_EVENTS)}")
    print(f"SEC filings count: {wrapper.get_collection_count(COLLECTION_SEC_FILINGS)}")
    print(f"Sentiment signals count: {wrapper.get_collection_count(COLLECTION_SENTIMENT_SIGNALS)}")
    print("ChromaDB initialized successfully.")
