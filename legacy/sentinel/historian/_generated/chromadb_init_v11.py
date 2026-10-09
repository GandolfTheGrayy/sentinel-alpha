"""
ChromaDB Vector Database Initialization & Client Wrapper

This module initializes and manages the local ChromaDB vector store for Sentinel.
It creates persistent collections for market events, SEC filings, and news articles,
and exposes a typed client interface for RAG queries in the Historian pillar.

Integration: Called by sentinel/historian/rag_query.py during startup to ensure
the vector DB is ready for embedding-based semantic search across historical
financial signals.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


class ChromaDBClient:
    """Typed wrapper around ChromaDB persistent client with Sentinel collections."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize ChromaDB with persistent storage and standard collections.

        Args:
            db_path: Path to ChromaDB data directory. Defaults to ./data/chromadb
        """
        self.db_path = db_path or os.path.join("data", "chromadb")
        Path(self.db_path).mkdir(parents=True, exist_ok=True)

        settings = Settings(
            is_persistent=True,
            persist_directory=self.db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self._collections = {}

    def get_or_create_collection(self, name: str, metadata: Optional[dict] = None) -> chromadb.Collection:
        """Get or create a named collection in the vector database.

        Args:
            name: Collection name (e.g., "market_events", "sec_filings", "news")
            metadata: Optional metadata dict for the collection

        Returns:
            ChromaDB Collection object ready for add/query operations
        """
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {"description": f"Sentinel {name} collection"},
            )
        return self._collections[name]

    def initialize_standard_collections(self) -> None:
        """Create all standard Sentinel collections if they don't exist."""
        collections_meta = {
            "market_events": {
                "description": "Historical market events, earnings, M&A, guidance changes"
            },
            "sec_filings": {
                "description": "SEC 8-K, 10-Q, 10-K filing text and summaries"
            },
            "news_articles": {
                "description": "News headlines and snippets from Reuters, AP, CNBC"
            },
            "reddit_sentiment": {
                "description": "Reddit discussion posts and community sentiment signals"
            },
        }
        for collection_name, meta in collections_meta.items():
            self.get_or_create_collection(collection_name, meta)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Add documents to a collection.

        Args:
            collection_name: Name of the target collection
            documents: List of document texts to embed and store
            metadatas: List of metadata dicts (one per document)
            ids: List of unique document IDs
        """
        collection = self.get_or_create_collection(collection_name)
        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
    ) -> dict:
        """Query a collection by semantic similarity.

        Args:
            collection_name: Name of the collection to query
            query_texts: List of query strings
            n_results: Number of results per query

        Returns:
            Dict with 'ids', 'distances', 'metadatas', 'documents' keys
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.query(query_texts=query_texts, n_results=n_results)

    def delete_collection(self, name: str) -> None:
        """Delete a collection from the database.

        Args:
            name: Name of the collection to delete
        """
        self.client.delete_collection(name=name)
        if name in self._collections:
            del self._collections[name]

    def persist(self) -> None:
        """Explicitly persist in-memory state to disk."""
        self.client.persist()

    def get_collection_count(self, collection_name: str) -> int:
        """Get the number of documents in a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Number of documents stored
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.count()


def initialize_chromadb(db_path: Optional[str] = None) -> ChromaDBClient:
    """Factory function: create and initialize a ready-to-use ChromaDB client.

    Args:
        db_path: Optional custom path to ChromaDB data directory

    Returns:
        Initialized ChromaDBClient with all standard collections created
    """
    client = ChromaDBClient(db_path=db_path)
    client.initialize_standard_collections()
    return client


if __name__ == "__main__":
    client = initialize_chromadb()
    print(f"✓ ChromaDB initialized at {client.db_path}")
    print("✓ Standard collections created:")
    for name in ["market_events", "sec_filings", "news_articles", "reddit_sentiment"]:
        count = client.get_collection_count(name)
        print(f"  - {name}: {count} documents")
