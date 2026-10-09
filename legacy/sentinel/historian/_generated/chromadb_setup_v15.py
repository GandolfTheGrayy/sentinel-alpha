"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up and manages the local ChromaDB instance used by the historian
pillar. It initializes collections for market events, SEC filings, and news articles,
providing a typed client interface for RAG queries and document ingestion.

Used by: sentinel/historian/rag_query.py (embedding storage and retrieval)
"""

import os
from typing import Optional, Dict, List, Any
import chromadb
from chromadb.config import Settings


class SentinelChromaDB:
    """Typed wrapper around ChromaDB client for Sentinel's historian pillar."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            persist_dir: Path to persistent storage directory. Defaults to .sentineldb/
        """
        if persist_dir is None:
            persist_dir = os.path.join(os.path.expanduser("~"), ".sentineldb")

        os.makedirs(persist_dir, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir,
            anonymized_telemetry=False,
        )

        self.client = chromadb.Client(settings)
        self.persist_dir = persist_dir
        self._collections: Dict[str, Any] = {}

    def initialize_collections(self) -> None:
        """Create or retrieve standard collections for market events, filings, and news."""
        collection_names = [
            "market_events",
            "sec_filings_8k",
            "sec_filings_10q",
            "news_headlines",
            "reddit_posts",
        ]

        for name in collection_names:
            try:
                self._collections[name] = self.client.get_or_create_collection(
                    name=name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                print(f"Warning: Could not initialize collection {name}: {e}")

    def get_collection(self, collection_name: str) -> Any:
        """
        Retrieve a collection by name.

        Args:
            collection_name: Name of the collection (e.g., "market_events")

        Returns:
            ChromaDB collection object or None if not found.
        """
        if collection_name not in self._collections:
            try:
                self._collections[collection_name] = self.client.get_collection(
                    name=collection_name
                )
            except Exception as e:
                print(f"Error retrieving collection {collection_name}: {e}")
                return None
        return self._collections[collection_name]

    def upsert_documents(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Upsert documents into a collection with embeddings.

        Args:
            collection_name: Target collection name.
            ids: Unique document identifiers.
            embeddings: Pre-computed embedding vectors (list of floats).
            documents: Raw text content for each document.
            metadatas: Optional list of metadata dicts per document.
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            raise ValueError(f"Collection {collection_name} not found or failed to load")

        if metadatas is None:
            metadatas = [{} for _ in ids]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Query a collection by embedding vector.

        Args:
            collection_name: Collection to search.
            query_embedding: Query embedding vector.
            n_results: Number of results to return.

        Returns:
            Dict with "ids", "distances", "documents", "metadatas" keys.
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return {"ids": [], "distances": [], "documents": [], "metadatas": []}

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results

    def get_all(self, collection_name: str) -> Dict[str, Any]:
        """
        Retrieve all documents from a collection.

        Args:
            collection_name: Collection name.

        Returns:
            Dict with "ids", "embeddings", "documents", "metadatas" keys.
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        return collection.get()

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection by name.

        Args:
            collection_name: Name of collection to delete.
        """
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self._collections:
                del self._collections[collection_name]
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}")

    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.

        Args:
            collection_name: Collection name.

        Returns:
            Number of documents (or 0 if collection not found).
        """
        collection = self.get_collection(collection_name)
        if collection is None:
            return 0

        try:
            result = collection.get()
            return len(result.get("ids", []))
        except Exception:
            return 0

    def persist(self) -> None:
        """Explicitly persist all data to disk."""
        try:
            self.client.persist()
        except Exception as e:
            print(f"Warning: Could not persist ChromaDB: {e}")


def get_chroma_client(persist_dir: Optional[str] = None) -> SentinelChromaDB:
    """
    Factory function to instantiate and initialize a SentinelChromaDB client.

    Args:
        persist_dir: Optional custom persistence directory path.

    Returns:
        Initialized SentinelChromaDB instance with collections ready for use.
    """
    db = SentinelChromaDB(persist_dir=persist_dir)
    db.initialize_collections()
    return db
