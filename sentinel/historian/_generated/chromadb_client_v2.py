"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with collections
for market events, SEC filings, and sentiment signals. It provides a typed
client interface for RAG queries and document insertion across the historian
pillar. Called by rag_query.py and indexing routines in the scout pillar.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Any

import chromadb
from chromadb.config import Settings


# Default database path: sentinel/data/chromadb/
DB_DIR = Path(os.getenv("SENTINEL_DATA_DIR", "sentinel/data")) / "chromadb"


def initialize_chromadb(
    persist_dir: Optional[Path] = None,
    reset: bool = False,
) -> chromadb.Client:
    """
    Initialize a persistent ChromaDB client with default collections.

    Args:
        persist_dir: Path to ChromaDB storage. Defaults to sentinel/data/chromadb/.
        reset: If True, delete existing DB and reinitialize (caution).

    Returns:
        chromadb.Client configured for persistent storage.
    """
    if persist_dir is None:
        persist_dir = DB_DIR

    if reset and persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(persist_dir),
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def get_or_create_collection(
    client: chromadb.Client,
    name: str,
    metadata: Optional[dict[str, Any]] = None,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection with optional metadata.

    Args:
        client: ChromaDB client instance.
        name: Collection name (e.g., "market_events", "sec_filings").
        metadata: Optional dict of collection-level metadata.

    Returns:
        chromadb.Collection ready for embeddings and queries.
    """
    if metadata is None:
        metadata = {}
    return client.get_or_create_collection(
        name=name,
        metadata=metadata,
    )


def setup_default_collections(client: chromadb.Client) -> dict[str, chromadb.Collection]:
    """
    Create or retrieve standard Sentinel collections.

    Returns a dict mapping collection names to chromadb.Collection objects:
      - "market_events": Historical price movements, earnings, splits.
      - "sec_filings": 8-K, 10-Q, 10-K documents from EDGAR.
      - "news_sentiment": Aggregated headlines and sentiment scores.
      - "reddit_signals": Discussion volume and upvote trends.
      - "dev_health": GitHub issue/star velocity for tech stocks.

    Args:
        client: ChromaDB client instance.

    Returns:
        Dict[str, chromadb.Collection] of initialized collections.
    """
    collections = {}

    collections["market_events"] = get_or_create_collection(
        client,
        "market_events",
        metadata={
            "description": "Historical price movements, earnings announcements, splits.",
            "source": "yfinance, news, SEC calendars",
        },
    )

    collections["sec_filings"] = get_or_create_collection(
        client,
        "sec_filings",
        metadata={
            "description": "8-K, 10-Q, 10-K documents from EDGAR.",
            "source": "sec.gov via scout/sec_filings.py",
        },
    )

    collections["news_sentiment"] = get_or_create_collection(
        client,
        "news_sentiment",
        metadata={
            "description": "Aggregated headlines with certainty and sentiment scores.",
            "source": "scout/news.py, linguist/sample_score.py",
        },
    )

    collections["reddit_signals"] = get_or_create_collection(
        client,
        "reddit_signals",
        metadata={
            "description": "Subreddit discussion volume, upvote trends, user sentiment.",
            "source": "praw (Reddit API) via scout modules",
        },
    )

    collections["dev_health"] = get_or_create_collection(
        client,
        "dev_health",
        metadata={
            "description": "GitHub issue/star velocity for tech stocks.",
            "source": "scout/github_signals.py",
        },
    )

    return collections


class ChromaDBWrapper:
    """
    Typed wrapper around ChromaDB client for Sentinel RAG operations.

    Manages connection lifecycle, collection access, and provides methods
    for adding documents, querying embeddings, and retrieving metadata.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        """
        Initialize ChromaDB wrapper with persistent storage.

        Args:
            persist_dir: Path to ChromaDB storage. Defaults to sentinel/data/chromadb/.
        """
        self.client = initialize_chromadb(persist_dir=persist_dir)
        self.collections = setup_default_collections(self.client)

    def add_documents(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """
        Add documents and metadata to a collection for embedding.

        Args:
            collection_name: Target collection (e.g., "sec_filings").
            documents: List of text documents to embed.
            metadatas: List of metadata dicts (one per document).
            ids: List of unique document IDs.

        Raises:
            KeyError: If collection_name does not exist.
        """
        if collection_name not in self.collections:
            raise KeyError(f"Collection '{collection_name}' not found. "
                          f"Available: {list(self.collections.keys())}")

        collection = self.collections[collection_name]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        collection_name: str,
        query_texts: list[str],
        n_results: int = 5,
    ) -> dict[str, Any]:
        """
        Query a collection by semantic similarity.

        Args:
            collection_name: Target collection.
            query_texts: List of query strings.
            n_results: Number of nearest neighbors to return per query.

        Returns:
            Dict with keys 'ids', 'distances', 'documents', 'metadatas'.

        Raises:
            KeyError: If collection_name does not exist.
        """
        if collection_name not in self.collections:
            raise KeyError(f"Collection '{collection_name}' not found.")

        collection = self.collections[collection_name]
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
        )

    def get_collection(self, name: str) -> chromadb.Collection:
        """
        Retrieve a collection by name for advanced operations.

        Args:
            name: Collection name.

        Returns:
            chromadb.Collection instance.

        Raises:
            KeyError: If collection does not exist.
        """
        if name not in self.collections:
            raise KeyError(f"Collection '{name}' not found.")
        return self.collections[name]

    def list_collections(self) -> list[str]:
        """
        List all available collection names.

        Returns:
            List of collection names.
        """
        return list(self.collections.keys())

    def collection_count(self, collection_name: str) -> int:
        """
        Return the document count in a collection.

        Args:
            collection_name: Target collection.

        Returns:
            Number of embedded documents.

        Raises:
            KeyError: If collection does not exist.
        """
        if collection_name not in self.collections:
            raise KeyError(
