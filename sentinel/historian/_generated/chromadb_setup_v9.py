"""
Sentinel ChromaDB vector database setup and client wrapper.

This module initializes and manages a local ChromaDB instance for the Sentinel
Sentiment Engine's RAG pipeline. It defines typed collections for market events,
SEC filings, and sentiment signals, and provides a high-level client wrapper
that handles database initialization, collection management, and query operations.

Used by: sentinel/historian/rag_query.py (queries), sentinel/scout/* (ingestion)
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Any
import chromadb
from chromadb.config import Settings


class ChromaDBClient:
    """Typed wrapper around ChromaDB for Sentinel's RAG pipeline."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize ChromaDB client with persistent local storage.
        
        Args:
            db_path: Path to local ChromaDB directory. Defaults to ./data/chroma_db
        """
        if db_path is None:
            db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        
        self.client = chromadb.Client(settings)
        self._collections_cache = {}
    
    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> chromadb.Collection:
        """Get or create a collection with optional metadata.
        
        Args:
            name: Collection name (e.g., "market_events", "sec_filings")
            metadata: Optional dict of collection-level metadata
            
        Returns:
            ChromaDB Collection object
        """
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {},
        )
        self._collections_cache[name] = collection
        return collection
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection by name.
        
        Args:
            name: Collection name to delete
        """
        try:
            self.client.delete_collection(name=name)
            if name in self._collections_cache:
                del self._collections_cache[name]
        except ValueError:
            pass  # Collection doesn't exist
    
    def persist(self) -> None:
        """Flush all changes to disk."""
        self.client.persist()
    
    def reset(self) -> None:
        """Clear all data and reset the database."""
        self.client.reset()
        self._collections_cache.clear()


def initialize_sentinel_collections(client: ChromaDBClient) -> dict[str, chromadb.Collection]:
    """Initialize all standard Sentinel collections.
    
    Args:
        client: ChromaDBClient instance
        
    Returns:
        Dict mapping collection names to collection objects
    """
    collections = {}
    
    # Market events: historical price movements, earnings, splits, etc.
    collections["market_events"] = client.get_or_create_collection(
        name="market_events",
        metadata={
            "description": "Historical market events (earnings, splits, rallies, crashes)",
            "embedding_model": "gemini-embedding",
        },
    )
    
    # SEC filings: 8-K, 10-Q, 10-K documents and extracted risk factors
    collections["sec_filings"] = client.get_or_create_collection(
        name="sec_filings",
        metadata={
            "description": "SEC EDGAR filings (8-K, 10-Q, 10-K) with extracted sentiment",
            "source": "SEC EDGAR",
        },
    )
    
    # News and sentiment: headlines, Reddit posts, HN discussions
    collections["sentiment_signals"] = client.get_or_create_collection(
        name="sentiment_signals",
        metadata={
            "description": "News headlines, social media, analyst sentiment",
            "sources": ["news_api", "reddit", "hackernews"],
        },
    )
    
    # Developer health: GitHub activity, commits, issues (for tech companies)
    collections["developer_signals"] = client.get_or_create_collection(
        name="developer_signals",
        metadata={
            "description": "GitHub activity, commit trends, contributor health",
            "source": "GitHub API",
        },
    )
    
    # Regulatory whispers: SEC comment letters, shareholder proposals, litigation
    collections["regulatory_signals"] = client.get_or_create_collection(
        name="regulatory_signals",
        metadata={
            "description": "Regulatory filings, shareholder actions, litigation signals",
            "source": "SEC EDGAR + news",
        },
    )
    
    return collections


def setup_chromadb(
    db_path: Optional[str] = None,
    reset: bool = False,
) -> tuple[ChromaDBClient, dict[str, chromadb.Collection]]:
    """One-shot setup: initialize client and all collections.
    
    Args:
        db_path: Custom ChromaDB path (defaults to env CHROMA_DB_PATH or ./data/chroma_db)
        reset: If True, wipe and reinitialize the database
        
    Returns:
        Tuple of (initialized ChromaDBClient, dict of collections)
    """
    client = ChromaDBClient(db_path=db_path)
    
    if reset:
        client.reset()
    
    collections = initialize_sentinel_collections(client)
    client.persist()
    
    return client, collections


def get_chromadb_client(db_path: Optional[str] = None) -> ChromaDBClient:
    """Get or create the singleton ChromaDB client.
    
    Args:
        db_path: Custom ChromaDB path
        
    Returns:
        ChromaDBClient instance
    """
    return ChromaDBClient(db_path=db_path)
