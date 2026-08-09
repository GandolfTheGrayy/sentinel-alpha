"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up a local ChromaDB instance with collections for market events,
SEC filings, and sentiment signals. It provides a typed client interface for
inserting documents and querying embeddings via the RAG pipeline.

Role in Sentinel:
  - Initializes persistent vector DB on first run
  - Manages collections: "market_events", "sec_filings", "sentiment_signals"
  - Provides get_client() for use by historian/rag_query.py
  - Handles schema versioning and automatic migrations
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any
import chromadb
from chromadb.config import Settings


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DB_DIR = Path(__file__).parent.parent.parent / "data" / "chromadb"
DB_PERSIST_PATH = str(DB_DIR)
SCHEMA_VERSION = 1

COLLECTIONS = {
    "market_events": {
        "description": "Historical market events, earnings announcements, regulatory changes",
        "metadata": {"schema_version": SCHEMA_VERSION},
    },
    "sec_filings": {
        "description": "SEC 8-K, 10-Q, 10-K filings with extracted key facts",
        "metadata": {"schema_version": SCHEMA_VERSION},
    },
    "sentiment_signals": {
        "description": "Social media, news, and developer sentiment time-series",
        "metadata": {"schema_version": SCHEMA_VERSION},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class ChromaDBClient:
    """Typed wrapper around ChromaDB client with collection management."""

    def __init__(self, client: chromadb.Client) -> None:
        """Initialize wrapper with ChromaDB client."""
        self._client = client
        self._collections: Dict[str, Any] = {}

    def get_collection(self, name: str) -> Any:
        """Get or initialize a collection by name."""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata=COLLECTIONS[name]["metadata"],
            )
        return self._collections[name]

    def insert_document(
        self,
        collection: str,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or upsert a document into a collection."""
        coll = self.get_collection(collection)
        coll.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def insert_batch(
        self,
        collection: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch insert documents into a collection."""
        coll = self.get_collection(collection)
        coll.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas or [{} for _ in ids],
        )

    def query(
        self,
        collection: str,
        query_text: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """Query a collection by text similarity."""
        coll = self.get_collection(collection)
        results = coll.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def delete_collection(self, name: str) -> None:
        """Delete a collection (for testing/reset)."""
        self._client.delete_collection(name=name)
        if name in self._collections:
            del self._collections[name]

    def list_collections(self) -> List[str]:
        """List all collection names."""
        return [c.name for c in self._client.list_collections()]

    def collection_count(self, name: str) -> int:
        """Get document count in a collection."""
        coll = self.get_collection(name)
        return coll.count()


# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def init_chromadb() -> ChromaDBClient:
    """Initialize ChromaDB with persistent storage and default collections."""
    # Ensure data directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Configure ChromaDB for persistence
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_PERSIST_PATH,
        anonymized_telemetry=False,
    )

    # Create or connect to persistent client
    client = chromadb.Client(settings)

    # Initialize wrapper
    wrapper = ChromaDBClient(client)

    # Create default collections
    for collection_name in COLLECTIONS.keys():
        wrapper.get_collection(collection_name)

    return wrapper


def get_client() -> ChromaDBClient:
    """Get or create the global ChromaDB client (singleton pattern)."""
    if not hasattr(get_client, "_instance"):
        get_client._instance = init_chromadb()
    return get_client._instance


def reset_db() -> None:
    """Reset all collections (for testing/development)."""
    client = get_client()
    for collection_name in COLLECTIONS.keys():
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    # Reinitialize
    for collection_name in COLLECTIONS.keys():
        client.get_collection(collection_name)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA MIGRATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_schema_version() -> int:
    """Read schema version from metadata store."""
    metadata_file = DB_DIR / "schema_version.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
                return data.get("version", 0)
        except Exception:
            return 0
    return 0


def _set_schema_version(version: int) -> None:
    """Write schema version to metadata store."""
    metadata_file = DB_DIR / "schema_version.json"
    DB_DIR.mkdir(parents=True, exist_ok=True)
    with open(metadata_file, "w") as f:
        json.dump({"version": version}, f)


def check_schema_migration() -> bool:
    """Check if schema migration is needed; return True if migrated."""
    current = _get_schema_version()
    if current < SCHEMA_VERSION:
        reset_db()
        _set_schema_version(SCHEMA_VERSION)
        return True
    return False


if __name__ == "__main__":
    # Quick smoke test
    print(f"Initializing ChromaDB at {DB_PERSIST_PATH}...")
    wrapper = init_chromadb()
    print(f"Collections: {wrapper.list_collections()}")

    # Insert a test document
    wrapper.insert_document(
        "market_events",
        "test_1",
        "Apple announced Q3 earnings beat expectations.",
        {"ticker": "AAPL", "date": "2024-01-15", "type": "earnings"},
    )
    print(f"market_events count: {wrapper.collection_count('market_events')}")

    # Query test
    results = wrapper.query("market_events", "Apple earnings", n_results=1)
