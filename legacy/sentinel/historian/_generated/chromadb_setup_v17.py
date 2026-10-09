"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with typed collections
for market events, SEC filings, and news articles. It provides a singleton client
wrapper that ensures consistent embedding and querying across the historian pillar.

Used by: sentinel/historian/rag_query.py (RAG pipeline)
         sentinel/judge/predictor.py (historical event lookup during prediction)
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Any

import chromadb
from chromadb.config import Settings


_DB_PATH = Path(os.getenv("SENTINEL_DB_PATH", "./sentinel_data/chromadb"))
_COLLECTIONS = {
    "market_events": "Historical market events, earnings, economic data",
    "sec_filings": "SEC 8-K, 10-Q, 10-K documents and summaries",
    "news_articles": "News headlines and article snippets with sentiment",
}


class ChromaDBClient:
    """Typed wrapper for ChromaDB collections; ensures singleton initialization."""

    _instance: Optional["ChromaDBClient"] = None
    _client: Optional[chromadb.HttpClient | chromadb.PersistentClient] = None
    _collections: dict[str, chromadb.Collection] = {}

    def __new__(cls) -> "ChromaDBClient":
        """Return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self) -> None:
        """Initialize persistent ChromaDB client and create collections."""
        _DB_PATH.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(_DB_PATH),
            anonymized_telemetry=False,
        )

        self._client = chromadb.Client(settings)

        for collection_name, description in _COLLECTIONS.items():
            try:
                self._collections[collection_name] = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"description": description, "hnsw:space": "cosine"},
                )
            except Exception as e:
                print(f"Warning: failed to create collection '{collection_name}': {e}")

    def add_market_events(
        self,
        ticker: str,
        event_type: str,
        description: str,
        date: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a market event (earnings, split, dividend) to the events collection."""
        doc_id = f"{ticker}_{event_type}_{date}".replace(" ", "_")
        meta = metadata or {}
        meta.update({"ticker": ticker, "event_type": event_type, "date": date})

        self._collections["market_events"].add(
            ids=[doc_id],
            documents=[description],
            metadatas=[meta],
        )
        return doc_id

    def add_sec_filing(
        self,
        ticker: str,
        accession_number: str,
        filing_type: str,
        text_summary: str,
        filing_date: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a SEC filing summary to the filings collection."""
        doc_id = accession_number
        meta = metadata or {}
        meta.update({
            "ticker": ticker,
            "filing_type": filing_type,
            "accession_number": accession_number,
            "filing_date": filing_date,
        })

        self._collections["sec_filings"].add(
            ids=[doc_id],
            documents=[text_summary],
            metadatas=[meta],
        )
        return doc_id

    def add_news_article(
        self,
        ticker: str,
        headline: str,
        summary: str,
        date: str,
        source: str,
        sentiment_score: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a news article snippet to the news collection."""
        doc_id = f"{ticker}_{source}_{date}".replace(" ", "_").replace("/", "_")
        meta = metadata or {}
        meta.update({
            "ticker": ticker,
            "source": source,
            "date": date,
        })
        if sentiment_score is not None:
            meta["sentiment_score"] = sentiment_score

        full_text = f"{headline}\n{summary}"
        self._collections["news_articles"].add(
            ids=[doc_id],
            documents=[full_text],
            metadatas=[meta],
        )
        return doc_id

    def query_market_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Query market events collection; optionally filter by ticker."""
        where_clause = {"ticker": ticker} if ticker else None
        return self._collections["market_events"].query(
            query_texts=[query_text],
            where=where_clause,
            n_results=n_results,
        )

    def query_sec_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Query SEC filings collection; optionally filter by ticker and/or filing type."""
        where = {}
        if ticker:
            where["ticker"] = ticker
        if filing_type:
            where["filing_type"] = filing_type

        where_clause = where if where else None
        return self._collections["sec_filings"].query(
            query_texts=[query_text],
            where=where_clause,
            n_results=n_results,
        )

    def query_news_articles(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Query news articles collection; optionally filter by ticker and/or source."""
        where = {}
        if ticker:
            where["ticker"] = ticker
        if source:
            where["source"] = source

        where_clause = where if where else None
        return self._collections["news_articles"].query(
            query_texts=[query_text],
            where=where_clause,
            n_results=n_results,
        )

    def get_collection(self, name: str) -> chromadb.Collection:
        """Return the raw ChromaDB collection by name."""
        if name not in self._collections:
            raise ValueError(f"Collection '{name}' not found. Available: {list(_COLLECTIONS.keys())}")
        return self._collections[name]

    def delete_by_filter(
        self,
        collection_name: str,
        where: dict[str, Any],
    ) -> None:
        """Delete documents from a collection matching a where filter."""
        collection = self.get_collection(collection_name)
        collection.delete(where=where)

    def stats(self) -> dict[str, int]:
        """Return document counts per collection."""
        return {
            name: collection.count()
            for name, collection in self._collections.items()
        }


def get_chromadb_client() -> ChromaDBClient:
    """Factory function to retrieve the singleton ChromaDB client."""
    return ChromaDBClient()


def reset_chromadb() -> None:
    """Dangerous: wipe all collections (for testing)."""
    client = get_chromadb_client()
    for collection_name in _COLLECTIONS.keys():
        try:
            client._client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Warning: failed to delete collection '{collection_name}': {e}")
    ChromaDBClient._instance = None
    ChromaDBClient._
