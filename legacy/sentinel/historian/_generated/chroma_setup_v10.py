"""
Sentinel Historian — ChromaDB Vector Database Initialization & Client Wrapper

This module initializes and manages a local ChromaDB instance for Sentinel's RAG pipeline.
It creates persistent collections for market events and SEC filings, providing a typed
client wrapper for embedding-based similarity search across historical financial signals.

Integration: Called once during Sentinel startup (sentinel/pipeline.py) to ensure the
vector DB is ready. Historian.rag_query uses the client to retrieve contextual events.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


def get_chroma_client() -> chromadb.Client:
    """Initialize and return a persistent ChromaDB client with local SQLite backing."""
    db_dir = Path(os.getenv("SENTINEL_DB_DIR", "./data/chroma_db"))
    db_dir.mkdir(parents=True, exist_ok=True)
    
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(db_dir),
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def init_collections(client: chromadb.Client) -> dict:
    """Initialize market events and SEC filings collections; return collection dict."""
    try:
        market_events_col = client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, corrections, crashes, rallies"},
        )
    except Exception as e:
        print(f"[WARN] Failed to get/create market_events collection: {e}")
        market_events_col = None

    try:
        sec_filings_col = client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC 8-K, 10-Q, 10-K excerpts and summaries"},
        )
    except Exception as e:
        print(f"[WARN] Failed to get/create sec_filings collection: {e}")
        sec_filings_col = None

    try:
        sentiment_signals_col = client.get_or_create_collection(
            name="sentiment_signals",
            metadata={"description": "Reddit/HN sentiment posts, news headlines, tone shifts"},
        )
    except Exception as e:
        print(f"[WARN] Failed to get/create sentiment_signals collection: {e}")
        sentiment_signals_col = None

    return {
        "market_events": market_events_col,
        "sec_filings": sec_filings_col,
        "sentiment_signals": sentiment_signals_col,
    }


class ChromaVectorDB:
    """Typed wrapper around ChromaDB client for Sentinel's RAG queries."""

    def __init__(self, client: Optional[chromadb.Client] = None):
        """Initialize ChromaVectorDB with an optional pre-initialized client."""
        self.client = client or get_chroma_client()
        self.collections = init_collections(self.client)

    def add_market_event(
        self,
        event_id: str,
        date: str,
        ticker: str,
        event_type: str,
        description: str,
        metadata_dict: Optional[dict] = None,
    ) -> None:
        """Add a market event (crash, earnings beat, regulatory action) to vector DB."""
        if not self.collections["market_events"]:
            print("[WARN] market_events collection unavailable; skipping add")
            return

        meta = metadata_dict or {}
        meta.update({"date": date, "ticker": ticker, "event_type": event_type})

        self.collections["market_events"].add(
            ids=[event_id],
            documents=[description],
            metadatas=[meta],
        )

    def add_sec_filing(
        self,
        filing_id: str,
        ticker: str,
        form_type: str,
        date: str,
        content_excerpt: str,
        metadata_dict: Optional[dict] = None,
    ) -> None:
        """Add a SEC filing excerpt (8-K, 10-Q, 10-K) to the vector DB."""
        if not self.collections["sec_filings"]:
            print("[WARN] sec_filings collection unavailable; skipping add")
            return

        meta = metadata_dict or {}
        meta.update({"ticker": ticker, "form_type": form_type, "date": date})

        self.collections["sec_filings"].add(
            ids=[filing_id],
            documents=[content_excerpt],
            metadatas=[meta],
        )

    def add_sentiment_signal(
        self,
        signal_id: str,
        ticker: str,
        source: str,
        date: str,
        text: str,
        metadata_dict: Optional[dict] = None,
    ) -> None:
        """Add a sentiment signal (Reddit post, news headline, etc.) to the vector DB."""
        if not self.collections["sentiment_signals"]:
            print("[WARN] sentiment_signals collection unavailable; skipping add")
            return

        meta = metadata_dict or {}
        meta.update({"ticker": ticker, "source": source, "date": date})

        self.collections["sentiment_signals"].add(
            ids=[signal_id],
            documents=[text],
            metadatas=[meta],
        )

    def query_market_events(
        self, query_text: str, ticker: Optional[str] = None, n_results: int = 5
    ) -> dict:
        """Retrieve similar market events via vector similarity."""
        if not self.collections["market_events"]:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        where_filter = None
        if ticker:
            where_filter = {"ticker": {"$eq": ticker}}

        try:
            results = self.collections["market_events"].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )
            return {
                "ids": results.get("ids", [[]])[0],
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
            }
        except Exception as e:
            print(f"[WARN] market_events query failed: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def query_sec_filings(
        self, query_text: str, ticker: Optional[str] = None, n_results: int = 5
    ) -> dict:
        """Retrieve similar SEC filings via vector similarity."""
        if not self.collections["sec_filings"]:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        where_filter = None
        if ticker:
            where_filter = {"ticker": {"$eq": ticker}}

        try:
            results = self.collections["sec_filings"].query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_filter,
            )
            return {
                "ids": results.get("ids", [[]])[0],
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0],
                "distances": results.get("distances", [[]])[0],
            }
        except Exception as e:
            print(f"[WARN] sec_filings query failed: {e}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def query_sentiment_signals(
        self, query_text: str, ticker: Optional[str] = None, n_results: int = 5
    ) -> dict:
        """Retrieve similar sentiment signals via vector similarity."""
        if not self.collections["sentiment_signals"]:
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

        where_filter = None
        if ticker:
            where_filter = {"ticker": {"$eq":
