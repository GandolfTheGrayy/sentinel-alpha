"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module initializes and manages the local ChromaDB instance, creating
collections for market events and SEC filings. It provides a typed client
wrapper that enforces schema consistency and enables RAG queries across
historical data ingested by Scout modules.

Part of the Historian pillar: persistent storage + retrieval of embeddings
for cross-referencing sentiment signals with past market behavior.
"""

import os
from typing import Optional
import chromadb
from chromadb.config import Settings


# Global client instance (lazy-loaded)
_client: Optional[chromadb.Client] = None
_db_path: str = os.getenv("SENTINEL_DB_PATH", ".sentinel_chromadb")


def get_client() -> chromadb.Client:
    """Initialize and return the global ChromaDB client (singleton pattern)."""
    global _client
    if _client is None:
        os.makedirs(_db_path, exist_ok=True)
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=_db_path,
            anonymized_telemetry=False,
        )
        _client = chromadb.Client(settings)
    return _client


def init_collections() -> dict[str, chromadb.Collection]:
    """Create and return all required collections (idempotent)."""
    client = get_client()
    
    collections = {}
    
    # Market Events collection: historical price movements, anomalies, macro shifts
    collections["market_events"] = client.get_or_create_collection(
        name="market_events",
        metadata={"description": "Historical market events, crashes, rallies, macro shifts"},
    )
    
    # SEC Filings collection: 8-K, 10-Q, 10-K documents indexed by ticker + date
    collections["sec_filings"] = client.get_or_create_collection(
        name="sec_filings",
        metadata={"description": "SEC EDGAR filings (8-K, 10-Q, 10-K) indexed by ticker"},
    )
    
    # News & Sentiment collection: headline sentiment over time
    collections["news_sentiment"] = client.get_or_create_collection(
        name="news_sentiment",
        metadata={"description": "News headlines and derived sentiment signals by ticker and date"},
    )
    
    # Reddit/Social collection: niche sentiment from Reddit, HN, etc.
    collections["social_signals"] = client.get_or_create_collection(
        name="social_signals",
        metadata={"description": "Reddit/HN/social media signals keyed by ticker and sentiment"},
    )
    
    return collections


class ChromaDBClient:
    """Typed wrapper around ChromaDB client for Sentinel queries."""
    
    def __init__(self) -> None:
        """Initialize the wrapper and ensure collections exist."""
        self._client = get_client()
        self._collections = init_collections()
    
    def add_market_event(
        self,
        event_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        """Add a market event embedding to the market_events collection."""
        self._collections["market_events"].add(
            ids=[event_id],
            documents=[text],
            metadatas=[metadata],
        )
    
    def add_sec_filing(
        self,
        filing_id: str,
        text: str,
        ticker: str,
        filing_type: str,
        date_filed: str,
    ) -> None:
        """Add a SEC filing embedding indexed by ticker and type."""
        metadata = {
            "ticker": ticker,
            "filing_type": filing_type,
            "date_filed": date_filed,
        }
        self._collections["sec_filings"].add(
            ids=[filing_id],
            documents=[text],
            metadatas=[metadata],
        )
    
    def add_news_item(
        self,
        news_id: str,
        headline: str,
        ticker: str,
        date_published: str,
        sentiment_score: float,
    ) -> None:
        """Add a news headline with sentiment to the news_sentiment collection."""
        metadata = {
            "ticker": ticker,
            "date_published": date_published,
            "sentiment_score": str(sentiment_score),
        }
        self._collections["news_sentiment"].add(
            ids=[news_id],
            documents=[headline],
            metadatas=[metadata],
        )
    
    def add_social_signal(
        self,
        signal_id: str,
        text: str,
        ticker: str,
        source: str,
        date_posted: str,
        sentiment: str,
    ) -> None:
        """Add a social media signal (Reddit, HN) to social_signals collection."""
        metadata = {
            "ticker": ticker,
            "source": source,
            "date_posted": date_posted,
            "sentiment": sentiment,
        }
        self._collections["social_signals"].add(
            ids=[signal_id],
            documents=[text],
            metadatas=[metadata],
        )
    
    def query_market_events(
        self,
        query_text: str,
        n_results: int = 5,
    ) -> dict:
        """Retrieve semantically similar historical market events."""
        return self._collections["market_events"].query(
            query_texts=[query_text],
            n_results=n_results,
        )
    
    def query_sec_by_ticker(
        self,
        ticker: str,
        query_text: str,
        filing_type: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """Query SEC filings for a given ticker, optionally filtered by type."""
        where = {"ticker": ticker}
        if filing_type:
            where["filing_type"] = filing_type
        
        return self._collections["sec_filings"].query(
            query_texts=[query_text],
            where=where,
            n_results=n_results,
        )
    
    def query_news_sentiment(
        self,
        ticker: str,
        query_text: str,
        n_results: int = 5,
    ) -> dict:
        """Retrieve news items and sentiment for a ticker."""
        return self._collections["news_sentiment"].query(
            query_texts=[query_text],
            where={"ticker": ticker},
            n_results=n_results,
        )
    
    def query_social_signals(
        self,
        ticker: str,
        query_text: str,
        source: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """Retrieve social signals for a ticker, optionally filtered by source."""
        where = {"ticker": ticker}
        if source:
            where["source"] = source
        
        return self._collections["social_signals"].query(
            query_texts=[query_text],
            where=where,
            n_results=n_results,
        )
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """Retrieve a collection by name for advanced queries."""
        return self._collections.get(name)
    
    def list_collections(self) -> dict[str, chromadb.Collection]:
        """Return all initialized collections."""
        return self._collections.copy()
    
    def reset_db(self) -> None:
        """Delete and reinitialize all collections (dev/testing only)."""
        for collection_name in list(self._collections.keys()):
            self._client.delete_collection(name=collection_name)
        self._collections = init_collections()


if __name__ == "__main__":
    # Quick sanity check: initialize and list collections
    db = ChromaDBClient()
    print("ChromaDB Collections initialized:")
    for name in db.list_collections().keys():
        print(f"  - {name}")
