"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up and manages the local ChromaDB instance used by the Historian
pillar for RAG operations. It initializes collections for market events, SEC filings,
and news sentiment, and provides a typed client interface for embedding-based retrieval
across the Sentinel pipeline.
"""

import os
from pathlib import Path
from typing import Optional, Any
import chromadb
from chromadb.config import Settings


# Global ChromaDB client instance
_client: Optional[chromadb.HttpClient | chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Initialize and return the singleton ChromaDB persistent client."""
    global _client
    if _client is not None:
        return _client

    db_path = os.getenv("SENTINEL_CHROMA_DB_PATH", "./data/chroma_db")
    Path(db_path).mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(path=db_path)
    return _client


def init_collections() -> dict[str, chromadb.Collection]:
    """Initialize and return all required ChromaDB collections."""
    client = get_chroma_client()

    collections = {}

    # Market events collection: historical price movements, earnings, splits
    collections["market_events"] = client.get_or_create_collection(
        name="market_events",
        metadata={"description": "Historical market events: earnings, splits, M&A"},
        embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction(),
    )

    # SEC filings collection: 8-K, 10-Q, 10-K documents
    collections["sec_filings"] = client.get_or_create_collection(
        name="sec_filings",
        metadata={"description": "SEC EDGAR filings (8-K, 10-Q, 10-K) with metadata"},
        embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction(),
    )

    # News sentiment collection: headlines and sentiment scores
    collections["news_sentiment"] = client.get_or_create_collection(
        name="news_sentiment",
        metadata={"description": "News headlines with sentiment polarity and ticker"},
        embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction(),
    )

    # Reddit/social sentiment collection: discourse and topic shifts
    collections["social_sentiment"] = client.get_or_create_collection(
        name="social_sentiment",
        metadata={"description": "Reddit, HN, and community sentiment for tickers"},
        embedding_function=chromadb.utils.embedding_functions.DefaultEmbeddingFunction(),
    )

    return collections


def add_market_event(
    collection: chromadb.Collection,
    event_id: str,
    ticker: str,
    event_type: str,
    description: str,
    date: str,
    impact_direction: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Add a historical market event to the market_events collection."""
    doc_metadata = {
        "ticker": ticker,
        "event_type": event_type,
        "date": date,
        "impact_direction": impact_direction,
    }
    if metadata:
        doc_metadata.update(metadata)

    collection.add(
        ids=[event_id],
        documents=[description],
        metadatas=[doc_metadata],
    )


def add_sec_filing(
    collection: chromadb.Collection,
    filing_id: str,
    ticker: str,
    filing_type: str,
    filing_date: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Add a SEC filing document to the sec_filings collection."""
    doc_metadata = {
        "ticker": ticker,
        "filing_type": filing_type,
        "filing_date": filing_date,
    }
    if metadata:
        doc_metadata.update(metadata)

    collection.add(
        ids=[filing_id],
        documents=[content],
        metadatas=[doc_metadata],
    )


def add_news_item(
    collection: chromadb.Collection,
    news_id: str,
    ticker: str,
    headline: str,
    sentiment_score: float,
    published_date: str,
    source: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Add a news headline with sentiment to the news_sentiment collection."""
    doc_metadata = {
        "ticker": ticker,
        "sentiment_score": str(sentiment_score),
        "published_date": published_date,
        "source": source,
    }
    if metadata:
        doc_metadata.update(metadata)

    collection.add(
        ids=[news_id],
        documents=[headline],
        metadatas=[doc_metadata],
    )


def add_social_signal(
    collection: chromadb.Collection,
    signal_id: str,
    ticker: str,
    platform: str,
    text: str,
    sentiment_score: float,
    timestamp: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Add a social media/community signal to the social_sentiment collection."""
    doc_metadata = {
        "ticker": ticker,
        "platform": platform,
        "sentiment_score": str(sentiment_score),
        "timestamp": timestamp,
    }
    if metadata:
        doc_metadata.update(metadata)

    collection.add(
        ids=[signal_id],
        documents=[text],
        metadatas=[doc_metadata],
    )


def query_collection(
    collection: chromadb.Collection,
    query_text: str,
    ticker_filter: Optional[str] = None,
    n_results: int = 5,
) -> dict[str, Any]:
    """Query a collection by semantic similarity with optional ticker filter."""
    where_filter = None
    if ticker_filter:
        where_filter = {"ticker": {"$eq": ticker_filter}}

    results = collection.query(
        query_texts=[query_text],
        where=where_filter,
        n_results=n_results,
    )

    return results


def delete_collection(collection_name: str) -> None:
    """Delete a collection by name (useful for testing/reset)."""
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass


def reset_database() -> None:
    """Clear all collections and reinitialize (testing only)."""
    global _client
    client = get_chroma_client()
    for collection_name in ["market_events", "sec_filings", "news_sentiment", "social_sentiment"]:
        delete_collection(collection_name)
    _client = None
