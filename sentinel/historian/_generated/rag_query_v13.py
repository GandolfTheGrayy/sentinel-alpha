"""
RAG query interface for the Sentinel Sentiment Engine.

This module provides ChromaDB-backed historical event retrieval.
Given a SentimentResidual (current market signal), it queries a vector database
of historical SEC filings, news, and market events to find the top-k most similar
past scenarios. Results are returned as HistoricalMatch objects with confidence
scores, enabling the Judge to contextualize predictions against precedent.

Fits into the historian pillar: bridges live sentiment signals (from Linguist)
with historical patterns stored in ChromaDB, returning ranked matches for
Judge calibration and post-mortem analysis.
"""

import os
from dataclasses import dataclass
from typing import Optional
import json
import sqlite3
from pathlib import Path

import chromadb
from chromadb.config import Settings

# =============================================================================
# Data structures
# =============================================================================


@dataclass
class SentimentResidual:
    """A current market signal to be matched against history."""
    ticker: str
    headline: str
    sentiment_score: float  # -1.0 to 1.0
    source: str  # e.g. "news", "sec_filing", "reddit"
    timestamp: str  # ISO 8601
    context: Optional[str] = None  # Additional context (e.g. filing type)


@dataclass
class HistoricalMatch:
    """A historical event matched via RAG similarity."""
    ticker: str
    headline: str
    sentiment_score: float
    source: str
    timestamp: str
    similarity_score: float  # 0.0 to 1.0
    market_outcome: Optional[str] = None  # e.g. "+2.3%", "-1.5%"
    days_to_outcome: Optional[int] = None  # How many days until price moved
    context: Optional[str] = None


# =============================================================================
# ChromaDB initialization and persistence
# =============================================================================


def init_chromadb(db_path: str = "sentinel_chromadb") -> chromadb.Client:
    """Initialize a persistent ChromaDB client for historical event storage."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def get_or_create_collection(
    client: chromadb.Client, collection_name: str = "historical_events"
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for historical events."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


# =============================================================================
# Document ingestion (stub for spine integration)
# =============================================================================


def add_historical_event(
    collection: chromadb.Collection,
    ticker: str,
    headline: str,
    sentiment_score: float,
    source: str,
    timestamp: str,
    market_outcome: Optional[str] = None,
    days_to_outcome: Optional[int] = None,
    context: Optional[str] = None,
) -> str:
    """
    Add a historical event to the ChromaDB collection.
    Returns the document ID.
    """
    doc_id = f"{ticker}_{timestamp}_{source}"
    
    # Construct embedding text: headline + context for semantic search
    embedding_text = headline
    if context:
        embedding_text = f"{headline} {context}"
    
    # Metadata: store structured fields for filtering
    metadata = {
        "ticker": ticker,
        "source": source,
        "timestamp": timestamp,
        "sentiment_score": sentiment_score,
    }
    if market_outcome:
        metadata["market_outcome"] = market_outcome
    if days_to_outcome is not None:
        metadata["days_to_outcome"] = days_to_outcome
    
    collection.add(
        ids=[doc_id],
        documents=[embedding_text],
        metadatas=[metadata],
    )
    
    return doc_id


# =============================================================================
# RAG query interface (core)
# =============================================================================


def query_historical_matches(
    collection: chromadb.Collection,
    residual: SentimentResidual,
    top_k: int = 5,
    ticker_filter: bool = True,
) -> list[HistoricalMatch]:
    """
    Query ChromaDB for the top-k most similar historical events.
    
    Args:
        collection: ChromaDB collection of historical events
        residual: Current SentimentResidual to match
        top_k: Number of matches to return
        ticker_filter: If True, only return matches for the same ticker
    
    Returns:
        List of HistoricalMatch objects, ranked by similarity_score (descending)
    """
    # Build query text from residual
    query_text = residual.headline
    if residual.context:
        query_text = f"{residual.headline} {residual.context}"
    
    # Build where filter
    where_filter = None
    if ticker_filter:
        where_filter = {"ticker": {"$eq": residual.ticker}}
    
    # Query ChromaDB
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter,
        )
    except Exception as e:
        # Log and return empty list if collection is empty or query fails
        print(f"ChromaDB query failed: {e}")
        return []
    
    if not results or not results["ids"] or len(results["ids"]) == 0:
        return []
    
    # Build HistoricalMatch objects from results
    matches = []
    doc_ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    
    for doc_id, distance, metadata in zip(doc_ids, distances, metadatas):
        # ChromaDB returns distance; convert to similarity (cosine: 1 - distance)
        similarity_score = max(0.0, 1.0 - distance)
        
        match = HistoricalMatch(
            ticker=metadata.get("ticker", residual.ticker),
            headline=metadata.get("headline", ""),
            sentiment_score=metadata.get("sentiment_score", 0.0),
            source=metadata.get("source", "unknown"),
            timestamp=metadata.get("timestamp", ""),
            similarity_score=similarity_score,
            market_outcome=metadata.get("market_outcome"),
            days_to_outcome=metadata.get("days_to_outcome"),
            context=metadata.get("context"),
        )
        matches.append(match)
    
    return matches


# =============================================================================
# Batch import from local corpus (for testing and cold-start)
# =============================================================================


def import_historical_corpus(
    collection: chromadb.Collection,
    corpus_file: str,
) -> int:
    """
    Bulk-import historical events from a JSON corpus file.
    
    Expected format: list of dicts with keys:
      ticker, headline, sentiment_score, source, timestamp,
      [market_outcome], [days_to_outcome], [context]
    
    Returns the number of events imported.
    """
    if not os.path.exists(corpus_file):
        print(f"Corpus file not found: {corpus_file}")
        return 0
    
    try:
        with open(corpus_file, "r") as f:
            events = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse corpus JSON: {e}")
        return 0
    
    count = 0
    for event in events:
        try:
            add_historical_event(
                collection,
                ticker=event["ticker"],
                headline=event["headline"],
                sentiment_score=event["sentiment_score"],
                source=event["source"],
                timestamp=event["timestamp"],
                market_outcome=event.get("market_outcome"),
                days_to_outcome=event.get("days_to_outcome"),
                context=event.get("context"),
            )
            count += 1
        except Key
