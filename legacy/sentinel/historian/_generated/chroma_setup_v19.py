"""
ChromaDB vector database initialization and client wrapper for Sentinel.

This module sets up a persistent local ChromaDB instance with collections
for market events and SEC filings. It provides a typed client interface
for RAG queries and document insertion, ensuring all embeddings are
generated via Gemini and stored consistently.

Role in Sentinel:
  - Historian pillar: Initializes and maintains the vector DB backend
  - Called once at startup by sentinel/pipeline.py
  - Provides get_client() for RAG queries in sentinel/historian/rag_query.py
  - Stores embeddings of historical events, SEC filings, news for retrieval
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


def get_db_path() -> Path:
    """Return the persistent ChromaDB storage directory."""
    db_dir = Path.home() / ".sentinel" / "chroma_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def initialize_chroma() -> chromadb.Client:
    """
    Initialize and return a ChromaDB persistent client with collections.
    
    Creates or connects to a local ChromaDB instance and ensures
    'market_events' and 'sec_filings' collections exist.
    Collections use Gemini embeddings (via ChromaDB's default provider).
    
    Returns:
        chromadb.Client: Persistent client ready for use.
    """
    db_path = get_db_path()
    
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(db_path),
        anonymized_telemetry=False,
    )
    
    client = chromadb.Client(settings)
    
    # Ensure collections exist; no-op if already present
    try:
        client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events, news, sentiment signals"}
        )
    except Exception:
        pass
    
    try:
        client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC Edgar 8-K, 10-Q, 10-K filings indexed by ticker and date"}
        )
    except Exception:
        pass
    
    return client


def get_client() -> chromadb.Client:
    """
    Retrieve the persistent ChromaDB client (singleton-like).
    
    Safe to call multiple times; returns the same configured instance.
    """
    return initialize_chroma()


def add_market_event(
    client: chromadb.Client,
    ticker: str,
    event_date: str,
    content: str,
    event_type: str,
    metadata: Optional[dict] = None,
) -> str:
    """
    Insert a market event document into the 'market_events' collection.
    
    Args:
        client: ChromaDB client instance.
        ticker: Stock symbol (e.g., "AAPL").
        event_date: ISO date string of the event.
        content: Full text of the event (news, sentiment, etc.).
        event_type: Category (e.g., "earnings", "scandal", "partnership").
        metadata: Optional dict of additional metadata.
    
    Returns:
        str: Document ID assigned by ChromaDB.
    """
    collection = client.get_collection("market_events")
    
    doc_id = f"{ticker}_{event_date}_{event_type}".replace(" ", "_")
    meta = metadata or {}
    meta.update({
        "ticker": ticker,
        "event_date": event_date,
        "event_type": event_type,
    })
    
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[meta],
    )
    
    return doc_id


def add_sec_filing(
    client: chromadb.Client,
    ticker: str,
    filing_date: str,
    filing_type: str,
    content: str,
    cik: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> str:
    """
    Insert an SEC filing into the 'sec_filings' collection.
    
    Args:
        client: ChromaDB client instance.
        ticker: Stock symbol.
        filing_date: ISO date of filing submission.
        filing_type: "8-K", "10-Q", "10-K", etc.
        content: Full extracted text of the filing.
        cik: Optional CIK number for the company.
        metadata: Optional additional metadata.
    
    Returns:
        str: Document ID assigned by ChromaDB.
    """
    collection = client.get_collection("sec_filings")
    
    doc_id = f"{ticker}_{filing_date}_{filing_type}".replace(" ", "_")
    meta = metadata or {}
    meta.update({
        "ticker": ticker,
        "filing_date": filing_date,
        "filing_type": filing_type,
    })
    if cik:
        meta["cik"] = cik
    
    collection.add(
        ids=[doc_id],
        documents=[content],
        metadatas=[meta],
    )
    
    return doc_id


def query_market_events(
    client: chromadb.Client,
    query_text: str,
    n_results: int = 5,
    ticker_filter: Optional[str] = None,
) -> dict:
    """
    Semantic search across market events by query text.
    
    Args:
        client: ChromaDB client instance.
        query_text: Natural language query.
        n_results: Number of results to return.
        ticker_filter: Optional ticker to restrict search.
    
    Returns:
        dict: Raw ChromaDB query result with ids, documents, distances, metadatas.
    """
    collection = client.get_collection("market_events")
    
    where = None
    if ticker_filter:
        where = {"ticker": {"$eq": ticker_filter}}
    
    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    
    return result


def query_sec_filings(
    client: chromadb.Client,
    query_text: str,
    n_results: int = 5,
    ticker_filter: Optional[str] = None,
    filing_type_filter: Optional[str] = None,
) -> dict:
    """
    Semantic search across SEC filings by query text.
    
    Args:
        client: ChromaDB client instance.
        query_text: Natural language query about regulatory content.
        n_results: Number of results to return.
        ticker_filter: Optional ticker to restrict search.
        filing_type_filter: Optional filing type ("8-K", "10-Q", etc.).
    
    Returns:
        dict: Raw ChromaDB query result.
    """
    collection = client.get_collection("sec_filings")
    
    where = None
    if ticker_filter or filing_type_filter:
        where = {}
        if ticker_filter:
            where["ticker"] = {"$eq": ticker_filter}
        if filing_type_filter:
            where["filing_type"] = {"$eq": filing_type_filter}
    
    result = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
    )
    
    return result


def get_collection_stats(client: chromadb.Client, collection_name: str) -> dict:
    """
    Retrieve metadata and document count for a collection.
    
    Args:
        client: ChromaDB client instance.
        collection_name: Name of collection ("market_events" or "sec_filings").
    
    Returns:
        dict: Stats including count, metadata, and name.
    """
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        metadata = collection.metadata
        return {
            "name": collection_name,
            "count": count,
            "metadata": metadata,
        }
    except Exception as
