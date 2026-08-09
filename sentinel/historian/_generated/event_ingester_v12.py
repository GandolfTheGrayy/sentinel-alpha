"""
Historical market event ingestion pipeline for Sentinel.

This module reads past market events from a CSV file, generates embeddings
using Gemini's embedding API, and stores them in ChromaDB for RAG retrieval.
Events are indexed by company ticker, event type, and temporal proximity to
enable historical context injection during prediction synthesis.

Fits into Sentinel.historian as the corpus-building foundation for rag_query.py.
"""

import csv
import os
import json
from typing import Optional
import sqlite3
from datetime import datetime

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import pandas as pd


def _init_chroma_client(db_path: str = "sentinel_events.db") -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def _get_embedding(text: str, api_key: Optional[str] = None) -> list[float]:
    """Fetch embedding vector from Gemini API for a given text."""
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    
    genai.configure(api_key=api_key)
    model = "embedding-001"
    
    result = genai.embed_content(
        model=model,
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return result["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    collection_name: str = "market_events",
    api_key: Optional[str] = None
) -> chromadb.Collection:
    """
    Read historical events from CSV and embed into ChromaDB.
    
    Expected CSV columns: date, ticker, event_type, description, impact_direction
    (impact_direction: 'bullish', 'bearish', 'neutral')
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    client = _init_chroma_client()
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Historical market events with embeddings"}
    )
    
    # Read CSV
    df = pd.read_csv(csv_path)
    required_cols = {"date", "ticker", "event_type", "description", "impact_direction"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Embed and ingest
    for idx, row in df.iterrows():
        event_text = (
            f"Date: {row['date']}. "
            f"Ticker: {row['ticker']}. "
            f"Event: {row['event_type']}. "
            f"Description: {row['description']}. "
            f"Impact: {row['impact_direction']}"
        )
        
        embedding = _get_embedding(event_text, api_key=api_key)
        
        metadata = {
            "date": row["date"],
            "ticker": row["ticker"],
            "event_type": row["event_type"],
            "impact_direction": row["impact_direction"],
        }
        
        doc_id = f"{row['ticker']}_{row['date']}_{idx}"
        
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[event_text],
            metadatas=[metadata]
        )
    
    print(f"Ingested {len(df)} events into ChromaDB collection '{collection_name}'")
    return collection


def query_historical_events(
    query_text: str,
    ticker: Optional[str] = None,
    n_results: int = 5,
    collection_name: str = "market_events",
    api_key: Optional[str] = None
) -> list[dict]:
    """
    Retrieve historical events similar to query via RAG.
    
    Returns list of dicts: {id, document, metadata, distance}
    """
    client = _init_chroma_client()
    collection = client.get_collection(name=collection_name)
    
    query_embedding = _get_embedding(query_text, api_key=api_key)
    
    where_filter = None
    if ticker:
        where_filter = {"ticker": {"$eq": ticker}}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    
    # Flatten and restructure results
    output = []
    if results["ids"] and len(results["ids"]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
    
    return output


def clear_collection(collection_name: str = "market_events") -> None:
    """Delete all events from a ChromaDB collection."""
    client = _init_chroma_client()
    client.delete_collection(name=collection_name)
    print(f"Cleared collection '{collection_name}'")


if __name__ == "__main__":
    # Example usage
    sample_csv = "sample_events.csv"
    
    # Create a minimal sample CSV if it doesn't exist
    if not os.path.exists(sample_csv):
        sample_data = {
            "date": ["2023-01-15", "2023-02-20", "2023-03-10"],
            "ticker": ["AAPL", "AAPL", "MSFT"],
            "event_type": ["earnings_beat", "product_launch", "acquisition"],
            "description": [
                "Q4 earnings beat expectations by 15%",
                "New iPhone model announced",
                "Acquires AI startup for $2B"
            ],
            "impact_direction": ["bullish", "bullish", "neutral"]
        }
        pd.DataFrame(sample_data).to_csv(sample_csv, index=False)
        print(f"Created sample CSV: {sample_csv}")
    
    # Ingest events
    print("Ingesting events from CSV...")
    collection = ingest_events_from_csv(sample_csv)
    
    # Query events
    print("\nQuerying for AAPL product announcements...")
    results = query_historical_events(
        "iPhone product launch announcement",
        ticker="AAPL",
        n_results=3
    )
    for result in results:
        print(f"  {result['metadata']['date']}: {result['document'][:100]}...")
