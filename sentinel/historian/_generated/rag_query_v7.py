"""
Sentinel Historian RAG Query Interface.

This module provides the core RAG (Retrieval-Augmented Generation) pipeline for
Sentinel. Given a SentimentResidual (current market signal), it queries ChromaDB
for the top-k most similar historical events, scores them by relevance, and
returns a ranked HistoricalMatch list. Used by Judge to contextualize predictions
with historical precedent and calibrate confidence scores.

Role in Sentinel:
  - Bridges Scout (live signals) and Judge (predictions).
  - Embeds current sentiment into ChromaDB vector space.
  - Retrieves semantically similar past events with metadata.
  - Weights matches by temporal proximity, outcome magnitude, and text similarity.
  - Enables Judge to say "this looks like the 2019 short-squeeze event with 73% confidence."
"""

import os
import json
import sqlite3
from dataclasses import dataclass
from typing import Optional
import chromadb
import numpy as np
import google.generativeai as genai


@dataclass
class HistoricalMatch:
    """Ranked historical event retrieved from ChromaDB."""
    event_id: str
    ticker: str
    date: str
    description: str
    embedding_similarity: float
    temporal_recency_score: float
    outcome_magnitude: float
    composite_relevance: float
    metadata: dict


def init_chromadb(db_path: str = "sentinel.db") -> chromadb.Client:
    """Initialize or connect to ChromaDB client."""
    client = chromadb.Client(
        settings=chromadb.config.Settings(
            chroma_db_impl="duckdb",
            persist_directory=os.path.dirname(db_path) or ".",
            anonymized_telemetry=False
        )
    )
    return client


def get_or_create_collection(client: chromadb.Client, name: str = "sentinel_events") -> chromadb.Collection:
    """Get or create ChromaDB collection for historical events."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def embed_text_gemini(text: str) -> list[float]:
    """Embed text using Gemini embedding model."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return response["embedding"]


def ingest_historical_event(
    collection: chromadb.Collection,
    event_id: str,
    ticker: str,
    date: str,
    description: str,
    metadata: dict
) -> None:
    """Ingest a historical event into ChromaDB with embedding."""
    embedding = embed_text_gemini(description)
    collection.add(
        ids=[event_id],
        embeddings=[embedding],
        metadatas=[{
            "ticker": ticker,
            "date": date,
            **metadata
        }],
        documents=[description]
    )


def query_rag(
    collection: chromadb.Collection,
    query_text: str,
    ticker: str,
    k: int = 5,
    max_days_ago: Optional[int] = None
) -> list[HistoricalMatch]:
    """
    Query ChromaDB for top-k historical matches to current sentiment signal.
    
    Args:
        collection: ChromaDB collection instance.
        query_text: Current sentiment text to embed and match.
        ticker: Target ticker symbol for filtering.
        k: Number of top matches to return.
        max_days_ago: Optional cutoff; ignore events older than this (days).
    
    Returns:
        List of HistoricalMatch objects ranked by composite relevance.
    """
    query_embedding = embed_text_gemсин(query_text)
    
    where_clause = {"ticker": {"$eq": ticker}} if ticker else None
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause
    )
    
    matches = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for i, event_id in enumerate(results["ids"][0]):
            similarity = results["distances"][0][i] if results["distances"] else 0.0
            similarity_score = 1.0 / (1.0 + similarity)
            
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            description = results["documents"][0][i] if results["documents"] else ""
            
            event_date = metadata.get("date", "")
            outcome_magnitude = float(metadata.get("outcome_magnitude", 0.5))
            temporal_score = _compute_temporal_score(event_date, max_days_ago)
            
            composite = (
                0.5 * similarity_score +
                0.3 * temporal_score +
                0.2 * (outcome_magnitude / 100.0 if outcome_magnitude > 0 else 0.0)
            )
            
            match = HistoricalMatch(
                event_id=event_id,
                ticker=metadata.get("ticker", ticker),
                date=event_date,
                description=description,
                embedding_similarity=similarity_score,
                temporal_recency_score=temporal_score,
                outcome_magnitude=outcome_magnitude,
                composite_relevance=composite,
                metadata=metadata
            )
            matches.append(match)
    
    matches.sort(key=lambda m: m.composite_relevance, reverse=True)
    return matches


def _compute_temporal_score(event_date: str, max_days_ago: Optional[int]) -> float:
    """
    Compute recency score: recent events score higher (0.0–1.0).
    
    Assumes event_date is ISO format (YYYY-MM-DD).
    """
    from datetime import datetime, timedelta
    
    try:
        event = datetime.fromisoformat(event_date)
        today = datetime.now()
        days_diff = (today - event).days
        
        if max_days_ago and days_diff > max_days_ago:
            return 0.0
        
        max_lookback = 365 * 5
        if days_diff < 0:
            return 0.5
        
        score = max(0.0, 1.0 - (days_diff / max_lookback))
        return score
    except (ValueError, TypeError):
        return 0.5


def batch_ingest_events(
    collection: chromadb.Collection,
    events: list[dict]
) -> None:
    """
    Bulk ingest historical events from list of dicts.
    
    Each dict must have: event_id, ticker, date, description, metadata.
    """
    for event in events:
        ingest_historical_event(
            collection=collection,
            event_id=event["event_id"],
            ticker=event["ticker"],
            date=event["date"],
            description=event["description"],
            metadata=event.get("metadata", {})
        )


def load_events_from_sqlite(db_path: str, table: str = "historical_events") -> list[dict]:
    """Load historical events from SQLite for bulk ingestion."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    rows = cursor.fetchall()
    conn.close()
    
    events = []
    for row in rows:
        events.append(dict(row))
    return events


if __name__ == "__main__":
    client = init_chromadb()
    collection = get_or_create_collection(client)
    
    sample_query = "Tesla stock rose 15% after Elon announced record production"
    matches = query_rag(collection, sample_query, ticker
