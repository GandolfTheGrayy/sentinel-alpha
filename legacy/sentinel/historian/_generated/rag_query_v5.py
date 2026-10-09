"""
RAG query interface for Sentinel Sentiment Engine.

Given a current SentimentResidual (text embedding + metadata), queries ChromaDB
for the top-k most similar historical events and returns a HistoricalMatch list.
Used by judge/predictor.py to contextualize predictions with analogous past moves.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
import chromadb
import numpy as np
import google.generativeai as genai


@dataclass
class HistoricalMatch:
    """Represents a single historical event matched via RAG."""
    event_id: str
    ticker: str
    date: str
    event_type: str
    summary: str
    market_move_percent: float
    similarity_score: float
    embedding_distance: float


def _init_chroma_client() -> chromadb.Client:
    """Initialize ChromaDB client, creating persistent storage if needed."""
    db_path = os.path.expanduser("~/.sentinel/chroma_db")
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def _get_or_create_collection(
    client: chromadb.Client, collection_name: str = "historical_events"
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for historical events."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def _embed_text_with_gemini(text: str) -> list[float]:
    """Embed text using Gemini's embedding API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    
    result = genai.embed_content(
        model="models/embedding-001",
        content=text
    )
    return result["embedding"]


def query_historical_events(
    sentiment_text: str,
    ticker: str,
    k: int = 5,
    distance_threshold: float = 0.3
) -> list[HistoricalMatch]:
    """
    Query ChromaDB for top-k historical events similar to current sentiment signal.
    
    Args:
        sentiment_text: The current sentiment narrative or raw signal text.
        ticker: Stock ticker symbol to optionally filter results.
        k: Number of top matches to return.
        distance_threshold: Maximum embedding distance (0.0-1.0) to include result.
    
    Returns:
        List of HistoricalMatch objects ranked by similarity.
    """
    client = _init_chroma_client()
    collection = _get_or_create_collection(client)
    
    # Embed the query text
    query_embedding = _embed_text_with_gemini(sentiment_text)
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"ticker": ticker} if ticker else None
    )
    
    matches: list[HistoricalMatch] = []
    
    if results and results["ids"] and len(results["ids"]) > 0:
        for idx, event_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][idx] if results["distances"] else 0.0
            
            # Skip if distance exceeds threshold
            if distance > distance_threshold:
                continue
            
            # Extract metadata
            metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
            
            match = HistoricalMatch(
                event_id=event_id,
                ticker=metadata.get("ticker", "UNKNOWN"),
                date=metadata.get("date", ""),
                event_type=metadata.get("event_type", ""),
                summary=metadata.get("summary", ""),
                market_move_percent=float(metadata.get("market_move_percent", 0.0)),
                similarity_score=1.0 - distance,  # Convert distance to similarity
                embedding_distance=distance
            )
            matches.append(match)
    
    return matches


def ingest_historical_event(
    event_id: str,
    ticker: str,
    date: str,
    event_type: str,
    summary: str,
    market_move_percent: float
) -> None:
    """
    Ingest a historical event into the RAG corpus.
    
    Args:
        event_id: Unique identifier for the event.
        ticker: Stock ticker symbol.
        date: Event date (ISO format recommended).
        event_type: Category (e.g., "earnings_miss", "sec_filing", "reddit_surge").
        summary: Text narrative of the event.
        market_move_percent: Actual market move (%) following the event.
    """
    client = _init_chroma_client()
    collection = _get_or_create_collection(client)
    
    # Embed the summary
    embedding = _embed_text_with_gemini(summary)
    
    # Add to collection
    collection.add(
        ids=[event_id],
        embeddings=[embedding],
        documents=[summary],
        metadatas=[{
            "ticker": ticker,
            "date": date,
            "event_type": event_type,
            "summary": summary,
            "market_move_percent": market_move_percent
        }]
    )


def batch_ingest_historical_events(events: list[dict]) -> None:
    """
    Batch ingest multiple historical events.
    
    Args:
        events: List of dicts with keys: event_id, ticker, date, event_type, summary, market_move_percent.
    """
    for event in events:
        ingest_historical_event(
            event_id=event["event_id"],
            ticker=event["ticker"],
            date=event["date"],
            event_type=event["event_type"],
            summary=event["summary"],
            market_move_percent=event["market_move_percent"]
        )


def clear_collection(collection_name: str = "historical_events") -> None:
    """Clear all events from a ChromaDB collection."""
    client = _init_chroma_client()
    client.delete_collection(name=collection_name)
    _get_or_create_collection(client, collection_name)


if __name__ == "__main__":
    # Example usage
    print("RAG Query Interface for Sentinel Sentiment Engine")
    print("Designed to be imported and called by judge/predictor.py")
