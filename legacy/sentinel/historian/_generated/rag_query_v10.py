"""
RAG query interface for Sentinel Sentiment Engine.

This module queries ChromaDB for historical events similar to a given
SentimentResidual, returning ranked HistoricalMatch objects. It bridges
the Linguist's current sentiment analysis with the Historian's vector DB
of past market events, enabling confidence calibration and pattern detection.

Used by: sentinel/judge/predictor.py (per-ticker prediction context).
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
import chromadb
import numpy as np


@dataclass
class HistoricalMatch:
    """A past market event semantically similar to current sentiment."""
    event_id: str
    ticker: str
    date: str
    description: str
    sentiment_score: float
    price_movement_pct: float
    similarity_score: float
    confidence_notes: str


def init_chroma_client() -> chromadb.Client:
    """Initialize persistent ChromaDB client pointing to sentinel_db directory."""
    db_dir = os.path.join(os.path.dirname(__file__), "..", "..", "_data", "sentinel_db")
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)
    return client


def get_or_create_collection(client: chromadb.Client, collection_name: str = "sentiment_events") -> chromadb.Collection:
    """Get or create the sentiment_events collection in ChromaDB."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_historical_event(
    collection: chromadb.Collection,
    event_id: str,
    ticker: str,
    date: str,
    description: str,
    sentiment_score: float,
    price_movement_pct: float,
    embedding: Optional[list] = None
) -> None:
    """Ingest a historical event into ChromaDB with metadata."""
    metadata = {
        "ticker": ticker,
        "date": date,
        "sentiment_score": sentiment_score,
        "price_movement_pct": price_movement_pct,
    }
    
    if embedding is None:
        embedding = [0.0] * 1536
    
    collection.add(
        ids=[event_id],
        embeddings=[embedding],
        documents=[description],
        metadatas=[metadata]
    )


def query_historical_events(
    collection: chromadb.Collection,
    query_embedding: list,
    query_text: str,
    ticker: Optional[str] = None,
    top_k: int = 5
) -> list[HistoricalMatch]:
    """
    Query ChromaDB for top-k historical events similar to the given query.
    
    Args:
        collection: ChromaDB collection of historical sentiment events.
        query_embedding: Dense vector representation of current sentiment.
        query_text: Human-readable description of current sentiment state.
        ticker: Optional ticker filter; if None, searches all events.
        top_k: Number of top matches to return.
    
    Returns:
        List of HistoricalMatch objects ranked by similarity.
    """
    where_filter = None
    if ticker:
        where_filter = {"ticker": ticker}
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["embeddings", "documents", "metadatas", "distances"]
    )
    
    matches = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for i, event_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            
            match = HistoricalMatch(
                event_id=event_id,
                ticker=metadata.get("ticker", "UNKNOWN"),
                date=metadata.get("date", ""),
                description=results["documents"][0][i] if results["documents"] else "",
                sentiment_score=float(metadata.get("sentiment_score", 0.0)),
                price_movement_pct=float(metadata.get("price_movement_pct", 0.0)),
                similarity_score=similarity,
                confidence_notes=f"Vector similarity: {similarity:.3f}"
            )
            matches.append(match)
    
    return matches


def compute_embedding_from_text(text: str) -> list:
    """
    Placeholder: compute dense embedding for text.
    In production, use Gemini embeddings API or another service.
    Currently returns a dummy vector.
    """
    return [0.0] * 1536


def enrich_prediction_with_history(
    matches: list[HistoricalMatch],
    current_sentiment_score: float
) -> dict:
    """
    Synthesize historical matches into confidence adjustments for prediction.
    
    Args:
        matches: List of HistoricalMatch objects from RAG query.
        current_sentiment_score: Current sentiment score (e.g., -0.8 to +0.8).
    
    Returns:
        Dict with: avg_historical_move, confidence_adjustment, supporting_matches.
    """
    if not matches:
        return {
            "avg_historical_move": 0.0,
            "confidence_adjustment": 0.0,
            "supporting_matches": [],
            "notes": "No historical parallels found."
        }
    
    historical_moves = [m.price_movement_pct for m in matches]
    avg_move = np.mean(historical_moves) if historical_moves else 0.0
    std_move = np.std(historical_moves) if len(historical_moves) > 1 else 0.0
    
    avg_similarity = np.mean([m.similarity_score for m in matches])
    confidence_adjustment = avg_similarity * 0.2
    
    top_matches = [
        {
            "date": m.date,
            "ticker": m.ticker,
            "sentiment_score": m.sentiment_score,
            "price_movement_pct": m.price_movement_pct,
            "similarity": m.similarity_score
        }
        for m in matches[:3]
    ]
    
    return {
        "avg_historical_move": float(avg_move),
        "std_historical_move": float(std_move),
        "confidence_adjustment": float(confidence_adjustment),
        "supporting_matches": top_matches,
        "notes": f"Found {len(matches)} historical parallels; avg move: {avg_move:.2f}%, similarity: {avg_similarity:.3f}"
    }


if __name__ == "__main__":
    client = init_chroma_client()
    collection = get_or_create_collection(client)
    
    ingest_historical_event(
        collection,
        event_id="evt_001",
        ticker="AAPL",
        date="2024-01-15",
        description="Positive earnings beat with forward guidance raised",
        sentiment_score=0.75,
        price_movement_pct=3.5,
        embedding=compute_embedding_from_text("Positive earnings beat with forward guidance raised")
    )
    
    test_query = compute_embedding_from_text("Strong earnings outlook")
    matches = query_historical_events(collection, test_query, "Strong earnings outlook", ticker="AAPL", top_k=3)
    
    print(f"Found {len(matches)} historical matches:")
    for match in matches:
        print(f"  {match.event_id}: {match.date} {match.ticker} -> {match.price_movement_pct:+.2f}% (sim: {match.similarity_score:.3f})")
    
    enrichment = enrich_prediction_with_history(matches, current_sentiment_score=0.7)
    print(f"\nEnrichment: {json.dumps(enrichment, indent=2)}")
