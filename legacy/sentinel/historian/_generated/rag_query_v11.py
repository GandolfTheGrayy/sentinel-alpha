"""
RAG query interface for Sentinel Historian pillar.

Given a SentimentResidual (current sentiment analysis), queries ChromaDB
for the top-k most similar historical events and returns ranked HistoricalMatch
results. Integrates with Gemini embeddings for semantic similarity lookup.
This module bridges live sentiment signals with historical precedent.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.config import Settings
import google.generativeai as genai


@dataclass
class SentimentResidual:
    """Current sentiment analysis snapshot for a single ticker."""
    ticker: str
    headline: str
    sentiment_score: float
    certainty: float
    source: str
    timestamp: str


@dataclass
class HistoricalMatch:
    """A single historical event matched via RAG to current sentiment."""
    ticker: str
    event_date: str
    event_headline: str
    event_sentiment: float
    similarity_score: float
    outcome_direction: str
    outcome_magnitude: float
    days_to_move: int
    metadata: dict


def _init_chromadb_client(persist_dir: str = "./sentinel_chroma") -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def _get_embedding_for_text(text: str) -> list[float]:
    """
    Generate embedding vector for text using Gemini API.
    
    Uses the Gemini embedding model (text-embedding-004 or equivalent)
    to convert text into a semantic vector.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
    )
    return response["embedding"]


def init_rag_collection(
    collection_name: str = "sentinel_events",
    persist_dir: str = "./sentinel_chroma",
) -> chromadb.Collection:
    """
    Initialize or retrieve a ChromaDB collection for historical events.
    """
    client = _init_chromadb_client(persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Historical market events indexed for Sentinel RAG"},
    )
    return collection


def ingest_historical_event(
    collection: chromadb.Collection,
    ticker: str,
    event_date: str,
    headline: str,
    sentiment_score: float,
    outcome_direction: str,
    outcome_magnitude: float,
    days_to_move: int,
    metadata: Optional[dict] = None,
) -> None:
    """
    Ingest a single historical event into the RAG collection.
    
    Embeds the headline and stores all metadata for later retrieval.
    """
    if metadata is None:
        metadata = {}
    
    embedding = _get_embedding_for_text(headline)
    
    doc_id = f"{ticker}_{event_date}_{len(collection.get()['ids'])}"
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[headline],
        metadatas=[{
            "ticker": ticker,
            "event_date": event_date,
            "sentiment_score": sentiment_score,
            "outcome_direction": outcome_direction,
            "outcome_magnitude": outcome_magnitude,
            "days_to_move": days_to_move,
            **metadata,
        }],
    )


def query_rag(
    collection: chromadb.Collection,
    residual: SentimentResidual,
    top_k: int = 5,
) -> list[HistoricalMatch]:
    """
    Query ChromaDB for top-k historical events similar to current sentiment.
    
    Returns ranked HistoricalMatch objects ordered by similarity score.
    """
    query_embedding = _get_embedding_for_text(residual.headline)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"ticker": residual.ticker} if residual.ticker else None,
    )
    
    matches = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = 1.0 / (1.0 + distance)
            
            match = HistoricalMatch(
                ticker=meta.get("ticker", residual.ticker),
                event_date=meta.get("event_date", ""),
                event_headline=results["documents"][0][i] if results["documents"] else "",
                event_sentiment=float(meta.get("sentiment_score", 0.0)),
                similarity_score=similarity,
                outcome_direction=meta.get("outcome_direction", "UNKNOWN"),
                outcome_magnitude=float(meta.get("outcome_magnitude", 0.0)),
                days_to_move=int(meta.get("days_to_move", 0)),
                metadata=meta,
            )
            matches.append(match)
    
    return sorted(matches, key=lambda x: x.similarity_score, reverse=True)


def query_rag_batch(
    collection: chromadb.Collection,
    residuals: list[SentimentResidual],
    top_k: int = 5,
) -> dict[str, list[HistoricalMatch]]:
    """
    Batch-query RAG for multiple SentimentResiduals.
    
    Returns a dict mapping ticker to list of HistoricalMatch results.
    """
    results = {}
    for residual in residuals:
        results[residual.ticker] = query_rag(collection, residual, top_k)
    return results


def export_collection_stats(collection: chromadb.Collection) -> dict:
    """
    Export summary statistics about the RAG collection.
    
    Returns document count, ticker distribution, and date range.
    """
    all_data = collection.get()
    doc_count = len(all_data.get("ids", []))
    
    tickers = {}
    dates = []
    
    if all_data.get("metadatas"):
        for meta in all_data["metadatas"]:
            ticker = meta.get("ticker", "UNKNOWN")
            tickers[ticker] = tickers.get(ticker, 0) + 1
            if "event_date" in meta:
                dates.append(meta["event_date"])
    
    return {
        "total_documents": doc_count,
        "ticker_distribution": tickers,
        "date_range": (min(dates) if dates else None, max(dates) if dates else None),
    }
