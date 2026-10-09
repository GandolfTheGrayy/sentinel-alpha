"""
RAG query interface for Sentinel Historian pillar.

Given a current SentimentResidual (sentiment analysis + metadata), queries ChromaDB
for the top-k most similar historical events and returns a HistoricalMatch list.
This module bridges live sentiment signals (from Linguist) with historical precedent,
enabling Judge to contextualize predictions with past market behavior under similar
conditions.

Integration:
  - Linguist produces SentimentResidual objects (ticker, text, scores, timestamp).
  - Historian.rag_query receives SentimentResidual, embeds it via Gemini,
    queries ChromaDB for semantically similar filings/news events,
    returns ranked HistoricalMatch objects with confidence scores.
  - Judge.predictor consumes HistoricalMatch list to calibrate daily forecasts.
"""

import os
import json
import sqlite3
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# Initialize Gemini for embedding only (not text generation in this module).
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


@dataclass
class SentimentResidual:
    """Input: live sentiment signal from Linguist pillar."""
    ticker: str
    text: str  # The source text analyzed (headline, filing excerpt, social post, etc.)
    certainty_score: float  # 0.0–1.0, higher = more confident direction
    sentiment_score: float  # -1.0 to +1.0, negative=bearish, positive=bullish
    residual_type: str  # "news", "sec_filing", "social", "earnings_alert", etc.
    timestamp: str  # ISO 8601 timestamp


@dataclass
class HistoricalMatch:
    """Output: ranked historical event matching current sentiment."""
    ticker: str
    matched_event_text: str  # Original text from the matched historical event
    matched_event_type: str  # Type of event: "news", "sec_filing", "social", etc.
    matched_event_date: str  # ISO 8601 date when the historical event occurred
    similarity_score: float  # 0.0–1.0, cosine distance to current residual
    historical_outcome: Optional[str]  # If available: "price_up", "price_down", "neutral", or raw price change %
    confidence: float  # 0.0–1.0, confidence in the match and outcome relevance


def _get_or_create_chroma_client() -> chromadb.Client:
    """
    Initialize ChromaDB client with persistent storage.
    """
    chroma_dir = os.environ.get("CHROMA_DB_PATH", "./sentinel_chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=chroma_dir,
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def _embed_text_via_gemini(text: str) -> List[float]:
    """
    Embed text using Gemini embedding model.
    Returns a list of floats (embedding vector).
    """
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="semantic_similarity",
        )
        return response["embedding"]
    except Exception as e:
        print(f"Gemini embedding error: {e}")
        raise


def _load_outcome_from_db(event_id: str) -> Optional[str]:
    """
    Query local SQLite DB for historical price outcome associated with event_id.
    Returns outcome string like "price_up_3.5%" or None if not found.
    """
    db_path = os.environ.get("SENTINEL_DB_PATH", "./sentinel.db")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT outcome FROM historical_outcomes WHERE event_id = ?", (event_id,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        print(f"Outcome lookup error: {e}")
        return None


def query_historical_events(
    residual: SentimentResidual,
    top_k: int = 5,
    min_similarity: float = 0.5,
) -> List[HistoricalMatch]:
    """
    Query ChromaDB for top-k historical events matching the sentiment residual.
    
    Args:
        residual: Current SentimentResidual from Linguist.
        top_k: Number of historical matches to return (default 5).
        min_similarity: Minimum cosine similarity score to include (default 0.5).
    
    Returns:
        List of HistoricalMatch objects sorted by similarity (descending).
    """
    try:
        # Embed the current residual text.
        embedding = _embed_text_via_gemini(residual.text)
        
        # Get ChromaDB client.
        client = _get_or_create_chroma_client()
        
        # Query the collection for this ticker.
        collection_name = f"ticker_{residual.ticker.lower()}"
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist yet; return empty list.
            return []
        
        # Query with embedding.
        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=None,  # Could filter by event type here if needed.
        )
        
        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                similarity = float(results["distances"][0][i])
                # ChromaDB returns Euclidean distance; convert to similarity.
                cosine_similarity = 1.0 / (1.0 + similarity)
                
                if cosine_similarity < min_similarity:
                    continue
                
                # Extract metadata and text.
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                text = results["documents"][0][i] if results["documents"] else ""
                
                # Load historical outcome from DB.
                outcome = _load_outcome_from_db(event_id)
                
                # Compute confidence as product of similarity and outcome availability.
                confidence = cosine_similarity * (0.9 if outcome else 0.6)
                
                match = HistoricalMatch(
                    ticker=residual.ticker,
                    matched_event_text=text,
                    matched_event_type=metadata.get("event_type", "unknown"),
                    matched_event_date=metadata.get("event_date", "unknown"),
                    similarity_score=cosine_similarity,
                    historical_outcome=outcome,
                    confidence=confidence,
                )
                matches.append(match)
        
        # Sort by similarity descending.
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
    
    except Exception as e:
        print(f"RAG query error for {residual.ticker}: {e}")
        return []


def ingest_historical_event(
    ticker: str,
    text: str,
    event_type: str,
    event_date: str,
    event_id: Optional[str] = None,
) -> str:
    """
    Ingest a single historical event into ChromaDB for future RAG queries.
    
    Args:
        ticker: Stock ticker symbol.
        text: Full text of the event (news headline, SEC filing excerpt, etc.).
        event_type: Category of event ("news", "sec_filing", "social",
