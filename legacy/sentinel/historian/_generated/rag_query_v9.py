"""
Sentinel Historian RAG Query Interface.

This module provides the RAG (Retrieval-Augmented Generation) query layer for
Sentinel. Given a current SentimentResidual (ticker, sentiment score, context),
it queries ChromaDB for the top-k most similar historical events and returns
ranked HistoricalMatch results with confidence scores. These matches inform
the Judge's final price movement prediction by grounding current sentiment
signals in historical precedent.

Uses Gemini embeddings for semantic similarity (via google-generativeai SDK)
and ChromaDB for vector storage and retrieval.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional
import json

import chromadb
import google.generativeai as genai
import pandas as pd


@dataclass
class SentimentResidual:
    """Represents a current sentiment signal for a ticker."""
    ticker: str
    sentiment_score: float
    context: str
    timestamp: str


@dataclass
class HistoricalMatch:
    """Represents a matched historical event from RAG retrieval."""
    event_id: str
    ticker: str
    date: str
    event_description: str
    historical_sentiment_score: float
    actual_price_move_pct: float
    similarity_score: float
    confidence: float


class RAGQueryEngine:
    """ChromaDB-backed RAG engine for historical event retrieval."""

    def __init__(
        self,
        db_path: str = "./sentinel_chroma.db",
        collection_name: str = "historical_events"
    ):
        """
        Initialize RAG query engine with ChromaDB backend.

        Args:
            db_path: Path to ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection to query.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    def _embed_text(self, text: str) -> list[float]:
        """
        Embed text using Gemini embedding model.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = genai.embed_content(
            model="models/embedding-001",
            content=text
        )
        return response["embedding"]

    def ingest_historical_event(
        self,
        event_id: str,
        ticker: str,
        date: str,
        description: str,
        sentiment_score: float,
        actual_price_move_pct: float
    ) -> None:
        """
        Ingest a historical event into the RAG corpus.

        Args:
            event_id: Unique event identifier.
            ticker: Stock ticker symbol.
            date: Date of the event (ISO format).
            description: Natural language description of the event.
            sentiment_score: Sentiment score at the time (-1.0 to 1.0).
            actual_price_move_pct: Actual subsequent price move (%).
        """
        embedding = self._embed_text(description)
        metadata = {
            "ticker": ticker,
            "date": date,
            "sentiment_score": str(sentiment_score),
            "actual_price_move_pct": str(actual_price_move_pct)
        }
        self.collection.add(
            ids=[event_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[description]
        )

    def query(
        self,
        residual: SentimentResidual,
        k: int = 5,
        ticker_filter: bool = True
    ) -> list[HistoricalMatch]:
        """
        Query ChromaDB for top-k similar historical events.

        Args:
            residual: Current SentimentResidual to match against history.
            k: Number of top matches to return.
            ticker_filter: If True, prefer matches from the same ticker.

        Returns:
            List of HistoricalMatch objects ranked by similarity.
        """
        query_text = f"{residual.ticker}: {residual.context}"
        query_embedding = self._embed_text(query_text)

        where_filter = None
        if ticker_filter:
            where_filter = {"ticker": {"$eq": residual.ticker}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["embeddings", "metadatas", "documents", "distances"]
        )

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                confidence = self._compute_confidence(
                    similarity,
                    float(metadata.get("sentiment_score", 0.0)),
                    residual.sentiment_score
                )

                match = HistoricalMatch(
                    event_id=event_id,
                    ticker=metadata.get("ticker", ""),
                    date=metadata.get("date", ""),
                    event_description=results["documents"][0][i],
                    historical_sentiment_score=float(
                        metadata.get("sentiment_score", 0.0)
                    ),
                    actual_price_move_pct=float(
                        metadata.get("actual_price_move_pct", 0.0)
                    ),
                    similarity_score=similarity,
                    confidence=confidence
                )
                matches.append(match)

        return matches

    def _compute_confidence(
        self,
        similarity: float,
        historical_sentiment: float,
        current_sentiment: float
    ) -> float:
        """
        Compute confidence score for a match.

        Blends embedding similarity with sentiment alignment.

        Args:
            similarity: Cosine similarity (0 to 1).
            historical_sentiment: Sentiment score from historical event.
            current_sentiment: Current sentiment score.

        Returns:
            Confidence score (0 to 1).
        """
        sentiment_alignment = 1.0 - abs(
            historical_sentiment - current_sentiment
        ) / 2.0
        confidence = 0.6 * similarity + 0.4 * sentiment_alignment
        return max(0.0, min(1.0, confidence))

    def export_matches_to_dataframe(
        self,
        matches: list[HistoricalMatch]
    ) -> pd.DataFrame:
        """
        Convert HistoricalMatch list to pandas DataFrame for analysis.

        Args:
            matches: List of HistoricalMatch objects.

        Returns:
            DataFrame with columns for all match attributes.
        """
        data = [
            {
                "event_id": m.event_id,
                "ticker": m.ticker,
                "date": m.date,
                "event_description": m.event_description,
                "historical_sentiment": m.historical_sentiment_score,
                "actual_move_pct": m.actual_price_move_pct,
                "similarity": m.similarity_score,
                "confidence": m.confidence
            }
            for m in matches
        ]
        return pd.DataFrame(data)


def bootstrap_historical_corpus(
    chroma_db_path: str = "./sentinel_chroma.db"
) -> None:
    """
    Populate ChromaDB with synthetic historical events for testing.

    Args:
        chroma_db_path: Path to ChromaDB storage.
    """
    engine = RAGQueryEngine(db_path=chroma_db_path)

    sample_events = [
        {
            "event_id": "evt_001",
            "ticker": "NVDA",
            "date": "2023-05-18",
            "description": "Strong earnings beat, raised guidance, AI
