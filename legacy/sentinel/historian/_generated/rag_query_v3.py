"""
RAG Query Interface for Sentinel Historian.

This module queries ChromaDB for historical events matching a given SentimentResidual.
It retrieves the top-k most similar past market movements, regulatory filings, and
sentiment shifts to contextualize current predictions. Used by Judge to calibrate
confidence scores against historical precedent.

Integrates with ChromaDB vector store (initialized elsewhere) and returns structured
HistoricalMatch objects for downstream consumption by predictor.py.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
import chromadb
from chromadb.config import Settings


@dataclass
class SentimentResidual:
    """Container for current sentiment signal to query against history."""
    ticker: str
    headline: str
    sentiment_score: float
    source: str
    timestamp: str


@dataclass
class HistoricalMatch:
    """Single historical event matched by RAG query."""
    event_id: str
    ticker: str
    similarity_score: float
    historical_headline: str
    historical_sentiment: float
    historical_outcome: Optional[str]
    days_later_price_change: Optional[float]
    metadata: dict


class HistorianRAGQuery:
    """
    RAG interface for querying ChromaDB historical event embeddings.
    Retrieves top-k similar past events to contextualize current market moves.
    """

    def __init__(self, db_path: str = "./data/chroma_db"):
        """
        Initialize ChromaDB client and collection.

        Args:
            db_path: Path to persistent ChromaDB directory.
        """
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Retrieve or create the 'sentinel_events' collection in ChromaDB."""
        try:
            return self.client.get_collection(name="sentinel_events")
        except Exception:
            return self.client.create_collection(
                name="sentinel_events",
                metadata={"hnsw:space": "cosine"},
            )

    def query(
        self,
        residual: SentimentResidual,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical events similar to current residual.

        Args:
            residual: Current SentimentResidual (ticker, headline, sentiment).
            top_k: Number of historical matches to retrieve.
            min_similarity: Minimum cosine similarity threshold (0–1).

        Returns:
            List of HistoricalMatch objects ranked by similarity.
        """
        query_text = f"{residual.ticker} {residual.headline}"

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"ticker": residual.ticker} if residual.ticker else None,
        )

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = 1.0 - distance
                if similarity >= min_similarity:
                    metadata = (
                        results["metadatas"][0][i]
                        if results["metadatas"]
                        else {}
                    )
                    match = HistoricalMatch(
                        event_id=event_id,
                        ticker=metadata.get("ticker", "UNKNOWN"),
                        similarity_score=similarity,
                        historical_headline=metadata.get(
                            "headline", ""
                        ),
                        historical_sentiment=float(
                            metadata.get("sentiment_score", 0.0)
                        ),
                        historical_outcome=metadata.get("outcome"),
                        days_later_price_change=float(
                            metadata.get("days_later_price_change", 0.0)
                        )
                        if metadata.get("days_later_price_change")
                        else None,
                        metadata=metadata,
                    )
                    matches.append(match)

        return sorted(
            matches, key=lambda m: m.similarity_score, reverse=True
        )

    def ingest_event(
        self,
        event_id: str,
        ticker: str,
        headline: str,
        sentiment_score: float,
        outcome: Optional[str] = None,
        days_later_price_change: Optional[float] = None,
        extra_metadata: Optional[dict] = None,
    ) -> None:
        """
        Ingest a historical event into ChromaDB for future RAG queries.

        Args:
            event_id: Unique identifier for the event.
            ticker: Stock ticker symbol.
            headline: Event headline or description.
            sentiment_score: Numerical sentiment (-1.0 to 1.0).
            outcome: Optional categorical outcome (e.g., "UP", "DOWN", "NEUTRAL").
            days_later_price_change: Optional realized price change (%) N days later.
            extra_metadata: Optional additional metadata dict.
        """
        metadata = {
            "ticker": ticker,
            "headline": headline,
            "sentiment_score": str(sentiment_score),
            "outcome": outcome or "UNKNOWN",
            "days_later_price_change": str(days_later_price_change)
            if days_later_price_change is not None
            else "0.0",
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        text = f"{ticker} {headline}"
        self.collection.add(
            ids=[event_id],
            documents=[text],
            metadatas=[metadata],
        )

    def ingest_batch(self, events: list[dict]) -> None:
        """
        Ingest multiple historical events in batch.

        Args:
            events: List of dicts with keys: event_id, ticker, headline,
                    sentiment_score, outcome, days_later_price_change.
        """
        ids = []
        documents = []
        metadatas = []

        for event in events:
            ids.append(event["event_id"])
            documents.append(
                f"{event['ticker']} {event['headline']}"
            )
            metadatas.append(
                {
                    "ticker": event["ticker"],
                    "headline": event["headline"],
                    "sentiment_score": str(event["sentiment_score"]),
                    "outcome": event.get("outcome", "UNKNOWN"),
                    "days_later_price_change": str(
                        event.get("days_later_price_change", 0.0)
                    ),
                }
            )

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

    def persist(self) -> None:
        """Persist ChromaDB collection to disk."""
        self.client.persist()


def query_historical_context(
    residual: SentimentResidual, top_k: int = 5
) -> list[HistoricalMatch]:
    """
    Convenience function: query historical context for a sentiment residual.

    Args:
        residual: Current SentimentResidual to match against history.
        top_k: Number of historical matches to return.

    Returns:
        List of HistoricalMatch objects ranked by similarity.
    """
    db_path = os.getenv("SENTINEL_CHROMA_DB", "./data/chroma_db")
    querier = HistorianRAGQuery(db_path=db_path)
    return querier.query(residual, top_k=top_k, min_similarity=0.5)
