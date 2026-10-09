"""
RAG Query Interface for Sentinel Sentiment Engine.

This module provides the HistoricalMatcher class, which queries ChromaDB
for the top-k most similar historical events given a current SentimentResidual.
It returns ranked HistoricalMatch objects with confidence scores, enabling
the Judge to contextualize predictions against past market behavior.

Part of the Historian pillar: historical event lookup and confidence weighting.
"""

import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

import chromadb
import numpy as np
from anthropic import Anthropic


@dataclass
class HistoricalMatch:
    """A single historical event match from ChromaDB."""

    event_id: str
    ticker: str
    date: str
    description: str
    embedding_distance: float
    confidence_score: float
    outcome: Optional[str] = None


@dataclass
class SentimentResidual:
    """Input: current sentiment signal for a ticker."""

    ticker: str
    sentiment_text: str
    sentiment_score: float
    source: str


class HistoricalMatcher:
    """Query ChromaDB for similar historical events and rank by confidence."""

    def __init__(
        self,
        chroma_db_path: str = "./chroma_data",
        sqlite_db_path: str = "./sentinel_history.db",
        top_k: int = 5,
    ):
        """
        Initialize HistoricalMatcher with ChromaDB and SQLite backends.

        Args:
            chroma_db_path: Path to ChromaDB persistent storage.
            sqlite_db_path: Path to SQLite metadata store.
            top_k: Number of top matches to return.
        """
        self.top_k = top_k
        self.sqlite_db_path = sqlite_db_path
        self._init_chromadb(chroma_db_path)
        self._init_sqlite()

    def _init_chromadb(self, db_path: str) -> None:
        """Initialize ChromaDB client and collection."""
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.chroma_client.get_collection(
                name="historical_events"
            )
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="historical_events",
                metadata={"hnsw:space": "cosine"},
            )

    def _init_sqlite(self) -> None:
        """Initialize SQLite metadata store for historical events."""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_events (
                event_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def ingest_event(
        self,
        event_id: str,
        ticker: str,
        date: str,
        description: str,
        source: str,
        outcome: Optional[str] = None,
    ) -> None:
        """
        Ingest a historical event into ChromaDB and SQLite.

        Args:
            event_id: Unique event identifier.
            ticker: Stock ticker symbol.
            date: Event date (ISO format).
            description: Event description text.
            source: Source of the event (e.g., "SEC", "news", "reddit").
            outcome: Optional market outcome (e.g., "up 5%", "down 2%").
        """
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO historical_events
            (event_id, ticker, date, description, source, outcome)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (event_id, ticker, date, description, source, outcome),
        )
        conn.commit()
        conn.close()

        self.collection.upsert(
            ids=[event_id],
            documents=[description],
            metadatas=[
                {
                    "ticker": ticker,
                    "date": date,
                    "source": source,
                    "outcome": outcome or "unknown",
                }
            ],
        )

    def query(self, residual: SentimentResidual) -> list[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical events similar to current residual.

        Args:
            residual: Current SentimentResidual (ticker, text, score, source).

        Returns:
            List of HistoricalMatch objects ranked by confidence.
        """
        if self.collection.count() == 0:
            return []

        query_text = residual.sentiment_text
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=self.top_k,
            )
        except Exception:
            return []

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                distance = (
                    results["distances"][0][i]
                    if results.get("distances")
                    else 0.0
                )
                metadata = (
                    results["metadatas"][0][i]
                    if results.get("metadatas")
                    else {}
                )

                confidence = self._compute_confidence(
                    distance, residual.sentiment_score
                )

                match = HistoricalMatch(
                    event_id=event_id,
                    ticker=metadata.get("ticker", residual.ticker),
                    date=metadata.get("date", "unknown"),
                    description=results["documents"][0][i]
                    if results.get("documents")
                    else "",
                    embedding_distance=float(distance),
                    confidence_score=confidence,
                    outcome=metadata.get("outcome"),
                )
                matches.append(match)

        matches.sort(key=lambda m: m.confidence_score, reverse=True)
        return matches[: self.top_k]

    def _compute_confidence(
        self, embedding_distance: float, sentiment_score: float
    ) -> float:
        """
        Compute confidence score from embedding distance and sentiment magnitude.

        Args:
            embedding_distance: Cosine distance (0 = identical, 1 = orthogonal).
            sentiment_score: Sentiment magnitude (-1 to +1).

        Returns:
            Confidence score (0 to 1).
        """
        similarity = 1.0 - min(embedding_distance, 1.0)
        magnitude = abs(sentiment_score)
        confidence = (similarity * 0.7) + (magnitude * 0.3)
        return float(np.clip(confidence, 0.0, 1.0))

    def synthesize_with_claude(
        self, residual: SentimentResidual, matches: list[HistoricalMatch]
    ) -> str:
        """
        Use Claude to synthesize historical matches into a reasoning narrative.

        Args:
            residual: Current SentimentResidual.
            matches: List of HistoricalMatch objects from query().

        Returns:
            Claude-generated synthesis text.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "No ANTHROPIC_API_KEY set; skipping synthesis."

        client = Anthropic(api_key=api_key)

        match_text = "\n".join(
            [
                f"  - Date: {m.date}, Ticker: {m.ticker}, "
                f"Confidence: {m.confidence_score:.2f}, Outcome: {m.outcome}"
                f"\n    Description: {m.description[:200]}"
                for m in matches
            ]
        )

        prompt = f"""
You are
