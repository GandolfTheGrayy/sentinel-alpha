"""
RAG Query Interface for Sentinel Historian.

Given a current SentimentResidual (ticker, sentiment score, key signals),
queries ChromaDB for the top-k most similar historical events and returns
a HistoricalMatch list with confidence scores. Integrates with the Historian
pillar to ground real-time predictions in historical precedent.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional
import chromadb
import numpy as np


@dataclass
class SentimentResidual:
    """Encapsulates current sentiment state for a ticker."""
    ticker: str
    sentiment_score: float
    key_signals: list[str]
    timestamp: str


@dataclass
class HistoricalMatch:
    """Represents a matched historical event from ChromaDB."""
    event_id: str
    ticker: str
    event_date: str
    description: str
    historical_sentiment: float
    actual_return: float
    similarity_score: float
    confidence: float


class HistorianRAGQuery:
    """Query interface for ChromaDB historical event retrieval."""

    def __init__(self, db_path: str = "sentinel_history.db", chroma_path: str = ".chroma"):
        """
        Initialize RAG query interface.
        
        Args:
            db_path: Path to SQLite database of historical events.
            chroma_path: Path to ChromaDB persistent storage.
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure ChromaDB collection exists; create if absent."""
        try:
            self.collection = self.client.get_collection(name="sentiment_events")
        except Exception:
            self.collection = self.client.create_collection(
                name="sentiment_events",
                metadata={"hnsw:space": "cosine"}
            )

    def ingest_historical_event(
        self,
        event_id: str,
        ticker: str,
        event_date: str,
        description: str,
        sentiment_score: float,
        actual_return: float
    ) -> None:
        """
        Ingest a historical event into ChromaDB.
        
        Args:
            event_id: Unique identifier for the event.
            ticker: Stock ticker symbol.
            event_date: Date of the event (ISO format).
            description: Text description of the event.
            sentiment_score: Historical sentiment (-1.0 to 1.0).
            actual_return: Actual return following the event (%).
        """
        metadata = {
            "ticker": ticker,
            "event_date": event_date,
            "sentiment_score": sentiment_score,
            "actual_return": actual_return
        }
        self.collection.add(
            ids=[event_id],
            documents=[description],
            metadatas=[metadata]
        )
        self._store_event_local(
            event_id, ticker, event_date, description, sentiment_score, actual_return
        )

    def _store_event_local(
        self,
        event_id: str,
        ticker: str,
        event_date: str,
        description: str,
        sentiment_score: float,
        actual_return: float
    ) -> None:
        """Store event in local SQLite for audit and recovery."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_events (
                    event_id TEXT PRIMARY KEY,
                    ticker TEXT,
                    event_date TEXT,
                    description TEXT,
                    sentiment_score REAL,
                    actual_return REAL
                )
                """
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO historical_events
                (event_id, ticker, event_date, description, sentiment_score, actual_return)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_id, ticker, event_date, description, sentiment_score, actual_return)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to store event locally: {e}")

    def query(
        self,
        residual: SentimentResidual,
        top_k: int = 5,
        ticker_filter: Optional[str] = None
    ) -> list[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical matches.
        
        Args:
            residual: Current SentimentResidual to match against history.
            top_k: Number of historical matches to return.
            ticker_filter: If provided, restrict matches to this ticker.
        
        Returns:
            List of HistoricalMatch objects, sorted by similarity descending.
        """
        if not self.collection:
            return []

        query_text = " ".join(residual.key_signals)
        where_filter = None
        if ticker_filter:
            where_filter = {"ticker": {"$eq": ticker_filter}}

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            print(f"Warning: ChromaDB query failed: {e}")
            return []

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                similarity = 1.0 - distance if distance is not None else 0.0
                similarity = max(0.0, min(1.0, similarity))
                
                confidence = self._compute_confidence(
                    similarity, metadata, residual
                )

                match = HistoricalMatch(
                    event_id=event_id,
                    ticker=metadata.get("ticker", "UNKNOWN"),
                    event_date=metadata.get("event_date", ""),
                    description=results["documents"][0][i],
                    historical_sentiment=float(metadata.get("sentiment_score", 0.0)),
                    actual_return=float(metadata.get("actual_return", 0.0)),
                    similarity_score=similarity,
                    confidence=confidence
                )
                matches.append(match)

        return sorted(matches, key=lambda m: m.confidence, reverse=True)

    def _compute_confidence(
        self,
        similarity: float,
        metadata: dict,
        residual: SentimentResidual
    ) -> float:
        """
        Compute confidence score for a match.
        
        Blends similarity with metadata alignment (sentiment direction, ticker match).
        
        Args:
            similarity: ChromaDB cosine similarity (0.0–1.0).
            metadata: Event metadata dict.
            residual: Current sentiment residual.
        
        Returns:
            Confidence score (0.0–1.0).
        """
        base_confidence = similarity
        
        ticker_match = 1.0 if metadata.get("ticker") == residual.ticker else 0.7
        
        hist_sentiment = metadata.get("sentiment_score", 0.0)
        sentiment_direction_match = 1.0 if np.sign(hist_sentiment) == np.sign(residual.sentiment_score) else 0.8
        
        confidence = 0.5 * base_confidence + 0.3 * ticker_match + 0.2 * sentiment_direction_match
        return float(np.clip(confidence, 0.0, 1.0))

    def bulk_ingest_from_csv(self, csv_path: str) -> int:
        """
        Bulk ingest historical events from CSV.
        
        Expects columns: event_id, ticker, event_date, description, sentiment_score, actual_return.
        
        Args:
            csv_path: Path to CSV
