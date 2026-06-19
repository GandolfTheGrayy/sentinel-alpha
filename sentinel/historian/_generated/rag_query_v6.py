"""
RAG query interface for Sentinel Sentiment Engine.

This module queries ChromaDB for historical events similar to a current
SentimentResidual, returning ranked HistoricalMatch objects with confidence
scores. It bridges real-time sentiment signals with historical precedent,
enabling the Judge to calibrate predictions against past market behavior.

Used by sentinel/judge/predictor.py during per-ticker analysis.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional
import chromadb
import numpy as np


@dataclass
class SentimentResidual:
    """Represents current sentiment signal for a ticker."""
    ticker: str
    signal_text: str
    signal_type: str  # e.g. "news", "reddit", "sec_filing"
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0


@dataclass
class HistoricalMatch:
    """A historical event matched to current sentiment."""
    event_id: str
    event_date: str
    event_ticker: str
    event_text: str
    event_type: str
    historical_sentiment_score: float
    similarity_score: float  # 0.0 to 1.0, from ChromaDB
    days_to_resolution: int
    actual_price_move: float  # percentage move post-event
    confidence_weight: float  # 0.0 to 1.0


class RAGQueryEngine:
    """ChromaDB-backed RAG query engine for historical event retrieval."""

    def __init__(self, db_path: str = "sentinel_history.db", chroma_dir: str = "./chroma_data"):
        """Initialize RAG engine with ChromaDB and SQLite backend.
        
        Args:
            db_path: Path to SQLite database storing event metadata.
            chroma_dir: Directory containing ChromaDB persistent storage.
        """
        self.db_path = db_path
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        
        # Attempt to get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name="sentiment_events")
        except Exception:
            # Collection doesn't exist; will be populated later
            self.collection = None
        
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema for event metadata if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_events (
                event_id TEXT PRIMARY KEY,
                event_date TEXT NOT NULL,
                event_ticker TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_text TEXT NOT NULL,
                sentiment_score REAL,
                days_to_resolution INTEGER,
                actual_price_move REAL,
                embedding_id TEXT
            )
        """)
        conn.commit()
        conn.close()

    def ingest_event(
        self,
        event_id: str,
        event_date: str,
        event_ticker: str,
        event_type: str,
        event_text: str,
        sentiment_score: float,
        days_to_resolution: int,
        actual_price_move: float,
        embedding: Optional[list] = None,
    ) -> None:
        """Ingest a historical event into SQLite and ChromaDB.
        
        Args:
            event_id: Unique identifier for the event.
            event_date: ISO date string (YYYY-MM-DD).
            event_ticker: Stock ticker symbol.
            event_type: Category (e.g., "earnings", "sec_filing", "news").
            event_text: Full event narrative/text.
            sentiment_score: Historical sentiment (-1.0 to 1.0).
            days_to_resolution: Days from event to price settlement.
            actual_price_move: Actual percentage move post-event.
            embedding: Pre-computed embedding vector (optional; will be generated if None).
        """
        # Store metadata in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO historical_events
            (event_id, event_date, event_ticker, event_type, event_text,
             sentiment_score, days_to_resolution, actual_price_move, embedding_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, event_date, event_ticker, event_type, event_text,
              sentiment_score, days_to_resolution, actual_price_move, event_id))
        conn.commit()
        conn.close()

        # Add to ChromaDB if collection exists
        if self.collection is None:
            self.collection = self.chroma_client.get_or_create_collection(
                name="sentiment_events",
                metadata={"hnsw:space": "cosine"}
            )
        
        # Use provided embedding or let ChromaDB auto-embed via default
        if embedding:
            self.collection.add(
                ids=[event_id],
                documents=[event_text],
                metadatas=[{
                    "event_date": event_date,
                    "event_ticker": event_ticker,
                    "event_type": event_type,
                    "sentiment_score": str(sentiment_score),
                    "days_to_resolution": str(days_to_resolution),
                    "actual_price_move": str(actual_price_move),
                }],
                embeddings=[embedding],
            )
        else:
            self.collection.add(
                ids=[event_id],
                documents=[event_text],
                metadatas=[{
                    "event_date": event_date,
                    "event_ticker": event_ticker,
                    "event_type": event_type,
                    "sentiment_score": str(sentiment_score),
                    "days_to_resolution": str(days_to_resolution),
                    "actual_price_move": str(actual_price_move),
                }],
            )

    def query(self, residual: SentimentResidual, top_k: int = 5) -> list[HistoricalMatch]:
        """Query ChromaDB for top-k historical matches to current sentiment.
        
        Args:
            residual: Current SentimentResidual signal.
            top_k: Number of historical matches to return.
            
        Returns:
            List of HistoricalMatch objects ranked by similarity.
        """
        if self.collection is None:
            return []

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[residual.signal_text],
            n_results=top_k,
            where=None,  # No ticker filtering; consider all events
        )

        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            event_ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for i, event_id in enumerate(event_ids):
                # Fetch full metadata from SQLite
                cursor.execute("""
                    SELECT event_date, event_ticker, event_type, event_text,
                           sentiment_score, days_to_resolution, actual_price_move
                    FROM historical_events WHERE event_id = ?
                """, (event_id,))
                row = cursor.fetchone()

                if row:
                    event_date, event_ticker, event_type, event_text, \
                        hist_sentiment, days_to_res, actual_move = row

                    # Convert distance to similarity (cosine: 0 = identical, 2 = opposite)
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity_score = 1.0 - (distance / 2.0)  # Normalize to [0, 1]
