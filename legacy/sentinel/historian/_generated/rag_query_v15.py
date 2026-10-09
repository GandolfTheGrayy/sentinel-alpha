"""
RAG query interface for Sentinel Historian pillar.

Given a current SentimentResidual (ticker, sentiment score, headline text, source),
queries ChromaDB vector database for the top-k most similar historical events.
Returns a HistoricalMatch list with confidence scores and historical context.

This module bridges real-time sentiment signals with historical market patterns,
enabling the Judge to contextualize predictions against precedent.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Optional
import sqlite3

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np


@dataclass
class SentimentResidual:
    """Current sentiment signal to query against history."""
    ticker: str
    sentiment_score: float
    headline_text: str
    source: str


@dataclass
class HistoricalMatch:
    """Historical event matched by RAG similarity."""
    event_id: str
    ticker: str
    date: str
    headline: str
    sentiment_score: float
    price_movement_pct: float
    confidence_score: float


class HistorianRAG:
    """ChromaDB-backed RAG engine for historical event lookup."""

    def __init__(
        self,
        db_path: str = "sentinel_historian.db",
        chroma_path: str = "./chroma_data",
        gemini_model: str = "gemini-3.1-flash-lite-preview",
    ):
        """
        Initialize ChromaDB client and Gemini embedding provider.

        Args:
            db_path: Path to SQLite DB storing event metadata (dates, prices).
            chroma_path: Path to ChromaDB persistent storage.
            gemini_model: Gemini model ID for embeddings/reasoning.
        """
        self.db_path = db_path
        self.gemini_model = gemini_model

        # Initialize Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        genai.configure(api_key=api_key)

        # Initialize ChromaDB with persistence
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_path,
            anonymized_telemetry=False,
        )
        self.chroma_client = chromadb.Client(settings)

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="sentinel_events",
            metadata={"description": "Historical market events with embeddings"},
        )

        # Ensure SQLite DB exists
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite metadata table if not present."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS historical_events (
                event_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                headline TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                price_movement_pct REAL NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
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
        headline: str,
        sentiment_score: float,
        price_movement_pct: float,
        source: str,
    ) -> None:
        """
        Ingest a historical event: embed headline and store metadata.

        Args:
            event_id: Unique identifier for this event.
            ticker: Stock ticker symbol.
            date: Event date (ISO format).
            headline: Event headline text.
            sentiment_score: Pre-computed sentiment (-1.0 to 1.0).
            price_movement_pct: Actual price change following event (%).
            source: Data source (e.g., "SEC", "NewsAPI", "Reddit").
        """
        # Embed headline via Gemini
        embedding = self._embed_text(headline)

        # Store in ChromaDB
        self.collection.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[headline],
            metadatas=[{
                "ticker": ticker,
                "date": date,
                "sentiment_score": sentiment_score,
                "price_movement_pct": price_movement_pct,
                "source": source,
            }],
        )

        # Store metadata in SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO historical_events
            (event_id, ticker, date, headline, sentiment_score, price_movement_pct, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, ticker, date, headline, sentiment_score, price_movement_pct, source),
        )
        conn.commit()
        conn.close()

    def query(
        self,
        residual: SentimentResidual,
        k: int = 5,
        ticker_filter: bool = True,
    ) -> List[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical events similar to current residual.

        Args:
            residual: Current SentimentResidual (ticker, score, headline, source).
            k: Number of top matches to return.
            ticker_filter: If True, only return matches for same ticker.

        Returns:
            List of HistoricalMatch objects sorted by confidence (descending).
        """
        # Embed the current headline
        query_embedding = self._embed_text(residual.headline_text)

        # Query ChromaDB
        where_filter = None
        if ticker_filter:
            where_filter = {"ticker": {"$eq": residual.ticker}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Convert distances to confidence scores (cosine distance → similarity)
        matches = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]

                # Cosine distance in ChromaDB: 0 = identical, 2 = opposite
                # Convert to confidence: confidence = 1 - (distance / 2)
                confidence = max(0.0, 1.0 - (distance / 2.0))

                # Fetch full record from SQLite
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT date, headline FROM historical_events WHERE event_id = ?",
                    (event_id,),
                )
                row = cursor.fetchone()
                conn.close()

                if row:
                    date, headline = row
                    match = HistoricalMatch(
                        event_id=event_id,
                        ticker=metadata["ticker"],
                        date=date,
                        headline=headline,
                        sentiment_score=metadata["sentiment_score"],
                        price_movement_pct=metadata["price_movement_pct"],
                        confidence_score=confidence,
                    )
                    matches.append(match)

        # Sort by confidence descending
        matches.sort(key=lambda m: m.confidence_score, reverse=True)
        return matches[:k]

    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using Gemini embeddings API.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY
