"""
RAG query interface for Sentinel Historian.

This module implements the Historian pillar's core responsibility: given a
current SentimentResidual (ticker, sentiment score, source signals), query
ChromaDB for the top-k most similar historical events and return ranked
HistoricalMatch results with confidence scores.

Used by Judge (predictor.py) to ground predictions in precedent and by
Linguist to detect anomalous sentiment patterns.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings


@dataclass
class SentimentResidual:
    """Input: current ticker sentiment snapshot for RAG lookup."""
    ticker: str
    sentiment_score: float  # [-1.0, 1.0]
    source: str  # "reddit", "news", "sec", "github"
    signal_text: str  # raw text snippet that triggered the signal
    timestamp: datetime


@dataclass
class HistoricalMatch:
    """Output: ranked historical event matching current signal."""
    event_id: str
    ticker: str
    historical_date: datetime
    historical_sentiment_score: float
    similarity_score: float  # [0.0, 1.0] from ChromaDB distance
    subsequent_price_move: Optional[float]  # % change in following period
    metadata: dict  # original event context


class HistorianRAG:
    """ChromaDB-backed retrieval for historical sentiment → price outcome pairs."""

    def __init__(self, db_path: str = "./sentinel_chroma"):
        """
        Initialize ChromaDB client and load embedded collection.
        
        Args:
            db_path: filesystem path to persistent ChromaDB storage
        """
        self.db_path = db_path
        self._ensure_db_exists()
        
        settings = Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            persist_directory=db_path,
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name="sentiment_events",
            metadata={"hnsw:space": "cosine"}
        )

    def _ensure_db_exists(self) -> None:
        """Create ChromaDB directory if missing."""
        os.makedirs(self.db_path, exist_ok=True)

    def ingest_event(
        self,
        event_id: str,
        ticker: str,
        event_date: datetime,
        sentiment_text: str,
        sentiment_score: float,
        price_move_pct: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a historical sentiment event to the collection.
        
        Args:
            event_id: unique identifier (e.g., "AAPL_2023-01-15_reddit_12345")
            ticker: stock ticker
            event_date: when the event occurred
            sentiment_text: raw text (headline, post, filing excerpt)
            sentiment_score: [-1.0, 1.0] labeling
            price_move_pct: subsequent % price change (for outcome tracking)
            metadata: arbitrary context dict
        """
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "ticker": ticker,
            "event_date": event_date.isoformat(),
            "sentiment_score": sentiment_score,
            "price_move_pct": price_move_pct,
        })
        
        self.collection.add(
            ids=[event_id],
            documents=[sentiment_text],
            metadatas=[metadata],
        )

    def query(
        self,
        residual: SentimentResidual,
        k: int = 5,
        ticker_filter: bool = True,
    ) -> list[HistoricalMatch]:
        """
        Retrieve top-k historical events most similar to current signal.
        
        Args:
            residual: current SentimentResidual snapshot
            k: number of matches to return
            ticker_filter: if True, only return matches for same ticker
        
        Returns:
            sorted list of HistoricalMatch, highest similarity first
        """
        where_clause = None
        if ticker_filter:
            where_clause = {"ticker": {"$eq": residual.ticker}}
        
        results = self.collection.query(
            query_texts=[residual.signal_text],
            n_results=k,
            where=where_clause,
        )
        
        matches = []
        if results and results["ids"] and len(results["ids"]) > 0:
            for i, event_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                
                match = HistoricalMatch(
                    event_id=event_id,
                    ticker=meta.get("ticker", residual.ticker),
                    historical_date=datetime.fromisoformat(
                        meta.get("event_date", residual.timestamp.isoformat())
                    ),
                    historical_sentiment_score=float(
                        meta.get("sentiment_score", 0.0)
                    ),
                    similarity_score=similarity,
                    subsequent_price_move=float(meta["price_move_pct"])
                    if meta.get("price_move_pct") is not None
                    else None,
                    metadata=meta,
                )
                matches.append(match)
        
        return matches

    def query_cross_ticker(
        self,
        residual: SentimentResidual,
        k: int = 5,
    ) -> list[HistoricalMatch]:
        """
        Retrieve top-k historical events ignoring ticker (sector-wide patterns).
        
        Args:
            residual: current SentimentResidual snapshot
            k: number of matches to return
        
        Returns:
            sorted list of HistoricalMatch across all tickers
        """
        return self.query(residual, k=k, ticker_filter=False)

    def get_event_outcomes(self, ticker: str) -> dict[str, float]:
        """
        Aggregate price outcomes for all historical events in a ticker.
        
        Args:
            ticker: stock symbol
        
        Returns:
            dict mapping sentiment_score → list of (event_id, price_move_pct)
        """
        results = self.collection.get(
            where={"ticker": {"$eq": ticker}},
            include=["metadatas"]
        )
        
        outcomes = {}
        if results and results["metadatas"]:
            for meta in results["metadatas"]:
                sentiment = float(meta.get("sentiment_score", 0.0))
                price_move = meta.get("price_move_pct")
                
                if sentiment not in outcomes:
                    outcomes[sentiment] = []
                if price_move is not None:
                    outcomes[sentiment].append(float(price_move))
        
        return outcomes

    def reset_collection(self) -> None:
        """Drop and recreate the sentiment_events collection (for testing)."""
        self.client.delete_collection(name="sentiment_events")
        self.collection = self.client.get_or_create_collection(
            name="sentiment_events",
            metadata={"hnsw:space": "cosine"}
        )

    def health_check(self) -> dict:
        """
        Return stats on the collection: count, date range, ticker diversity.
        
        Returns:
            dict with keys: total_events, earliest_date, latest_date, unique_tickers
        """
        results = self.collection.get(include=["metadatas"])
        
        if not results or not results["metadatas"]:
            return {
                "total_events": 0,
                "earliest_date": None,
                "latest_date": None,
                "unique_tickers": [],
            }
        
        dates = []
        tickers = set()
        
        for meta in results["metadatas"]:
            if "
