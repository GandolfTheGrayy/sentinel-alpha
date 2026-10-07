"""
RAG Query Interface for Sentinel Historian.

Given a current SentimentResidual (text + metadata), queries ChromaDB for the
top-k most similar historical events and returns a HistoricalMatch list.
Integrates with the Historian pillar to provide historical context and
confidence weighting for Judge predictions.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
import chromadb
from chromadb.config import Settings


@dataclass
class HistoricalMatch:
    """A single historical event matched by RAG similarity."""
    event_id: str
    similarity_score: float
    event_date: str
    ticker: str
    event_type: str
    summary: str
    outcome: Optional[str]
    metadata: dict


@dataclass
class SentimentResidual:
    """Current sentiment signal to query against history."""
    ticker: str
    text: str
    signal_type: str
    timestamp: str
    metadata: dict


class HistorianRAG:
    """ChromaDB-backed RAG pipeline for historical event lookup."""

    def __init__(self, db_path: str = "./data/chroma_db"):
        """Initialize ChromaDB client and load or create collection.
        
        Args:
            db_path: Path to ChromaDB persistent storage.
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name="sentinel_events",
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        summary: str,
        event_date: str,
        outcome: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Ingest a historical event into the ChromaDB collection.
        
        Args:
            event_id: Unique identifier for the event.
            ticker: Stock ticker symbol.
            event_type: Category (e.g., "earnings_miss", "sec_filing", "regulatory").
            summary: Text description of the event.
            event_date: ISO date string of when the event occurred.
            outcome: Optional description of market outcome (e.g., "+5% next day").
            metadata: Optional dict of additional context.
        """
        if metadata is None:
            metadata = {}
        
        doc_metadata = {
            "ticker": ticker,
            "event_type": event_type,
            "event_date": event_date,
            "outcome": outcome or "unknown",
            **metadata
        }
        
        self.collection.add(
            ids=[event_id],
            documents=[summary],
            metadatas=[doc_metadata],
        )

    def query(
        self,
        residual: SentimentResidual,
        k: int = 5,
        ticker_filter: bool = True,
    ) -> list[HistoricalMatch]:
        """Query ChromaDB for top-k similar historical events.
        
        Args:
            residual: Current SentimentResidual to match.
            k: Number of top matches to return.
            ticker_filter: If True, prioritize matches for the same ticker.
        
        Returns:
            List of HistoricalMatch objects ranked by similarity.
        """
        where_clause = None
        if ticker_filter:
            where_clause = {"ticker": {"$eq": residual.ticker}}
        
        results = self.collection.query(
            query_texts=[residual.text],
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
                    similarity_score=similarity,
                    event_date=meta.get("event_date", "unknown"),
                    ticker=meta.get("ticker", residual.ticker),
                    event_type=meta.get("event_type", "unknown"),
                    summary=results["documents"][0][i] if results["documents"] else "",
                    outcome=meta.get("outcome"),
                    metadata=meta,
                )
                matches.append(match)
        
        return matches

    def query_cross_ticker(
        self,
        residual: SentimentResidual,
        k: int = 5,
    ) -> list[HistoricalMatch]:
        """Query without ticker filter to find cross-sector parallels.
        
        Args:
            residual: Current SentimentResidual to match.
            k: Number of top matches to return.
        
        Returns:
            List of HistoricalMatch objects ranked by similarity (any ticker).
        """
        return self.query(residual, k=k, ticker_filter=False)

    def batch_query(
        self,
        residuals: list[SentimentResidual],
        k: int = 5,
    ) -> dict[str, list[HistoricalMatch]]:
        """Query multiple residuals and return results keyed by ticker.
        
        Args:
            residuals: List of SentimentResidual objects.
            k: Number of top matches per residual.
        
        Returns:
            Dict mapping ticker to list of HistoricalMatch objects.
        """
        results = {}
        for residual in residuals:
            key = f"{residual.ticker}_{residual.signal_type}"
            results[key] = self.query(residual, k=k)
        return results

    def list_events(self, ticker: Optional[str] = None) -> list[dict]:
        """List all ingested events, optionally filtered by ticker.
        
        Args:
            ticker: Optional ticker filter.
        
        Returns:
            List of event metadata dicts.
        """
        where_clause = {"ticker": {"$eq": ticker}} if ticker else None
        results = self.collection.get(where=where_clause)
        
        events = []
        if results and results["ids"]:
            for i, event_id in enumerate(results["ids"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                meta["id"] = event_id
                meta["document"] = results["documents"][i] if results["documents"] else ""
                events.append(meta)
        return events

    def delete_event(self, event_id: str) -> None:
        """Delete a single event from the collection.
        
        Args:
            event_id: ID of the event to delete.
        """
        self.collection.delete(ids=[event_id])

    def clear_collection(self) -> None:
        """Clear all events from the collection (use with caution)."""
        all_results = self.collection.get()
        if all_results and all_results["ids"]:
            self.collection.delete(ids=all_results["ids"])

    def similarity_score_to_confidence(
        self,
        similarity: float,
        historical_outcome_exists: bool = True,
    ) -> float:
        """Convert RAG similarity score to prediction confidence weight.
        
        Args:
            similarity: Cosine similarity score (0.0 to 1.0).
            historical_outcome_exists: Whether the matched event has a known outcome.
        
        Returns:
            Confidence weight (0.0 to 1.0) for use in Judge calibration.
        """
        base_confidence = similarity
        if not historical_outcome_exists:
            base_confidence *= 0.7
        return min(1.0, max(0.0, base_confidence))
