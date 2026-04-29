"""
RAG Query Engine for Sentinel Historian.

This module provides the core RAG (Retrieval-Augmented Generation) interface.
Given a current SentimentResidual (sentiment signal + metadata), it queries ChromaDB
for the top-k most similar historical events and returns ranked HistoricalMatch
results with confidence scores. Used by Judge and Linguist to contextualize
sentiment anomalies against past market behavior.
"""

import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Union
import chromadb
from chromadb.config import Settings


@dataclass
class SentimentResidual:
    """
    Represents a current sentiment signal to be matched against historical events.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL").
        signal_text: Free-form sentiment or event description.
        signal_type: Category of signal (e.g., "reddit_surge", "sec_filing_hedge", "dev_activity_drop").
        signal_score: Float [-1.0, 1.0] representing sentiment polarity.
        timestamp: ISO 8601 timestamp when signal was detected.
        metadata: Optional dict with additional context (e.g., subreddit, post_id).
    """
    ticker: str
    signal_text: str
    signal_type: str
    signal_score: float
    timestamp: str
    metadata: dict = field(default_factory=dict)


@dataclass
class HistoricalMatch:
    """
    Represents a historical event matched via RAG similarity search.
    
    Attributes:
        event_id: Unique identifier in ChromaDB.
        ticker: Stock symbol of historical event.
        event_text: Description of the historical event.
        event_type: Category (e.g., "earnings_beat", "sec_warning", "insider_buy").
        event_date: ISO 8601 date of the historical event.
        market_move_pct: Actual market movement (%) in days following event.
        similarity_score: Cosine similarity [0.0, 1.0] to current residual.
        confidence: Confidence score [0.0, 1.0] derived from similarity + historical frequency.
        matched_metadata: Dict with details on match reason.
    """
    event_id: str
    ticker: str
    event_text: str
    event_type: str
    event_date: str
    market_move_pct: float
    similarity_score: float
    confidence: float
    matched_metadata: dict = field(default_factory=dict)


class RAGQueryEngine:
    """
    Manages ChromaDB vector store and RAG query logic for historical event retrieval.
    """

    def __init__(
        self,
        chroma_db_path: str = "./sentinel_chroma_db",
        collection_name: str = "historical_events",
        sqlite_db_path: str = "./sentinel_events.db",
    ) -> None:
        """
        Initialize RAG query engine with ChromaDB and SQLite backends.
        
        Args:
            chroma_db_path: Path to ChromaDB persistent storage.
            collection_name: Name of ChromaDB collection for historical events.
            sqlite_db_path: Path to SQLite metadata store.
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.sqlite_db_path = sqlite_db_path
        
        # Initialize ChromaDB client with persistence.
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_db_path,
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Ensure SQLite schema exists.
        self._init_sqlite_schema()

    def _init_sqlite_schema(self) -> None:
        """Initialize SQLite schema for historical event metadata."""
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_events (
                event_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_text TEXT NOT NULL,
                market_move_pct REAL NOT NULL,
                frequency_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON historical_events (ticker, event_date)
        """)
        conn.commit()
        conn.close()

    def ingest_historical_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        event_date: str,
        event_text: str,
        market_move_pct: float,
        created_at: str,
    ) -> None:
        """
        Ingest a historical event into both ChromaDB vector store and SQLite metadata.
        
        Args:
            event_id: Unique identifier for the event.
            ticker: Stock symbol.
            event_type: Category of event (e.g., "earnings_beat").
            event_date: ISO 8601 date string.
            event_text: Free-form event description.
            market_move_pct: Observed market movement (%) post-event.
            created_at: ISO 8601 timestamp of ingestion.
        """
        # Add to ChromaDB vector store.
        self.collection.add(
            ids=[event_id],
            documents=[event_text],
            metadatas=[{
                "ticker": ticker,
                "event_type": event_type,
                "event_date": event_date,
                "market_move_pct": market_move_pct,
            }],
        )
        
        # Add metadata to SQLite.
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO historical_events 
            (event_id, ticker, event_type, event_date, event_text, market_move_pct, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (event_id, ticker, event_type, event_date, event_text, market_move_pct, created_at))
        conn.commit()
        conn.close()

    def query(
        self,
        residual: SentimentResidual,
        top_k: int = 5,
        ticker_filter: Optional[str] = None,
    ) -> list[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical events similar to current sentiment residual.
        
        Args:
            residual: Current SentimentResidual to match against history.
            top_k: Number of results to return (default 5).
            ticker_filter: If provided, restrict results to this ticker only.
            
        Returns:
            List of HistoricalMatch objects ranked by confidence.
        """
        # Query ChromaDB for similar documents.
        query_result = self.collection.query(
            query_texts=[residual.signal_text],
            n_results=top_k,
            where={"ticker": ticker_filter} if ticker_filter else None,
        )
        
        matches: list[HistoricalMatch] = []
        
        if not query_result or not query_result["ids"] or len(query_result["ids"]) == 0:
            return matches
        
        event_ids = query_result["ids"][0]
        distances = query_result["distances"][0]
        metadatas = query_result["metadatas"][0]
        
        # Fetch full event details from SQLite.
        conn = sqlite3.connect(self.sqlite_db_path)
        cursor = conn.cursor()
        
        for event_id, distance, metadata in zip(event_ids, distances, metadatas):
            # ChromaDB returns distance; convert to similarity [0, 1].
