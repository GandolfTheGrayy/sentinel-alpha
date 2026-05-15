"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module initializes a local ChromaDB instance, defines collections for
market events and SEC filings, and provides a typed client interface for
RAG queries in the historian pillar. Handles schema setup, collection creation,
and persistence across sessions.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any, List

import chromadb
from chromadb.config import Settings


DB_ROOT: Path = Path(__file__).parent.parent.parent / "data" / "chromadb"
DB_PATH: str = str(DB_ROOT)
SQLITE_PATH: str = str(DB_ROOT / "sentinel.db")


def ensure_db_directory() -> None:
    """Create ChromaDB data directory if it does not exist."""
    DB_ROOT.mkdir(parents=True, exist_ok=True)


def init_chromadb_client() -> chromadb.Client:
    """Initialize and return a ChromaDB client with persistent storage."""
    ensure_db_directory()
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_PATH,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def init_collections(client: chromadb.Client) -> Dict[str, chromadb.Collection]:
    """Initialize standard collections for market events and filings; return dict of collections."""
    collections = {}
    
    # Market events collection: news, sentiment shifts, volatility spikes
    collections["market_events"] = client.get_or_create_collection(
        name="market_events",
        metadata={"description": "News, sentiment, volatility events indexed by ticker + date"},
    )
    
    # SEC filings collection: 8-K, 10-Q, 10-K text and metadata
    collections["sec_filings"] = client.get_or_create_collection(
        name="sec_filings",
        metadata={"description": "SEC EDGAR filings (8-K, 10-Q, 10-K) indexed by ticker + filing date"},
    )
    
    # Analyst sentiment collection: earnings calls, guidance, rating changes
    collections["analyst_sentiment"] = client.get_or_create_collection(
        name="analyst_sentiment",
        metadata={"description": "Analyst reports, earnings call transcripts, and rating changes"},
    )
    
    # Historical patterns collection: past market moves, event outcomes, correlations
    collections["historical_patterns"] = client.get_or_create_collection(
        name="historical_patterns",
        metadata={"description": "Historical event outcomes, market correlations, and macro patterns"},
    )
    
    return collections


class SentinelChromaDB:
    """Typed wrapper around ChromaDB client for Sentinel RAG queries."""
    
    def __init__(self, persist: bool = True) -> None:
        """Initialize SentinelChromaDB with optional persistence."""
        ensure_db_directory()
        self.client: chromadb.Client = init_chromadb_client() if persist else chromadb.Client()
        self.collections: Dict[str, chromadb.Collection] = init_collections(self.client)
    
    def add_market_event(
        self,
        ticker: str,
        date: str,
        event_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a market event (news, sentiment) to the market_events collection."""
        doc_id = f"{ticker}_{date}_{event_type}"
        meta = metadata or {}
        meta.update({"ticker": ticker, "date": date, "event_type": event_type})
        self.collections["market_events"].add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
    
    def add_sec_filing(
        self,
        ticker: str,
        filing_date: str,
        filing_type: str,
        text: str,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a SEC filing to the sec_filings collection."""
        doc_id = f"{ticker}_{filing_date}_{filing_type}"
        meta = metadata or {}
        meta.update({"ticker": ticker, "filing_date": filing_date, "filing_type": filing_type})
        if url:
            meta["url"] = url
        self.collections["sec_filings"].add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
    
    def add_analyst_sentiment(
        self,
        ticker: str,
        date: str,
        sentiment_type: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add analyst sentiment (earnings call, rating change) to the analyst_sentiment collection."""
        doc_id = f"{ticker}_{date}_{sentiment_type}"
        meta = metadata or {}
        meta.update({"ticker": ticker, "date": date, "sentiment_type": sentiment_type})
        self.collections["analyst_sentiment"].add(
            ids=[doc_id],
            documents=[text],
            metadatas=[meta],
        )
    
    def add_historical_pattern(
        self,
        pattern_id: str,
        description: str,
        tags: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a historical pattern (past correlations, event outcomes) to historical_patterns collection."""
        meta = metadata or {}
        meta["tags"] = ",".join(tags)
        self.collections["historical_patterns"].add(
            ids=[pattern_id],
            documents=[description],
            metadatas=[meta],
        )
    
    def query_market_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Query market_events collection by semantic similarity; optionally filter by ticker."""
        where_filter = {"ticker": ticker} if ticker else None
        results = self.collections["market_events"].query(
            query_texts=[query_text],
            where=where_filter,
            n_results=limit,
        )
        return results
    
    def query_sec_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Query sec_filings collection by semantic similarity; optionally filter by ticker/filing_type."""
        where_filter = {}
        if ticker:
            where_filter["ticker"] = ticker
        if filing_type:
            where_filter["filing_type"] = filing_type
        where_clause = where_filter if where_filter else None
        
        results = self.collections["sec_filings"].query(
            query_texts=[query_text],
            where=where_clause,
            n_results=limit,
        )
        return results
    
    def query_analyst_sentiment(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Query analyst_sentiment collection by semantic similarity; optionally filter by ticker."""
        where_filter = {"ticker": ticker} if ticker else None
        results = self.collections["analyst_sentiment"].query(
            query_texts=[query_text],
            where=where_filter,
            n_results=limit,
        )
        return results
    
    def query_historical_patterns(
        self,
        query_text: str,
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Query historical_patterns collection by semantic similarity; optionally filter by tags."""
        where_filter = None
        if tags:
            where_filter = {"tags": {"$contains": ",".join(tags)}}
