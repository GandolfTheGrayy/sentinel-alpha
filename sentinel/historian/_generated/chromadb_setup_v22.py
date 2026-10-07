"""
ChromaDB vector database initialization and typed client wrapper for Sentinel.

This module sets up and manages a local ChromaDB instance with collections for
market events, SEC filings, and news sentiment. It provides a typed client wrapper
that abstracts persistence, embedding, and query operations for the RAG pipeline.

Role in Sentinel: Historian pillar foundation — enables semantic search over
historical market context, regulatory filings, and sentiment signals.
"""

import os
import sqlite3
from typing import Optional, TypeVar, Generic, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import chromadb
from chromadb.config import Settings


T = TypeVar("T")


@dataclass
class ChromaEvent:
    """Typed container for market event metadata."""
    ticker: str
    event_type: str  # e.g., "earnings", "sec_8k", "news"
    date: str  # ISO format YYYY-MM-DD
    title: str
    body: str
    source: str  # e.g., "edgar", "reuters", "reddit"
    sentiment: Optional[float] = None  # [-1, 1] range
    confidence: Optional[float] = None  # [0, 1] range


@dataclass
class ChromaFiling:
    """Typed container for SEC filing metadata."""
    ticker: str
    cik: str
    form_type: str  # e.g., "8-K", "10-Q", "10-K"
    filing_date: str  # ISO format YYYY-MM-DD
    accession_number: str
    url: str
    summary: str
    full_text: str


class ChromaDBClient:
    """
    Typed wrapper around ChromaDB for Sentinel's historian pillar.
    
    Manages local vector database initialization, collection creation,
    and provides typed methods for upserting and querying events/filings.
    """

    def __init__(self, db_path: str = "./data/chroma_db") -> None:
        """
        Initialize ChromaDB client with local storage.
        
        Args:
            db_path: Path to local ChromaDB directory. Defaults to ./data/chroma_db
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
        )
        self.client = chromadb.Client(settings)
        self._init_collections()

    def _init_collections(self) -> None:
        """Initialize or retrieve standard collections for events and filings."""
        self.events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Market events, news, and sentiment signals"},
        )
        self.filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC filings (8-K, 10-Q, 10-K) with full text"},
        )
        self.sentiment_collection = self.client.get_or_create_collection(
            name="sentiment_corpus",
            metadata={"description": "Aggregated sentiment snippets by ticker/date"},
        )

    def upsert_event(self, event: ChromaEvent) -> str:
        """
        Upsert a market event into the events collection.
        
        Args:
            event: ChromaEvent dataclass instance.
            
        Returns:
            Document ID (ticker_eventtype_date hash).
        """
        doc_id = f"{event.ticker}_{event.event_type}_{event.date}"
        metadata = {
            "ticker": event.ticker,
            "event_type": event.event_type,
            "date": event.date,
            "source": event.source,
        }
        if event.sentiment is not None:
            metadata["sentiment"] = event.sentiment
        if event.confidence is not None:
            metadata["confidence"] = event.confidence

        self.events_collection.upsert(
            ids=[doc_id],
            documents=[event.body],
            metadatas=[metadata],
        )
        return doc_id

    def upsert_filing(self, filing: ChromaFiling) -> str:
        """
        Upsert a SEC filing into the filings collection.
        
        Args:
            filing: ChromaFiling dataclass instance.
            
        Returns:
            Document ID (ticker_accession_number).
        """
        doc_id = f"{filing.ticker}_{filing.accession_number}"
        metadata = {
            "ticker": filing.ticker,
            "cik": filing.cik,
            "form_type": filing.form_type,
            "filing_date": filing.filing_date,
            "accession_number": filing.accession_number,
            "url": filing.url,
        }

        self.filings_collection.upsert(
            ids=[doc_id],
            documents=[filing.full_text],
            metadatas=[metadata],
        )
        return doc_id

    def query_events(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over market events via embedding similarity.
        
        Args:
            query_text: Natural language query string.
            ticker: Optional ticker filter.
            n_results: Number of results to return.
            
        Returns:
            List of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        where_filter = {"ticker": ticker} if ticker else None
        results = self.events_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
        )

        # Flatten results into list of dicts
        output = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return output

    def query_filings(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        form_type: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over SEC filings via embedding similarity.
        
        Args:
            query_text: Natural language query string.
            ticker: Optional ticker filter.
            form_type: Optional form type filter (e.g., "8-K").
            n_results: Number of results to return.
            
        Returns:
            List of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        filters = []
        if ticker:
            filters.append({"ticker": ticker})
        if form_type:
            filters.append({"form_type": form_type})

        where_filter = None
        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        results = self.filings_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter,
        )

        output = []
        if results["ids"] and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append(
                    {
                        "id": doc_id,
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
        return
