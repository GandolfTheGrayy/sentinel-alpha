"""
Sentinel Historian — ChromaDB Vector Database Setup Module.

Initializes and manages the local ChromaDB instance for RAG operations.
Defines typed collections for market events, SEC filings, and news sentiment.
Provides a typed client wrapper to ensure consistent embedding and retrieval
across the historian pillar.

Responsibilities:
  - Initialize ChromaDB client (persistent local storage).
  - Create/access collections: "market_events", "sec_filings", "news_sentiment".
  - Define embedding function and metadata schemas.
  - Provide ChromaClientWrapper for safe, typed access across Sentinel.
"""

import os
import sqlite3
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


class ChromaClientWrapper:
    """
    Typed wrapper around ChromaDB client for Sentinel historian operations.
    
    Manages persistent collections and ensures consistent embedding/retrieval.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """
        Initialize ChromaDB client and collections.
        
        Args:
            db_path: Path to ChromaDB data directory. Defaults to ~/.sentinel/chroma.
        """
        if db_path is None:
            db_path = os.path.expanduser("~/.sentinel/chroma")
        
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
        """
        Initialize or retrieve standard Sentinel collections.
        """
        # Market events: historical price moves, catalyst timings, volatility spikes
        self.market_events_collection = self.client.get_or_create_collection(
            name="market_events",
            metadata={"description": "Historical market events and catalyst timings"},
        )
        
        # SEC filings: 8-K, 10-Q, 10-K documents with embedding-based retrieval
        self.sec_filings_collection = self.client.get_or_create_collection(
            name="sec_filings",
            metadata={"description": "SEC EDGAR filings (8-K, 10-Q, 10-K)"},
        )
        
        # News sentiment: headlines, articles, sentiment scores
        self.news_sentiment_collection = self.client.get_or_create_collection(
            name="news_sentiment",
            metadata={"description": "News headlines and sentiment signals"},
        )

    def add_market_event(
        self,
        event_id: str,
        ticker: str,
        event_type: str,
        description: str,
        timestamp: str,
        price_impact: Optional[float] = None,
    ) -> None:
        """
        Add a market event to the vector store.
        
        Args:
            event_id: Unique event identifier.
            ticker: Stock ticker symbol.
            event_type: Category (e.g., "earnings", "acquisition", "downgrade").
            description: Human-readable event summary.
            timestamp: ISO 8601 datetime string.
            price_impact: Observed price change (%) following event, if known.
        """
        self.market_events_collection.add(
            ids=[event_id],
            documents=[description],
            metadatas=[{
                "ticker": ticker,
                "event_type": event_type,
                "timestamp": timestamp,
                "price_impact": price_impact,
            }],
        )

    def add_sec_filing(
        self,
        filing_id: str,
        ticker: str,
        filing_type: str,
        text: str,
        timestamp: str,
        url: Optional[str] = None,
    ) -> None:
        """
        Add a SEC filing to the vector store for embedding-based retrieval.
        
        Args:
            filing_id: Unique filing identifier (e.g., CIK-accession).
            ticker: Stock ticker symbol.
            filing_type: Type of filing (e.g., "8-K", "10-Q", "10-K").
            text: Full or excerpt text of the filing.
            timestamp: ISO 8601 datetime string of filing date.
            url: URL to the filing on SEC EDGAR.
        """
        self.sec_filings_collection.add(
            ids=[filing_id],
            documents=[text],
            metadatas=[{
                "ticker": ticker,
                "filing_type": filing_type,
                "timestamp": timestamp,
                "url": url,
            }],
        )

    def add_news_article(
        self,
        article_id: str,
        ticker: str,
        headline: str,
        text: str,
        timestamp: str,
        sentiment_score: Optional[float] = None,
        source: Optional[str] = None,
    ) -> None:
        """
        Add a news article/headline to the vector store.
        
        Args:
            article_id: Unique article identifier.
            ticker: Stock ticker symbol.
            headline: Article headline.
            text: Article body or excerpt.
            timestamp: ISO 8601 datetime string of publication.
            sentiment_score: Pre-computed sentiment score (-1.0 to 1.0).
            source: News source (e.g., "Reuters", "Bloomberg").
        """
        self.news_sentiment_collection.add(
            ids=[article_id],
            documents=[f"{headline}\n{text}"],
            metadatas=[{
                "ticker": ticker,
                "headline": headline,
                "timestamp": timestamp,
                "sentiment_score": sentiment_score,
                "source": source,
            }],
        )

    def query_market_events(
        self,
        query: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """
        Semantic search for market events by query text.
        
        Args:
            query: Natural language query (e.g., "earnings surprise").
            ticker: Optional ticker filter.
            n_results: Number of results to return.
        
        Returns:
            ChromaDB query result dict with ids, documents, metadatas, distances.
        """
        where_filter = {"ticker": ticker} if ticker else None
        return self.market_events_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )

    def query_sec_filings(
        self,
        query: str,
        ticker: Optional[str] = None,
        filing_type: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """
        Semantic search for SEC filings by query text.
        
        Args:
            query: Natural language query (e.g., "revenue decline").
            ticker: Optional ticker filter.
            filing_type: Optional filing type filter (e.g., "8-K").
            n_results: Number of results to return.
        
        Returns:
            ChromaDB query result dict with ids, documents, metadatas, distances.
        """
        where_filter = {}
        if ticker:
            where_filter["ticker"] = ticker
        if filing_type:
            where_filter["filing_type"] = filing_type
        
        return self.sec_filings_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None,
        )

    def query_news(
        self,
        query: str,
        ticker: Optional[str] = None,
        n_results: int = 5,
    ) -> dict:
        """
        Semantic search for news articles by query text.
        
        Args:
            query: Natural language query (e.g., "product launch").
            ticker: Optional ticker filter.
            n_results: Number of results to return.
        
        Returns:
            ChromaDB query result dict with ids, documents
