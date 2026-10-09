"""
RAG query interface for Sentinel Historian.

Given a SentimentResidual (current sentiment signal), queries ChromaDB for
the top-k most similar historical events and returns a ranked list of
HistoricalMatch objects. Bridges live sentiment analysis (from Linguist)
with historical precedent lookup to contextualize predictions.

Uses Gemini embeddings (via google-generativeai) to vectorize queries and
retrieve semantically similar past market events from the persistent ChromaDB.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, List

import chromadb
from google.generativeai import embed_content


@dataclass
class SentimentResidual:
    """Input: current sentiment signal to match against history."""
    ticker: str
    headline: str
    sentiment_score: float  # -1.0 to 1.0
    source: str  # "news", "reddit", "sec_filing", etc.
    timestamp: str  # ISO 8601
    context: Optional[str] = None  # Additional free-form context


@dataclass
class HistoricalMatch:
    """Output: a matched historical event with relevance score."""
    ticker: str
    event_headline: str
    event_timestamp: str
    event_sentiment_score: float
    similarity_score: float  # 0.0 to 1.0, from ChromaDB
    outcome_direction: str  # "up", "down", "neutral"
    outcome_magnitude: float  # e.g., 0.05 for +5% move
    days_to_outcome: int  # how many days until the price move resolved


class HistorianRAG:
    """
    ChromaDB-backed RAG engine for historical event retrieval.
    
    Manages vectorization, collection upsert, and similarity queries
    using Gemini embeddings.
    """

    def __init__(self, db_path: str = "./sentinel_history.db"):
        """
        Initialize ChromaDB client and load or create collections.
        
        Args:
            db_path: Path to persistent ChromaDB storage.
        """
        self.db_path = db_path
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Collections: one per data type for flexible querying
        self.news_collection = self.client.get_or_create_collection(
            name="historical_news",
            metadata={"description": "Historical news headlines with outcomes"}
        )
        self.sec_collection = self.client.get_or_create_collection(
            name="historical_sec",
            metadata={"description": "Historical SEC filings with outcomes"}
        )
        self.reddit_collection = self.client.get_or_create_collection(
            name="historical_reddit",
            metadata={"description": "Historical Reddit sentiment with outcomes"}
        )
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

    def _embed_text(self, text: str) -> List[float]:
        """
        Embed text using Gemini's embedding API.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        result = embed_content(
            model="models/embedding-001",
            content=text,
            api_key=self.gemini_api_key
        )
        return result["embedding"]

    def upsert_historical_event(
        self,
        ticker: str,
        source: str,
        headline: str,
        timestamp: str,
        sentiment_score: float,
        outcome_direction: str,
        outcome_magnitude: float,
        days_to_outcome: int,
        event_id: Optional[str] = None
    ) -> None:
        """
        Add or update a historical event in the appropriate collection.
        
        Args:
            ticker: Stock ticker.
            source: Data source ("news", "sec_filing", "reddit").
            headline: Event headline/description.
            timestamp: ISO 8601 timestamp of event.
            sentiment_score: Sentiment at time of event (-1.0 to 1.0).
            outcome_direction: Price move direction ("up", "down", "neutral").
            outcome_magnitude: Magnitude of price move (e.g., 0.05 for +5%).
            days_to_outcome: Days until price move resolved.
            event_id: Optional custom ID; auto-generated if None.
        """
        if not event_id:
            event_id = f"{ticker}_{source}_{timestamp}".replace(":", "-").replace(".", "-")
        
        embedding = self._embed_text(headline)
        
        metadata = {
            "ticker": ticker,
            "source": source,
            "timestamp": timestamp,
            "sentiment_score": sentiment_score,
            "outcome_direction": outcome_direction,
            "outcome_magnitude": outcome_magnitude,
            "days_to_outcome": days_to_outcome
        }
        
        # Route to correct collection
        if source == "news":
            collection = self.news_collection
        elif source == "sec_filing":
            collection = self.sec_collection
        elif source == "reddit":
            collection = self.reddit_collection
        else:
            # Default to news for unknown sources
            collection = self.news_collection
        
        collection.upsert(
            ids=[event_id],
            embeddings=[embedding],
            documents=[headline],
            metadatas=[metadata]
        )

    def query_historical_matches(
        self,
        residual: SentimentResidual,
        top_k: int = 5,
        include_sources: Optional[List[str]] = None
    ) -> List[HistoricalMatch]:
        """
        Query ChromaDB for top-k historical events similar to current residual.
        
        Args:
            residual: Current SentimentResidual to match.
            top_k: Number of historical matches to return.
            include_sources: Optional list of sources to filter by (e.g., ["news", "sec_filing"]).
                            If None, queries all collections.
        
        Returns:
            Ranked list of HistoricalMatch objects, sorted by similarity (highest first).
        """
        query_embedding = self._embed_text(residual.headline)
        
        if include_sources is None:
            include_sources = ["news", "sec_filing", "reddit"]
        
        # Query each enabled collection
        all_matches = []
        for source in include_sources:
            if source == "news":
                collection = self.news_collection
            elif source == "sec_filing":
                collection = self.sec_collection
            elif source == "reddit":
                collection = self.reddit_collection
            else:
                continue
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"ticker": residual.ticker} if residual.ticker != "*" else None
            )
            
            # Unpack results
            if results and results["ids"] and len(results["ids"]) > 0:
                for idx, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][idx]
                    distance = results["distances"][0][idx] if results["distances"] else 0.0
                    # Convert distance to similarity (0 = perfect match, 1 = dissimilar)
                    similarity = 1.0 - min(distance / 2.0, 1.0)
                    
                    match = HistoricalMatch(
                        ticker=metadata.get("ticker", residual.ticker),
                        event_headline=results["documents"][0][idx] if results["documents"] else "",
                        event_timestamp=metadata.get("timestamp", ""),
                        event_sentiment_score=float(metadata.get("sentiment_score", 0.0)),
                        similarity_score=similarity,
                        outcome_direction=metadata.get("outcome_direction", "neutral"),
                        outcome_magnitude=float(metadata.get("outcome_magnitude", 0.0)),
                        days_to_outcome=int(metadata.get("
