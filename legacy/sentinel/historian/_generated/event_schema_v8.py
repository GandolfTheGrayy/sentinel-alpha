"""
Sentinel Historian Event Schema — dataclass definitions for market events, historical matches, and confidence scoring.

This module defines the core data structures used throughout the Historian layer:
  - MarketEvent: A discrete market signal (price move, news, filing, sentiment spike)
  - HistoricalMatch: A past event matched to current signals via RAG similarity
  - ConfidenceReport: Aggregated confidence score with component breakdown and weighting

Used by rag_query.py (RAG pipeline), judge/predictor.py (final scoring), and judge/postmortem.py (analysis).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class MarketEvent:
    """
    A discrete, timestamped market signal or observation.
    
    Represents any actionable market event: price movement, news headline,
    SEC filing disclosure, sentiment spike, or developer activity signal.
    """
    ticker: str
    event_type: str  # "price_move", "news", "sec_filing", "sentiment_spike", "github_signal"
    timestamp: datetime
    headline: str
    raw_text: Optional[str] = None
    source: Optional[str] = None  # "yfinance", "newsapi", "sec_edgar", "reddit", "github"
    magnitude: Optional[float] = None  # percentage change, sentiment score, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Hash for deduplication in sets."""
        return hash((self.ticker, self.event_type, self.timestamp, self.headline))


@dataclass
class HistoricalMatch:
    """
    A historical market event matched to current signals via vector similarity.
    
    Returned by RAG queries; represents a past event semantically similar to
    current market conditions. Used to inform confidence scoring and anomaly detection.
    """
    historical_event: MarketEvent
    similarity_score: float  # 0.0 to 1.0, from vector embedding cosine distance
    days_ago: int  # how many days in the past this event occurred
    outcome: Optional[str] = None  # "up", "down", "neutral" — historical result
    outcome_magnitude: Optional[float] = None  # % move post-event
    relevance_notes: str = ""  # human-readable context from RAG lookup

    def recency_weight(self) -> float:
        """Compute exponential decay weight favoring recent matches."""
        return max(0.1, 1.0 - (self.days_ago / 365.0))

    def combined_score(self) -> float:
        """Weighted combination of similarity and recency."""
        return (self.similarity_score * 0.7) + (self.recency_weight() * 0.3)


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence score for a market prediction with component breakdown.
    
    Synthesizes signals from sentiment analysis, RAG matches, baseline strategies,
    and historical volatility to produce a final prediction confidence (0–100%).
    Used by judge/predictor.py to finalize daily predictions.
    """
    ticker: str
    prediction_date: datetime
    direction: str  # "up", "down", "neutral"
    confidence_percent: float  # 0–100
    
    # Component scores (0–100, linearly combined or weighted)
    sentiment_score: float
    linguistic_drift_score: Optional[float] = None
    rag_historical_score: float = 0.0
    baseline_consensus_score: float = 0.0
    
    # Supporting evidence
    matching_historical_events: List[HistoricalMatch] = field(default_factory=list)
    key_signals: List[str] = field(default_factory=list)  # ["bullish_sentiment", "sec_filing_positive", ...]
    risk_flags: List[str] = field(default_factory=list)  # ["high_volatility", "earnings_imminent", ...]
    
    # Metadata
    model_version: str = "sentinel-1.0"
    reasoning_summary: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def is_high_conviction(self) -> bool:
        """True if confidence exceeds 70%."""
        return self.confidence_percent >= 70.0

    def is_low_conviction(self) -> bool:
        """True if confidence is between 40–60% (near-neutral, risky to trade)."""
        return 40.0 <= self.confidence_percent <= 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date.isoformat(),
            "direction": self.direction,
            "confidence_percent": self.confidence_percent,
            "sentiment_score": self.sentiment_score,
            "linguistic_drift_score": self.linguistic_drift_score,
            "rag_historical_score": self.rag_historical_score,
            "baseline_consensus_score": self.baseline_consensus_score,
            "key_signals": self.key_signals,
            "risk_flags": self.risk_flags,
            "is_high_conviction": self.is_high_conviction(),
            "is_low_conviction": self.is_low_conviction(),
            "model_version": self.model_version,
            "reasoning_summary": self.reasoning_summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RAGQueryResult:
    """
    Result of a single RAG query — vector similarity search in ChromaDB.
    
    Wraps the raw vector DB hits and converts them to HistoricalMatch objects
    for downstream confidence scoring and post-mortem analysis.
    """
    query_text: str
    query_embedding: Optional[List[float]] = None
    matches: List[HistoricalMatch] = field(default_factory=list)
    query_timestamp: datetime = field(default_factory=datetime.utcnow)
    database_size: int = 0  # number of documents in ChromaDB at query time

    def best_match(self) -> Optional[HistoricalMatch]:
        """Return highest-scoring historical match, or None if empty."""
        return max(self.matches, key=lambda m: m.combined_score()) if self.matches else None

    def top_k(self, k: int = 5) -> List[HistoricalMatch]:
        """Return top-k matches by combined score."""
        return sorted(self.matches, key=lambda m: m.combined_score(), reverse=True)[:k]
