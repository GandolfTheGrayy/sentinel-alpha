"""
Event schema module for the Historian agent in Sentinel.

Defines core dataclasses for:
  - MarketEvent: Immutable representation of a historical market event with
    sentiment context and metadata.
  - HistoricalMatch: Result of a RAG query—a past event similar to a new signal,
    with similarity score and confidence bounds.
  - ConfidenceReport: Aggregated confidence metric across multiple matches,
    weighting by recency and event magnitude.

These schemas bridge the Scout (sentiment signals) → Historian (RAG lookup) →
Judge (backtesting) pipeline. All classes are frozen dataclasses to ensure
immutability during vector DB operations and cross-agent serialization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


class EventCategory(Enum):
    """Classification of market events for RAG bucketing and analysis."""
    EARNINGS_SURPRISE = "earnings_surprise"
    REGULATORY_FILING = "regulatory_filing"
    SENTIMENT_SHIFT = "sentiment_shift"
    DEVELOPER_SIGNAL = "developer_signal"
    SECTOR_ROTATION = "sector_rotation"
    MACRO_EVENT = "macro_event"
    ANOMALY = "anomaly"


class SentimentPolarity(Enum):
    """Polarity of extracted sentiment signal."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class MarketEvent:
    """
    Immutable representation of a historical market event with sentiment context.
    
    Used by Historian to index and retrieve similar past events. Each event
    contains the original signal, outcome, and metadata for RAG vector embedding.
    """
    event_id: str
    timestamp: datetime
    ticker: str
    category: EventCategory
    sentiment_polarity: SentimentPolarity
    signal_text: str
    signal_source: str  # e.g., "reddit/wallstreetbets", "sec/8-k", "github"
    price_move_pct: float  # Actual market response in % (outcome label)
    confidence_context: str  # Linguist's reasoning or Scout's raw data
    metadata: dict = field(default_factory=dict)  # extensible: issue_count, commit_velocity, etc.
    
    def to_embedding_text(self) -> str:
        """Render event as plaintext for vector embedding."""
        return (
            f"Event: {self.category.value}\n"
            f"Ticker: {self.ticker}\n"
            f"Polarity: {self.sentiment_polarity.value}\n"
            f"Source: {self.signal_source}\n"
            f"Signal: {self.signal_text}\n"
            f"Context: {self.confidence_context}"
        )


@dataclass(frozen=True)
class HistoricalMatch:
    """
    Result of a RAG query—a historical event similar to a new sentiment signal.
    
    Returned by Historian when searching the vector DB. Contains similarity score,
    the matched event, and confidence bounds to guide Judge's post-mortem weighting.
    """
    matched_event: MarketEvent
    similarity_score: float  # 0.0–1.0; cosine distance in vector space
    rank: int  # Ordinal position in results (0=closest match)
    price_move_direction_match: bool  # True if historical outcome aligns with current signal polarity
    days_since_event: int  # Temporal distance for decay weighting
    confidence_lower_bound: float  # Conservative estimate of prediction validity
    confidence_upper_bound: float  # Optimistic estimate
    
    def decayed_similarity(self) -> float:
        """Apply exponential time decay to similarity score (half-life: 365 days)."""
        decay_factor = 0.5 ** (self.days_since_event / 365.0)
        return self.similarity_score * decay_factor


@dataclass(frozen=True)
class ConfidenceReport:
    """
    Aggregated confidence metric across multiple historical matches.
    
    Produced by Historian after RAG query; used by Judge to calibrate prediction
    uncertainty. Incorporates recency, event magnitude, and consistency across
    multiple similar historical scenarios.
    """
    query_signal_id: str  # Link to originating sentiment signal
    query_timestamp: datetime
    ticker: str
    num_matches: int
    mean_similarity: float
    weighted_confidence: float  # 0.0–1.0; primary uncertainty metric
    confidence_lower_bound: float
    confidence_upper_bound: float
    consensus_direction: SentimentPolarity  # Most common outcome polarity across matches
    conflicting_signals_count: int  # Matches contradicting consensus
    recency_weighting: float  # Average temporal decay factor applied
    report_metadata: dict = field(default_factory=dict)  # Debug: match list, aggregation method, etc.
    
    def is_high_confidence(self, threshold: float = 0.70) -> bool:
        """Check if weighted confidence exceeds threshold for decision-making."""
        return self.weighted_confidence >= threshold
    
    def uncertainty_range(self) -> float:
        """Return width of confidence interval (upper - lower)."""
        return self.confidence_upper_bound - self.confidence_lower_bound


@dataclass
class RAGQuery:
    """
    Input structure for Historian RAG lookups.
    
    Represents a new sentiment signal from Scout ready for historical matching.
    """
    signal_id: str
    ticker: str
    signal_text: str
    signal_source: str
    sentiment_polarity: SentimentPolarity
    category_hint: Optional[EventCategory] = None
    top_k: int = 5  # Number of historical matches to retrieve
    query_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_embedding_text(self) -> str:
        """Render query as plaintext for vector embedding lookup."""
        return (
            f"Query: {self.category_hint.value if self.category_hint else 'unknown'}\n"
            f"Ticker: {self.ticker}\n"
            f"Polarity: {self.sentiment_polarity.value}\n"
            f"Source: {self.signal_source}\n"
            f"Signal: {self.signal_text}"
        )


@dataclass(frozen=True)
class PostMortemEntry:
    """
    Single row in Judge's backtest post-mortem log.
    
    Records predicted vs. actual outcomes for heuristic refinement.
    """
    date: datetime
    ticker: str
    predicted_direction: SentimentPolarity
    predicted_confidence: float
    actual_price_move_pct: float
    correct_prediction: bool
    historical_match_count: int
    top_match_similarity: Optional[float]
    notes: str = ""
