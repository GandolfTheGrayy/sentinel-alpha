"""
Sentinel Historian Event Schema — Dataclass definitions for MarketEvent,
HistoricalMatch, and ConfidenceReport.

This module provides the core data structures used throughout the Historian layer
to represent market events, historical pattern matches discovered via RAG, and
confidence scoring metadata. These schemas bridge the Scout ingestion layer,
RAG vector lookups, and Judge prediction synthesis.

Usage:
  - Scout modules serialize market signals into MarketEvent instances.
  - RAG query pipeline populates HistoricalMatch objects with retrieved context.
  - Judge predictor reads ConfidenceReport to calibrate final price predictions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class EventType(str, Enum):
    """Enumeration of market event categories recognized by Sentinel."""

    SEC_FILING = "sec_filing"
    NEWS_HEADLINE = "news_headline"
    REDDIT_POST = "reddit_post"
    GITHUB_SIGNAL = "github_signal"
    PRICE_MOVEMENT = "price_movement"
    EARNINGS_ANNOUNCEMENT = "earnings_announcement"
    REGULATORY_FILING = "regulatory_filing"


class SentimentPolarity(str, Enum):
    """Sentiment polarity classification for event text."""

    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    MIXED = "mixed"


class ConfidenceLevel(str, Enum):
    """Discrete confidence bands for prediction certainty."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class MarketEvent:
    """
    A raw market signal ingested by Scout modules.

    Represents a single data point (price tick, news headline, SEC filing, etc.)
    that feeds into the Historian RAG pipeline and Judge reasoning.
    """

    event_id: str
    """Unique identifier for this event (e.g., sha256 hash of content)."""

    ticker: str
    """Stock ticker symbol (e.g., 'TSLA', 'AAPL')."""

    event_type: EventType
    """Classification of the signal source."""

    timestamp: datetime
    """When the event occurred or was published."""

    title: str
    """Short headline or summary of the event."""

    body: str
    """Full text content of the signal."""

    source_url: Optional[str] = None
    """URL origin if applicable (news link, SEC EDGAR URL, etc.)."""

    metadata: dict = field(default_factory=dict)
    """
    Additional context:
      - 'score': raw sentiment score (float, -1.0 to 1.0)
      - 'filing_type': '8-K', '10-Q', '10-K' for SEC events
      - 'subreddit': subreddit name for Reddit posts
      - 'author': content creator for attribution
    """

    def __post_init__(self) -> None:
        """Validate required fields are non-empty."""
        if not self.ticker or not self.title or not self.body:
            raise ValueError(
                "MarketEvent requires non-empty ticker, title, and body"
            )


@dataclass
class HistoricalMatch:
    """
    A historical precedent retrieved via RAG similarity search.

    Represents a prior market event that is semantically or linguistically
    similar to a current signal, used by Judge to contextualize predictions.
    """

    match_id: str
    """Unique identifier for this historical match record."""

    reference_event: MarketEvent
    """The original historical market event matched."""

    query_event: MarketEvent
    """The current event that triggered the similarity search."""

    similarity_score: float
    """
    Cosine similarity or embedding distance (0.0 to 1.0, higher = more similar).
    Set by RAG vector DB lookup.
    """

    matched_date: datetime
    """When the historical event originally occurred."""

    days_to_resolution: Optional[int] = None
    """
    Number of days between matched event and observable market outcome.
    None if outcome is not yet known.
    """

    outcome_return_pct: Optional[float] = None
    """
    Observed price movement (%) in the days following the historical event.
    None if not yet resolved. Positive = up, negative = down.
    """

    precedent_type: str = ""
    """
    Category label (e.g., 'product_launch', 'regulatory_action', 'earnings_beat').
    Aids Judge in pattern classification.
    """

    metadata: dict = field(default_factory=dict)
    """
    Additional match metadata:
      - 'vector_db_collection': ChromaDB collection name
      - 'retrieval_timestamp': when match was retrieved
      - 'match_reasoning': free-form explanation of why match was relevant
    """

    def __post_init__(self) -> None:
        """Validate similarity score is in valid range."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(
                f"similarity_score must be in [0.0, 1.0], got {self.similarity_score}"
            )


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence metadata for a price prediction.

    Produced by Judge predictor; summarizes the certainty, supporting evidence,
    and risk factors behind a stock price forecast for a specific ticker
    and time horizon.
    """

    ticker: str
    """Stock ticker being predicted."""

    prediction_horizon_days: int
    """How many days ahead the prediction extends."""

    predicted_direction: str
    """'up', 'down', or 'neutral'."""

    confidence_level: ConfidenceLevel
    """Discrete confidence band (VERY_HIGH, HIGH, MODERATE, LOW, VERY_LOW)."""

    confidence_score: float
    """
    Numeric confidence (0.0 to 1.0).
    Derived from model disagreement, historical precedent alignment, and sentiment coherence.
    """

    supporting_events: List[MarketEvent] = field(default_factory=list)
    """Raw market signals that informed the prediction."""

    historical_matches: List[HistoricalMatch] = field(default_factory=list)
    """Retrieved precedents from RAG used in reasoning."""

    primary_reasoning: str = ""
    """Free-form summary of the main bullish/bearish thesis."""

    risk_factors: List[str] = field(default_factory=list)
    """List of downside or uncertainty scenarios (e.g., 'earnings miss risk')."""

    generated_at: datetime = field(default_factory=datetime.utcnow)
    """Timestamp when this report was generated."""

    metadata: dict = field(default_factory=dict)
    """
    Additional reasoning metadata:
      - 'model_agreement': % of base classifiers voting for direction
      - 'sentiment_polarity_distribution': histogram of polarities
      - 'historical_hit_rate': accuracy of matched precedents on this ticker
      - 'linguistic_markers': list of detected certainty/hesitation tokens
    """

    def __post_init__(self) -> None:
        """Validate confidence score is in valid range."""
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValueError(
                f"confidence_score must be in [0.0, 1.0], got {self.confidence_score}"
            )
        if self.prediction_horizon_days <= 0:
            raise ValueError(
                f"prediction_horizon_days must be positive, got {self.prediction_horizon_days}"
            )
        if self.predicted_direction not in ("up", "down", "neutral"):
            raise ValueError(
                f"predicted_direction must be 'up', 'down', or 'neutral', "
                f"got {self.predicted_direction}"
            )


@dataclass
class MarketOutcome:
    """
    Actual observed market outcome after a prediction period.

    Used by Judge post-mortem to compare predictions against ground truth
    and refine confidence calibration.
    """

    ticker: str
    """Stock ticker."""

    start_date: datetime
    """Date
