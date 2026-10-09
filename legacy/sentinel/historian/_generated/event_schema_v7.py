"""
Event schema module for Sentinel Historian layer.

Defines dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport
that standardize event representation across RAG pipelines, historical lookups,
and confidence scoring. These schemas enable structured cross-referencing of
market signals with historical precedent and calibration metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class EventCategory(str, Enum):
    """Classification of market events for filtering and correlation."""
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    INSIDER_TRADING = "insider_trading"
    ACQUISITION = "acquisition"
    BANKRUPTCY = "bankruptcy"
    MARKET_CRASH = "market_crash"
    SENTIMENT_SPIKE = "sentiment_spike"
    TECHNICAL_BREAKOUT = "technical_breakout"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    """Confidence tiers for historical match quality."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class MarketEvent:
    """
    Represents a discrete market event for historical reference.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL").
        event_date: When the event occurred.
        category: EventCategory enum for classification.
        title: Human-readable event summary.
        description: Extended details about the event.
        source: Origin (e.g., "SEC_EDGAR", "NEWS", "REDDIT", "GITHUB").
        price_before: Stock price at event start (optional).
        price_after: Stock price at event end window (optional).
        price_change_pct: Percentage change from before to after (optional).
        volume_impact: Relative trading volume change (optional).
        embedding_vector: Pre-computed embedding for RAG retrieval (optional).
        metadata: Arbitrary additional fields (regulatory_body, analyst_rating, etc.).
    """
    ticker: str
    event_date: datetime
    category: EventCategory
    title: str
    description: str
    source: str
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    price_change_pct: Optional[float] = None
    volume_impact: Optional[float] = None
    embedding_vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize MarketEvent to dictionary for storage or transmission."""
        return {
            "ticker": self.ticker,
            "event_date": self.event_date.isoformat(),
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "price_before": self.price_before,
            "price_after": self.price_after,
            "price_change_pct": self.price_change_pct,
            "volume_impact": self.volume_impact,
            "embedding_vector": self.embedding_vector,
            "metadata": self.metadata,
        }


@dataclass
class HistoricalMatch:
    """
    Represents a historical event that correlates with current signals.
    
    Used by RAG to link today's sentiment or technical moves to precedent.
    
    Attributes:
        reference_event: The historical MarketEvent this match points to.
        similarity_score: 0.0-1.0 cosine similarity or semantic closeness.
        reason: Why this historical event is relevant (e.g., "sentiment_magnitude").
        days_since: Calendar distance from reference to present (for decay weighting).
        outcome_summary: What happened to price after the reference event.
        confidence: ConfidenceLevel enum for match quality.
    """
    reference_event: MarketEvent
    similarity_score: float
    reason: str
    days_since: int
    outcome_summary: str
    confidence: ConfidenceLevel

    def to_dict(self) -> Dict[str, Any]:
        """Serialize HistoricalMatch to dictionary for logging and reporting."""
        return {
            "reference_event": self.reference_event.to_dict(),
            "similarity_score": self.similarity_score,
            "reason": self.reason,
            "days_since": self.days_since,
            "outcome_summary": self.outcome_summary,
            "confidence": self.confidence.value,
        }


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence metadata for a prediction or signal.
    
    Combines historical precedent, linguistic certainty, and signal concordance
    into a structured confidence assessment.
    
    Attributes:
        ticker: Stock symbol being analyzed.
        prediction_date: When this report was generated.
        base_confidence: 0.0-1.0 prior confidence from historical patterns.
        linguistic_adjustment: -1.0 to 1.0 adjustment based on tone/certainty.
        signal_concordance: 0.0-1.0 agreement across sentiment/technical/fundamental.
        final_confidence: Composite 0.0-1.0 confidence (base + adjustments, clamped).
        supporting_matches: List of HistoricalMatch objects backing the prediction.
        conflicting_signals: List of reasons confidence is lowered (for transparency).
        recommendation: "BUY", "SELL", "HOLD", or "UNCERTAIN".
        expected_move_pct: Point estimate of 5-day price move (optional).
        move_distribution: (percentile_5, median, percentile_95) range (optional).
    """
    ticker: str
    prediction_date: datetime
    base_confidence: float
    linguistic_adjustment: float
    signal_concordance: float
    final_confidence: float
    supporting_matches: List[HistoricalMatch] = field(default_factory=list)
    conflicting_signals: List[str] = field(default_factory=list)
    recommendation: str = "UNCERTAIN"
    expected_move_pct: Optional[float] = None
    move_distribution: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ConfidenceReport to dictionary for persistence and rendering."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date.isoformat(),
            "base_confidence": self.base_confidence,
            "linguistic_adjustment": self.linguistic_adjustment,
            "signal_concordance": self.signal_concordance,
            "final_confidence": self.final_confidence,
            "supporting_matches": [m.to_dict() for m in self.supporting_matches],
            "conflicting_signals": self.conflicting_signals,
            "recommendation": self.recommendation,
            "expected_move_pct": self.expected_move_pct,
            "move_distribution": self.move_distribution,
        }


def clamp_confidence(value: float) -> float:
    """Clamp confidence score to [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def compute_composite_confidence(
    base: float,
    linguistic_adj: float,
    concordance: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute weighted composite confidence from components.
    
    Args:
        base: Historical base confidence [0, 1].
        linguistic_adj: Linguistic adjustment [-1, 1].
        concordance: Signal concordance [0, 1].
        weights: Optional dict {"base": w1, "linguistic": w2, "concordance": w3}.
                 Defaults to equal weighting if not provided.
    
    Returns:
        Clamped composite confidence in [0, 1].
    """
    if weights is None:
        weights = {"base": 0.33, "linguistic": 0.33, "concordance": 0.34}
    
    adjusted_base = clamp_confidence(base + linguistic_adj)
    composite = (
        weights.get("base
