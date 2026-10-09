"""
Event schema module for Sentinel Historian layer.

Defines dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport
used across RAG pipeline, historical event lookup, and confidence score weighting.
These schemas enable structured representation of market signals, historical
precedents, and confidence metrics for downstream Judge reasoning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class EventSeverity(str, Enum):
    """Enumeration of market event severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NOISE = "noise"


class EventCategory(str, Enum):
    """Enumeration of market event categories."""
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    ACQUISITION = "acquisition"
    EXECUTIVE_CHANGE = "executive_change"
    LITIGATION = "litigation"
    MARKET_MACRO = "market_macro"
    SENTIMENT_SHIFT = "sentiment_shift"
    TECHNICAL = "technical"
    OTHER = "other"


@dataclass
class MarketEvent:
    """
    Represents a discrete market event signal detected from scraped sources.
    
    Attributes:
        event_id: Unique identifier for the event.
        ticker: Stock ticker symbol.
        timestamp: ISO datetime of event occurrence or announcement.
        category: EventCategory enum classifying the event type.
        severity: EventSeverity enum rating impact magnitude.
        title: Short headline or summary.
        description: Extended details extracted from source.
        source: Origin of signal (e.g., "SEC_8K", "REDDIT", "NEWS", "GITHUB").
        source_url: Link to primary source document.
        sentiment_polarity: Float [-1.0, 1.0] for direction (negative to positive).
        confidence: Float [0.0, 1.0] for extraction reliability.
        metadata: Dict for free-form extended attributes.
    """
    event_id: str
    ticker: str
    timestamp: datetime
    category: EventCategory
    severity: EventSeverity
    title: str
    description: str
    source: str
    source_url: str
    sentiment_polarity: float
    confidence: float
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate field ranges and types."""
        if not (-1.0 <= self.sentiment_polarity <= 1.0):
            raise ValueError(f"sentiment_polarity must be in [-1.0, 1.0], got {self.sentiment_polarity}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass
class HistoricalMatch:
    """
    Represents a historical precedent retrieved from RAG vector DB.
    
    Attributes:
        match_id: Unique identifier for the historical record.
        ticker: Stock ticker of the historical event.
        event_date: Date of the historical event.
        reference_event: The MarketEvent or description being matched.
        similarity_score: Float [0.0, 1.0] from vector similarity.
        price_movement_pct: Observed price change in % following the historical event.
        lookback_days: Number of days after event for price_movement_pct measurement.
        outcome_category: Categorical outcome (e.g., "BULLISH", "BEARISH", "NEUTRAL").
        notes: Analyst notes or context from historical analysis.
        embedding_model: Name of embedding model used for similarity (e.g., "gemini-3.1").
    """
    match_id: str
    ticker: str
    event_date: datetime
    reference_event: str
    similarity_score: float
    price_movement_pct: float
    lookback_days: int
    outcome_category: str
    notes: str
    embedding_model: str

    def __post_init__(self) -> None:
        """Validate similarity_score range."""
        if not (0.0 <= self.similarity_score <= 1.0):
            raise ValueError(f"similarity_score must be in [0.0, 1.0], got {self.similarity_score}")


@dataclass
class ConfidenceReport:
    """
    Aggregates confidence signals for a ticker prediction.
    
    Synthesized by Historian from MarketEvent signals, HistoricalMatch
    precedents, and linguistic analysis for use by Judge in final scoring.
    
    Attributes:
        ticker: Stock ticker.
        report_date: Date of confidence calculation.
        event_count: Number of active signals in prediction window.
        avg_event_severity: Average EventSeverity score (0.0–1.0).
        sentiment_aggregate: Weighted sentiment [-1.0, 1.0].
        historical_precedent_count: Number of matching historical events.
        avg_historical_outcome: Mean price_movement_pct from matches.
        historical_win_rate: Fraction [0.0, 1.0] of positive precedent outcomes.
        linguistic_certainty_score: Float [0.0, 1.0] from Linguist analysis.
        linguistic_drift_signal: Boolean; True if tone shift detected.
        regulatory_whisper_flag: Boolean; True if regulatory risk detected.
        overall_confidence: Composite confidence [0.0, 1.0] for prediction.
        reasoning_summary: Human-readable explanation of confidence.
        sources_cited: List of source identifiers backing the report.
    """
    ticker: str
    report_date: datetime
    event_count: int
    avg_event_severity: float
    sentiment_aggregate: float
    historical_precedent_count: int
    avg_historical_outcome: float
    historical_win_rate: float
    linguistic_certainty_score: float
    linguistic_drift_signal: bool
    regulatory_whisper_flag: bool
    overall_confidence: float
    reasoning_summary: str
    sources_cited: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate confidence and sentiment ranges."""
        if not (0.0 <= self.avg_event_severity <= 1.0):
            raise ValueError(f"avg_event_severity must be in [0.0, 1.0], got {self.avg_event_severity}")
        if not (-1.0 <= self.sentiment_aggregate <= 1.0):
            raise ValueError(f"sentiment_aggregate must be in [-1.0, 1.0], got {self.sentiment_aggregate}")
        if not (0.0 <= self.historical_win_rate <= 1.0):
            raise ValueError(f"historical_win_rate must be in [0.0, 1.0], got {self.historical_win_rate}")
        if not (0.0 <= self.linguistic_certainty_score <= 1.0):
            raise ValueError(f"linguistic_certainty_score must be in [0.0, 1.0], got {self.linguistic_certainty_score}")
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(f"overall_confidence must be in [0.0, 1.0], got {self.overall_confidence}")


def create_market_event(
    ticker: str,
    category: EventCategory,
    severity: EventSeverity,
    title: str,
    description: str,
    source: str,
    source_url: str,
    sentiment_polarity: float = 0.0,
    confidence: float = 0.8,
    metadata: Optional[dict] = None,
) -> MarketEvent:
    """Factory function to create a MarketEvent with auto-generated ID and timestamp."""
    from uuid import uuid4
    event_id = f"evt_{uuid4().hex[:12]}"
    timestamp = datetime.utcnow()
    return MarketEvent(
        event_id=event_id,
        ticker=ticker,
        timestamp=timestamp,
        category=category,
        severity=severity,
        title=title,
        description=description,
        source=source,
        source_url
