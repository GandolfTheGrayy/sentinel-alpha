"""
Sentinel Historian Event Schema — dataclass definitions for market events,
historical matches, and confidence scoring across the RAG pipeline.

This module defines the core data structures used by historian/ to represent:
  - MarketEvent: A timestamped market occurrence (price move, news, filing).
  - HistoricalMatch: A past event semantically similar to a current signal.
  - ConfidenceReport: Aggregated certainty metrics for a prediction.

These schemas are consumed by rag_query.py (ChromaDB lookups) and judge/predictor.py
(final scoring), ensuring type safety and consistent metadata threading.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime


@dataclass
class MarketEvent:
    """A timestamped market occurrence with semantic and numeric metadata."""

    ticker: str
    event_type: str  # e.g., "news", "sec_filing", "price_move", "social_sentiment"
    timestamp: datetime
    headline: str
    body: Optional[str] = None
    price_impact_pct: Optional[float] = None  # observed move after event
    source_url: Optional[str] = None
    sentiment_label: Optional[str] = None  # "bullish", "bearish", "neutral"
    embedding_vector: Optional[List[float]] = None  # for ChromaDB storage
    metadata: Dict[str, any] = field(default_factory=dict)  # extensible dict

    def __post_init__(self) -> None:
        """Validate event consistency."""
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime object")
        if self.event_type not in (
            "news",
            "sec_filing",
            "price_move",
            "social_sentiment",
        ):
            raise ValueError(
                f"event_type '{self.event_type}' not in recognized set"
            )


@dataclass
class HistoricalMatch:
    """A past MarketEvent semantically similar to a current signal."""

    past_event: MarketEvent
    current_signal_id: str  # reference to the triggering signal
    semantic_similarity: float  # 0–1 cosine distance or alike
    days_to_resolution: int  # how many days until price settled post-event
    actual_price_move_pct: float  # observed outcome in the historical case
    confidence_boost: float  # how much this match increases final certainty (0–1)
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate similarity and move ranges."""
        if not (0.0 <= self.semantic_similarity <= 1.0):
            raise ValueError("semantic_similarity must be in [0, 1]")
        if not (0.0 <= self.confidence_boost <= 1.0):
            raise ValueError("confidence_boost must be in [0, 1]")
        if self.days_to_resolution < 0:
            raise ValueError("days_to_resolution cannot be negative")


@dataclass
class ConfidenceReport:
    """Aggregated certainty metrics for a single prediction."""

    ticker: str
    prediction_timestamp: datetime
    predicted_direction: str  # "bullish", "bearish", "neutral"
    raw_certainty: float  # LLM's direct confidence (0–1)
    rag_boost: float  # increase from historical matches (0–1 additive)
    final_certainty: float  # raw + RAG boost, clamped to [0, 1]
    historical_matches: List[HistoricalMatch] = field(default_factory=list)
    linguistic_signals: Dict[str, float] = field(
        default_factory=dict
    )  # e.g., {"hesitation": 0.3, "regulatory_whisper": 0.7}
    reasoning_summary: Optional[str] = None
    flag_anomaly: bool = False  # true if unusual conviction or contradiction
    metadata: Dict[str, any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate certainty ranges and direction."""
        if not (0.0 <= self.raw_certainty <= 1.0):
            raise ValueError("raw_certainty must be in [0, 1]")
        if not (0.0 <= self.rag_boost <= 1.0):
            raise ValueError("rag_boost must be in [0, 1]")
        if not (0.0 <= self.final_certainty <= 1.0):
            raise ValueError("final_certainty must be in [0, 1]")
        if self.predicted_direction not in ("bullish", "bearish", "neutral"):
            raise ValueError(
                f"predicted_direction '{self.predicted_direction}' not recognized"
            )

    def confidence_strength(self) -> str:
        """Return human-readable strength label based on final_certainty."""
        if self.final_certainty >= 0.75:
            return "very_high"
        elif self.final_certainty >= 0.6:
            return "high"
        elif self.final_certainty >= 0.4:
            return "moderate"
        else:
            return "low"


@dataclass
class EmbeddingBatch:
    """A batch of MarketEvents with embeddings ready for ChromaDB insertion."""

    events: List[MarketEvent]
    batch_id: str
    created_at: datetime
    corpus_source: str  # e.g., "sec_filings", "news_headlines", "social_sentiment"
    total_tokens_used: int = 0

    def __post_init__(self) -> None:
        """Validate batch consistency."""
        if len(self.events) == 0:
            raise ValueError("EmbeddingBatch must contain at least one event")
        if not all(isinstance(e, MarketEvent) for e in self.events):
            raise TypeError("all items in events must be MarketEvent instances")


def create_confidence_report_from_prediction(
    ticker: str,
    direction: str,
    raw_certainty: float,
    matches: List[HistoricalMatch],
    linguistic_signals: Optional[Dict[str, float]] = None,
    reasoning: Optional[str] = None,
) -> ConfidenceReport:
    """Factory function to construct a ConfidenceReport with RAG boost applied."""
    if linguistic_signals is None:
        linguistic_signals = {}

    rag_boost = (
        sum(m.confidence_boost for m in matches) / len(matches)
        if matches
        else 0.0
    )
    rag_boost = min(rag_boost, 0.3)  # cap boost at 30% to avoid over-weighting

    final_cert = min(raw_certainty + rag_boost, 1.0)

    flag_anom = False
    if raw_certainty >= 0.8 and len(matches) == 0:
        flag_anom = True  # high conviction with zero historical support

    return ConfidenceReport(
        ticker=ticker,
        prediction_timestamp=datetime.utcnow(),
        predicted_direction=direction,
        raw_certainty=raw_certainty,
        rag_boost=rag_boost,
        final_certainty=final_cert,
        historical_matches=matches,
        linguistic_signals=linguistic_signals,
        reasoning_summary=reasoning,
        flag_anomaly=flag_anom,
    )
