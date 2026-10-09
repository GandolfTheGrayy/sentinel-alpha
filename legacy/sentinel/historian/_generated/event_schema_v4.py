"""
Sentinel Historian Event Schema — Dataclasses for RAG pipeline.

Defines MarketEvent (raw historical market occurrences), HistoricalMatch
(RAG retrieval results with similarity scores), and ConfidenceReport
(structured confidence assessment for prediction synthesis).

Used by historian/rag_query.py for vector DB storage, retrieval, and
by judge/predictor.py for final scoring and prediction assembly.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class EventCategory(str, Enum):
    """Classification of historical market events for filtering and context."""
    EARNINGS = "earnings"
    SEC_FILING = "sec_filing"
    NEWS = "news"
    REGULATORY = "regulatory"
    MACRO = "macro"
    SENTIMENT_SPIKE = "sentiment_spike"
    TECHNICAL = "technical"


class ConfidenceLevel(str, Enum):
    """Confidence grade assigned by Linguist or Judge modules."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class MarketEvent:
    """
    Raw historical market event, stored in ChromaDB and indexed for RAG.

    Attributes:
        ticker: Stock symbol (e.g., 'AAPL').
        date: Event occurrence date.
        category: EventCategory enum classifying the event type.
        title: Short headline or description.
        body: Full text content (news article, SEC excerpt, sentiment summary).
        source: Data source (e.g., 'sec_edgar', 'reuters', 'reddit', 'internal').
        price_impact: Observed price movement (%) on or shortly after event date.
        embedding_vector: Pre-computed vector for similarity search (ChromaDB).
        metadata: Additional context (e.g., filing type, sentiment score, author).
    """
    ticker: str
    date: datetime
    category: EventCategory
    title: str
    body: str
    source: str
    price_impact: Optional[float] = None
    embedding_vector: Optional[List[float]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize MarketEvent to dictionary for storage."""
        return {
            "ticker": self.ticker,
            "date": self.date.isoformat(),
            "category": self.category.value,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "price_impact": self.price_impact,
            "metadata": self.metadata,
        }


@dataclass
class HistoricalMatch:
    """
    Result of a single RAG retrieval hit from ChromaDB vector search.

    Represents a historical event matched to the current prediction context,
    with similarity score and optional confidence weighting.

    Attributes:
        event: The MarketEvent retrieved from the vector DB.
        similarity_score: Cosine similarity [0, 1] between query and event embedding.
        relevance_boost: Optional multiplier applied by Historian (e.g., recency, sector).
        reasoning: Explanation of why this match is relevant.
    """
    event: MarketEvent
    similarity_score: float
    relevance_boost: float = 1.0
    reasoning: str = ""

    def effective_score(self) -> float:
        """Compute final relevance score after applying boost multiplier."""
        return self.similarity_score * self.relevance_boost


@dataclass
class ConfidenceReport:
    """
    Structured confidence assessment aggregating Linguist signals and RAG matches.

    Output by Judge.predictor and used to weight final prediction direction
    and magnitude. Chains Linguist certainty analysis with Historical precedent.

    Attributes:
        ticker: Stock symbol.
        prediction_date: Date prediction was generated.
        confidence_level: Enum grade (very_high to very_low).
        confidence_score: Numeric confidence [0, 1].
        linguist_signals: Dict of Linguist module outputs (certainty, hesitation, drift).
        historical_matches: List of matching historical events from RAG.
        predicted_direction: 'up', 'down', or 'neutral'.
        predicted_magnitude: Expected price move (%).
        reasoning_summary: Markdown summary of prediction rationale.
        anomaly_flags: List of unusual patterns that reduced confidence.
    """
    ticker: str
    prediction_date: datetime
    confidence_level: ConfidenceLevel
    confidence_score: float
    linguist_signals: dict = field(default_factory=dict)
    historical_matches: List[HistoricalMatch] = field(default_factory=list)
    predicted_direction: str = "neutral"
    predicted_magnitude: float = 0.0
    reasoning_summary: str = ""
    anomaly_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize ConfidenceReport to dictionary for logging/storage."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date.isoformat(),
            "confidence_level": self.confidence_level.value,
            "confidence_score": self.confidence_score,
            "linguist_signals": self.linguist_signals,
            "historical_matches": [
                {
                    "event": match.event.to_dict(),
                    "similarity_score": match.similarity_score,
                    "relevance_boost": match.relevance_boost,
                    "reasoning": match.reasoning,
                }
                for match in self.historical_matches
            ],
            "predicted_direction": self.predicted_direction,
            "predicted_magnitude": self.predicted_magnitude,
            "reasoning_summary": self.reasoning_summary,
            "anomaly_flags": self.anomaly_flags,
        }
</
