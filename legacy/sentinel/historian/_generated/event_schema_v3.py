"""
Event schema module for the Historian pillar of Sentinel Sentiment Engine.

Defines dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport that
represent events, historical matches from RAG lookups, and confidence scoring used
across the Historian layer for temporal event correlation and confidence weighting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class MarketEvent:
    """
    Represents a discrete market-moving event ingested from Scout sources.
    
    Attributes:
        event_id: Unique identifier for the event.
        ticker: Stock ticker symbol.
        event_type: Classification (e.g., "earnings", "sec_filing", "news", "sentiment_spike").
        timestamp: UTC datetime when event occurred or was published.
        source: Origin of the event (e.g., "sec_edgar", "reddit", "news_api").
        headline: Brief title or summary text.
        body: Full text content if available.
        url: Source URL if applicable.
        metadata: Additional unstructured data (e.g., filing type, subreddit, news source).
        raw_sentiment_score: Initial sentiment estimate from Scout (range -1.0 to 1.0).
    """
    event_id: str
    ticker: str
    event_type: str
    timestamp: datetime
    source: str
    headline: str
    body: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_sentiment_score: float = 0.0


@dataclass
class HistoricalMatch:
    """
    Represents a historical event retrieved via RAG similarity search.
    
    Attributes:
        match_id: Unique identifier for this match record.
        query_event: The current event used as the query.
        historical_event: The similar historical event found in the corpus.
        similarity_score: Cosine similarity (0.0 to 1.0) from embedding comparison.
        temporal_gap_days: Number of days between query and historical event.
        market_outcome_direction: Direction of price move after historical event ("up", "down", "flat").
        market_outcome_magnitude: Absolute percentage price change observed.
        time_to_outcome_days: Days from historical event to observed outcome.
        context_notes: Qualitative notes on why this match is relevant.
    """
    match_id: str
    query_event: MarketEvent
    historical_event: MarketEvent
    similarity_score: float
    temporal_gap_days: int
    market_outcome_direction: str
    market_outcome_magnitude: float
    time_to_outcome_days: int
    context_notes: str = ""


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence scoring for a prediction on a single ticker.
    
    Attributes:
        ticker: Stock ticker symbol.
        prediction_timestamp: When the prediction was generated.
        predicted_direction: Predicted price direction ("up", "down", "neutral").
        predicted_magnitude: Expected magnitude of move in percentage points.
        base_confidence: Confidence from Linguist certainty analysis (0.0 to 1.0).
        historical_alignment_score: How well historical matches support prediction (0.0 to 1.0).
        historical_match_count: Number of relevant historical precedents found.
        regulatory_signal_strength: Confidence boost from "Regulatory Whispers" (0.0 to 1.0).
        sentiment_consensus: Cross-source sentiment agreement (0.0 to 1.0).
        combined_confidence: Final weighted confidence score (0.0 to 1.0).
        reasoning_summary: Concise explanation of confidence composition.
        contributing_matches: List of HistoricalMatch objects that influenced score.
        caveats: List of risk factors or uncertainty notes.
    """
    ticker: str
    prediction_timestamp: datetime
    predicted_direction: str
    predicted_magnitude: float
    base_confidence: float
    historical_alignment_score: float
    historical_match_count: int
    regulatory_signal_strength: float
    sentiment_consensus: float
    combined_confidence: float
    reasoning_summary: str
    contributing_matches: List[HistoricalMatch] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ConfidenceReport to dictionary, excluding nested dataclass lists for JSON serialization."""
        return {
            "ticker": self.ticker,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "predicted_direction": self.predicted_direction,
            "predicted_magnitude": self.predicted_magnitude,
            "base_confidence": self.base_confidence,
            "historical_alignment_score": self.historical_alignment_score,
            "historical_match_count": self.historical_match_count,
            "regulatory_signal_strength": self.regulatory_signal_strength,
            "sentiment_consensus": self.sentiment_consensus,
            "combined_confidence": self.combined_confidence,
            "reasoning_summary": self.reasoning_summary,
            "match_count": len(self.contributing_matches),
            "caveats": self.caveats,
        }


@dataclass
class EventCorpusMetadata:
    """
    Metadata about the historical event corpus in ChromaDB.
    
    Attributes:
        corpus_id: Identifier for this corpus version.
        created_at: When the corpus was initialized.
        last_updated: Last time events were added or indexed.
        total_events: Total number of events in the corpus.
        ticker_coverage: Dict mapping tickers to event counts.
        embedding_model: Name of the embedding model used.
        note: Optional description of corpus contents or curation.
    """
    corpus_id: str
    created_at: datetime
    last_updated: datetime
    total_events: int
    ticker_coverage: Dict[str, int] = field(default_factory=dict)
    embedding_model: str = "gemini-embedding-004"
    note: str = ""
