"""
Event schema module for Sentinel Historian layer.

Defines dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport
that structure the flow of historical market data, RAG retrieval results,
and confidence scoring across the Historian pillar and into Judge predictions.

Used by:
  - historian/rag_query.py: Returns HistoricalMatch objects from ChromaDB lookups
  - judge/predictor.py: Consumes ConfidenceReport to calibrate final predictions
  - judge/postmortem.py: Analyzes MarketEvent records for post-mortem calibration
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MarketEvent:
    """
    Represents a historical or live market event tied to a company/ticker.
    
    Used to record earnings announcements, SEC filings, news breaks, or
    sentiment spikes that may correlate with price movements.
    """
    ticker: str
    event_type: str
    timestamp: datetime
    headline: str
    source: str
    raw_content: Optional[str] = None
    relevance_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class HistoricalMatch:
    """
    Result of a RAG query against historical market events in ChromaDB.
    
    Returned by historian/rag_query.py when searching for similar past events
    that preceded known price movements. Each match includes the historical
    event, retrieval distance, and inferred signal strength.
    """
    event: MarketEvent
    similarity_score: float
    historical_return_pct: Optional[float] = None
    days_to_peak_move: Optional[int] = None
    match_context: str = ""
    confidence_basis: str = ""


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence signal for a ticker prediction.
    
    Synthesizes RAG matches, linguistic signals, and baseline heuristics
    into a structured confidence object consumed by judge/predictor.py.
    """
    ticker: str
    prediction_date: datetime
    signal_direction: str
    signal_strength: float
    linguistic_certainty: float
    rag_match_count: int
    historical_precedent_strength: float
    baseline_consensus: Optional[str] = None
    anomaly_flags: list = field(default_factory=list)
    rationale: str = ""
    
    def combined_confidence(self) -> float:
        """
        Compute weighted blend of linguistic + RAG + baseline signals.
        
        Returns: float in [0.0, 1.0] representing overall prediction confidence.
        """
        weights = {
            "linguistic": 0.40,
            "rag_match": 0.35,
            "historical": 0.25,
        }
        return (
            self.linguistic_certainty * weights["linguistic"]
            + min(self.rag_match_count / 5.0, 1.0) * weights["rag_match"]
            + self.historical_precedent_strength * weights["historical"]
        )


@dataclass
class PredictionOutcome:
    """
    Record of a prediction vs. actual market move for post-mortem calibration.
    
    Stored by judge/resolver.py and judge/postmortem.py for heuristic refinement
    and anomaly detection across the daily cohort.
    """
    ticker: str
    prediction_date: datetime
    predicted_direction: str
    predicted_confidence: float
    actual_return_pct: float
    prediction_hit: bool
    days_held: int
    confidence_report: Optional[ConfidenceReport] = None
    notes: str = ""
    metadata: dict = field(default_factory=dict)
