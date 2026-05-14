"""
Sentinel Historian Event Schema — dataclass definitions for MarketEvent,
HistoricalMatch, and ConfidenceReport.

These schemas structure the flow of historical market intelligence through the
Historian RAG pipeline: Scout injects raw events → Historian queries ChromaDB
for matches → Judge synthesizes matches into confidence-weighted predictions.

Used by: rag_query.py (historian spine), predictor.py (judge spine),
postmortem.py (judge spine).
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class MarketEvent:
    """
    A discrete market event: price move, filing, news headline, sentiment spike.
    
    Attributes:
        ticker: Stock symbol (e.g. "AAPL").
        event_type: Category ("price_move", "sec_filing", "news", "sentiment").
        timestamp: When the event occurred (UTC).
        headline: Brief event description.
        source: Where signal originated (e.g. "yfinance", "sec_edgar", "news_api").
        price_before: Stock price immediately before event (optional).
        price_after: Stock price immediately after event (optional).
        price_change_pct: Percentage change triggered by event (optional).
        raw_text: Full event body (filing excerpt, article, post).
        metadata: Arbitrary key-value context (e.g. filing_type="8-K", sentiment_score=0.75).
    """
    ticker: str
    event_type: str
    timestamp: datetime
    headline: str
    source: str
    raw_text: str
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    price_change_pct: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class HistoricalMatch:
    """
    A past MarketEvent semantically similar to a query event, retrieved from
    ChromaDB vector search.
    
    Attributes:
        event: The matched MarketEvent from history.
        similarity_score: Cosine distance [0, 1] where 1 = identical.
        days_ago: How many days in the past this match occurred.
        eventual_price_change_pct: Observed actual price change in following
                                    period (e.g., next 5 days).
        outcome_label: Classification of outcome ("bullish", "bearish", "neutral").
    """
    event: MarketEvent
    similarity_score: float
    days_ago: int
    eventual_price_change_pct: Optional[float] = None
    outcome_label: Optional[str] = None


@dataclass
class ConfidenceReport:
    """
    Aggregated confidence assessment for a single ticker's direction, synthesized
    from historical matches and linguistic signals.
    
    Attributes:
        ticker: Stock symbol being assessed.
        prediction_direction: "bullish", "bearish", or "neutral".
        confidence_score: [0, 1] where 1 = maximum conviction.
        supporting_matches: Historical matches backing this prediction.
        linguistic_certainty: Certainty score from Linguist [0, 1].
        reasoning: Human-readable summary of prediction rationale.
        event_count: Number of raw events considered.
        strongest_match_similarity: Highest similarity_score among supporting_matches.
        consensus_pct: Percentage of matches agreeing with prediction_direction.
    """
    ticker: str
    prediction_direction: str
    confidence_score: float
    supporting_matches: List[HistoricalMatch]
    linguistic_certainty: float
    reasoning: str
    event_count: int = 0
    strongest_match_similarity: float = 0.0
    consensus_pct: float = 0.0
