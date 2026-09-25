"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout signals (price momentum, news sentiment, social sentiment, developer health)
and Linguist scores (certainty, hesitation, linguistic drift, regulatory whispers) into a
composite SentimentResidual score using weighted formulas. This module synthesizes multi-source
evidence into a single normalized sentiment metric (-1.0 to +1.0) that feeds into Judge predictions.

The SentimentResidual accounts for:
  - Directional bias (bullish vs. bearish)
  - Confidence level (how certain the sentiment is)
  - Signal alignment (do Scout and Linguist agree?)
  - Historical precedent weighting (via Historian confidence)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ScoutSignals:
    """Raw signals from Scout ingestion layer."""
    
    price_momentum: float  # -1.0 (bearish) to +1.0 (bullish)
    price_momentum_confidence: float  # 0.0 to 1.0
    news_sentiment: float  # -1.0 to +1.0
    news_sentiment_count: int  # number of articles aggregated
    social_sentiment: float  # -1.0 to +1.0 (Reddit, HN, Twitter)
    social_sentiment_count: int  # engagement volume
    developer_health: float  # -1.0 (declining) to +1.0 (growing) for tech stocks
    developer_health_confidence: float  # 0.0 to 1.0


@dataclass
class LinguistScores:
    """Normalized scores from Linguist reasoning layer."""
    
    certainty_score: float  # 0.0 (hedging/uncertain) to 1.0 (definitive)
    hesitation_markers: float  # 0.0 (confident tone) to 1.0 (lots of "maybe", "could")
    linguistic_drift: float  # -1.0 (tone shifted negative) to +1.0 (shifted positive)
    linguistic_drift_confidence: float  # 0.0 to 1.0
    regulatory_whispers: float  # -1.0 (negative reg signals) to +1.0 (positive), 0.0 (neutral)
    regulatory_whispers_confidence: float  # 0.0 to 1.0


@dataclass
class HistorianContext:
    """Historical weighting and precedent signals from Historian/RAG layer."""
    
    similar_event_count: int  # how many historical precedents found
    historical_accuracy: float  # 0.0 to 1.0, based on past similar events
    base_volatility: float  # historical std dev of daily returns
    sector_baseline_sentiment: float  # -1.0 to +1.0, sector average sentiment


@dataclass
class SentimentResidual:
    """Composite sentiment score for a ticker + time window."""
    
    ticker: str
    timestamp: str  # ISO 8601
    residual: float  # -1.0 (bearish) to +1.0 (bullish), main output
    residual_confidence: float  # 0.0 to 1.0, how much to trust the residual
    
    # Component breakdown for transparency
    scout_component: float  # weighted avg of Scout signals
    linguist_component: float  # weighted avg of Linguist scores
    historian_component: float  # historical adjustment
    signal_alignment: float  # how well Scout and Linguist agree (-1 to +1)
    
    # Metadata
    dominant_signal: str  # which source had strongest influence
    anomaly_flags: List[str] = field(default_factory=list)


class SentimentAggregator:
    """Weighted sentiment fusion engine."""
    
    def __init__(
        self,
        scout_weight: float = 0.40,
        linguist_weight: float = 0.35,
        historian_weight: float = 0.25,
    ):
        """Initialize aggregator with component weights.
        
        Args:
            scout_weight: fraction of residual from Scout signals (0.0 to 1.0)
            linguist_weight: fraction from Linguist scores
            historian_weight: fraction from historical context
        
        Weights should sum to ~1.0; will be normalized if not.
        """
        total = scout_weight + linguist_weight + historian_weight
        if total <= 0:
            raise ValueError("Sum of weights must be > 0")
        
        self.scout_weight = scout_weight / total
        self.linguist_weight = linguist_weight / total
        self.historian_weight = historian_weight / total
    
    def aggregate(
        self,
        ticker: str,
        timestamp: str,
        scout: ScoutSignals,
        linguist: LinguistScores,
        historian: Optional[HistorianContext] = None,
    ) -> SentimentResidual:
        """Combine Scout, Linguist, and Historian signals into composite residual.
        
        Args:
            ticker: stock symbol (e.g., "AAPL")
            timestamp: ISO 8601 timestamp of aggregation
            scout: ScoutSignals object with price, news, social, dev signals
            linguist: LinguistScores from Linguist reasoning pipeline
            historian: optional HistorianContext for RAG weighting; uses defaults if None
        
        Returns:
            SentimentResidual with composite score and component breakdown.
        """
        if not historian:
            historian = HistorianContext(
                similar_event_count=0,
                historical_accuracy=0.5,
                base_volatility=0.02,
                sector_baseline_sentiment=0.0,
            )
        
        # === Scout Component ===
        scout_component = self._compute_scout_component(scout)
        scout_confidence = self._compute_scout_confidence(scout)
        
        # === Linguist Component ===
        linguist_component = self._compute_linguist_component(linguist)
        linguist_confidence = self._compute_linguist_confidence(linguist)
        
        # === Historian Component ===
        historian_component = self._compute_historian_component(scout_component, historian)
        historian_confidence = self._compute_historian_confidence(historian)
        
        # === Signal Alignment ===
        signal_alignment = self._compute_alignment(scout_component, linguist_component)
        
        # === Composite Residual ===
        residual = (
            self.scout_weight * scout_component
            + self.linguist_weight * linguist_component
            + self.historian_weight * historian_component
        )
        
        # Residual confidence: weighted avg of component confidences, penalized by misalignment
        raw_confidence = (
            self.scout_weight * scout_confidence
            + self.linguist_weight * linguist_confidence
            + self.historian_weight * historian_confidence
        )
        alignment_penalty = 1.0 - (abs(signal_alignment) * 0.1)  # max 10% penalty
        residual_confidence = raw_confidence * alignment_penalty
        residual_confidence = max(0.0, min(1.0, residual_confidence))
        
        # === Dominant Signal ===
        dominant_signal = self._identify_dominant_signal(
            scout_component, scout_confidence,
            linguist_component, linguist_confidence,
            historian_component, historian_confidence,
        )
        
        # === Anomaly Detection ===
        anomalies = self._detect_anomalies(
            scout, linguist, historian,
            scout_component, linguist_component,
            signal_alignment,
        )
        
        return SentimentResidual(
            ticker=ticker,
            timestamp=timestamp,
            residual=max(-1.0, min(1.0, residual)),  # clamp to [-1, 1]
            residual_confidence=residual_confidence,
            scout_component=scout_component,
            linguist_component=linguist_component,
            historian_component=historian_component,
            signal_alignment=signal_alignment,
            dominant_
