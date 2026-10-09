"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout signals (price momentum, news sentiment, SEC filing tone) and
Linguist scores (certainty, linguistic drift, regulatory whispers) into a
composite SentimentResidual score using a weighted formula. This residual
represents the "surprise" or "deviation from baseline" that predicts near-term
price movement.

Integration point: Called by judge/predictor.py to enrich per-ticker predictions
with multi-signal confidence scores.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class ScoutSignals:
    """Container for Scout-sourced market and textual signals."""

    price_momentum: float  # [-1.0, 1.0]: recent % change normalized
    news_sentiment: float  # [-1.0, 1.0]: aggregated headline tone
    sec_tone: float  # [-1.0, 1.0]: filing language positivity/negativity
    volume_anomaly: float  # [0.0, 2.0]: vol ratio vs. 20-day MA
    reddit_mentions: float  # [-1.0, 1.0]: community sentiment (scaled)


@dataclass
class LinguistScores:
    """Container for Linguist-sourced reasoning signals."""

    certainty: float  # [0.0, 1.0]: confidence in language analysis
    linguistic_drift: float  # [-1.0, 1.0]: tone shift vs. historical baseline
    regulatory_whispers: float  # [0.0, 1.0]: regulatory risk signal strength
    hedge_density: float  # [0.0, 1.0]: prevalence of cautious/hedged language


@dataclass
class SentimentResidual:
    """Output composite score and component breakdown."""

    composite_score: float  # [-1.0, 1.0]: final aggregated sentiment
    component_scores: Dict[str, float]  # breakdown of each signal's contribution
    confidence: float  # [0.0, 1.0]: overall confidence in the composite
    signal_count: int  # number of non-null inputs


class SentimentAggregator:
    """Weighted aggregator of Scout and Linguist signals into SentimentResidual."""

    def __init__(
        self,
        scout_weights: Optional[Dict[str, float]] = None,
        linguist_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize aggregator with configurable signal weights.

        Args:
            scout_weights: Dict mapping scout signal names to [0,1] weights.
                          Defaults to equal weighting if None.
            linguist_weights: Dict mapping linguist signal names to [0,1] weights.
                             Defaults to equal weighting if None.
        """
        # Default Scout weights: price momentum is primary, news/SEC secondary
        self.scout_weights = scout_weights or {
            "price_momentum": 0.35,
            "news_sentiment": 0.25,
            "sec_tone": 0.20,
            "volume_anomaly": 0.12,
            "reddit_mentions": 0.08,
        }

        # Default Linguist weights: certainty and drift are anchors
        self.linguist_weights = linguist_weights or {
            "certainty": 0.30,
            "linguistic_drift": 0.35,
            "regulatory_whispers": 0.20,
            "hedge_density": 0.15,
        }

        # Validate weights sum to ~1.0
        scout_sum = sum(self.scout_weights.values())
        linguist_sum = sum(self.linguist_weights.values())
        if not (0.99 < scout_sum < 1.01):
            raise ValueError(
                f"scout_weights must sum to 1.0, got {scout_sum}"
            )
        if not (0.99 < linguist_sum < 1.01):
            raise ValueError(
                f"linguist_weights must sum to 1.0, got {linguist_sum}"
            )

    def aggregate(
        self,
        scout: ScoutSignals,
        linguist: LinguistScores,
        scout_weight: float = 0.55,
        linguist_weight: float = 0.45,
    ) -> SentimentResidual:
        """
        Combine Scout and Linguist signals into composite SentimentResidual.

        Args:
            scout: ScoutSignals container with market/textual signals.
            linguist: LinguistScores container with reasoning/analysis signals.
            scout_weight: Relative weight of Scout pillar vs Linguist [0, 1].
            linguist_weight: Relative weight of Linguist pillar vs Scout [0, 1].

        Returns:
            SentimentResidual with composite_score, components, and confidence.

        Raises:
            ValueError: If scout_weight + linguist_weight != 1.0 (approx).
        """
        if not (0.99 < (scout_weight + linguist_weight) < 1.01):
            raise ValueError(
                f"scout_weight + linguist_weight must sum to 1.0, "
                f"got {scout_weight + linguist_weight}"
            )

        # Compute Scout pillar score
        scout_score = self._aggregate_scout(scout)
        scout_components = self._extract_scout_components(scout)

        # Compute Linguist pillar score
        linguist_score = self._aggregate_linguist(linguist)
        linguist_components = self._extract_linguist_components(linguist)

        # Blend pillars
        composite = (scout_score * scout_weight) + (
            linguist_score * linguist_weight
        )

        # Clamp to [-1.0, 1.0]
        composite = max(-1.0, min(1.0, composite))

        # Confidence: modulated by certainty and signal density
        signal_count = sum(
            [
                scout is not None,
                linguist.certainty > 0.0,
            ]
        )
        base_confidence = linguist.certainty
        density_boost = signal_count / 2.0  # max 1.0 if all signals present

        confidence = min(1.0, base_confidence * 0.7 + density_boost * 0.3)

        # Merge component breakdowns
        all_components = {**scout_components, **linguist_components}

        return SentimentResidual(
            composite_score=composite,
            component_scores=all_components,
            confidence=confidence,
            signal_count=signal_count,
        )

    def _aggregate_scout(self, scout: ScoutSignals) -> float:
        """
        Weighted blend of Scout signals into [−1, 1] score.

        Args:
            scout: ScoutSignals container.

        Returns:
            Weighted average of scout signals, clamped to [-1.0, 1.0].
        """
        signals = {
            "price_momentum": scout.price_momentum,
            "news_sentiment": scout.news_sentiment,
            "sec_tone": scout.sec_tone,
            "volume_anomaly": self._normalize_volume(scout.volume_anomaly),
            "reddit_mentions": scout.reddit_mentions,
        }

        weighted_sum = 0.0
        for signal_name, value in signals.items():
            if value is not None:
                weight = self.scout_weights.get(signal_name, 0.0)
                weighted_sum += value * weight

        return max(-1.0, min(1.0, weighted_sum))

    def _aggregate_linguist(self, linguist: LinguistScores) -> float:
        """
        Weighted blend of Linguist signals into [−1, 1] score.

        Args:
            linguist: LinguistScores container.

        Returns:
            Weighted average of linguist signals, clamped to [-1.0, 1.0].
        """
        signals = {
            "linguistic_drift": linguist.linguistic_drift,
            "regulatory_whispers": self._normalize_regulatory(
                linguist.regulatory_whispers
            ),
