"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout signals (price momentum, news volume, social sentiment) and
Linguist scores (certainty, linguistic drift, regulatory whispers) into a
composite SentimentResidual score using a weighted formula. This score
quantifies net bullish/bearish conviction across all data sources.

The aggregator normalizes heterogeneous signals to [-1, +1] scale and applies
configurable weights to balance urgency (recent news) against stability
(historical patterns). Used by Judge.predictor to contextualize final forecasts.
"""

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class ScoutSignals:
    """Container for raw Scout-layer observations."""
    price_momentum: float  # [-1, +1]: recent price trend
    news_volume_zscore: float  # [-3, +3]: deviation from baseline volume
    social_sentiment_net: float  # [-1, +1]: aggregate Reddit/HN tone
    github_velocity_delta: float  # [-1, +1]: dev activity trend (if applicable)
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            'price_momentum': self.price_momentum,
            'news_volume_zscore': self.news_volume_zscore,
            'social_sentiment_net': self.social_sentiment_net,
            'github_velocity_delta': self.github_velocity_delta,
        }


@dataclass
class LinguistScores:
    """Container for Linguist-layer analysis outputs."""
    certainty_score: float  # [0, 1]: confidence in tone direction
    linguistic_drift: float  # [-1, +1]: tone shift vs. historical avg
    regulatory_whisper_severity: float  # [0, 1]: concern level from regulatory language
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            'certainty_score': self.certainty_score,
            'linguistic_drift': self.linguistic_drift,
            'regulatory_whisper_severity': self.regulatory_whisper_severity,
        }


@dataclass
class SentimentResidual:
    """Composite sentiment signal output."""
    net_score: float  # [-1, +1]: final aggregated sentiment
    urgency_boost: float  # [0, 1]: how recent/intense the signal is
    conviction_confidence: float  # [0, 1]: how stable/repeatable the signal is
    component_breakdown: dict  # Individual contributions for audit trail
    interpretation: str  # Human-readable label: "strong_bull" | "mild_bull" | "neutral" | "mild_bear" | "strong_bear"


def _clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def _normalize_zscore(zscore: float, stddev_threshold: float = 3.0) -> float:
    """
    Normalize z-score to [-1, +1] range using sigmoid-like curve.
    
    Values beyond ±stddev_threshold saturate at ±1.
    """
    clamped = _clamp(zscore / stddev_threshold, -1.0, 1.0)
    return clamped


def aggregate_sentiment(
    scout_signals: ScoutSignals,
    linguist_scores: LinguistScores,
    weights: Optional[dict] = None,
) -> SentimentResidual:
    """
    Combine Scout and Linguist signals into a composite SentimentResidual.
    
    Default weights balance price momentum (30%), news volume (25%), social
    sentiment (20%), certainty modulation (15%), and regulatory risk (10%).
    """
    if weights is None:
        weights = {
            'price_momentum': 0.30,
            'news_volume': 0.25,
            'social_sentiment': 0.20,
            'certainty_modulation': 0.15,
            'regulatory_risk': 0.10,
        }
    
    # Normalize news volume z-score to [-1, +1]
    norm_news_volume = _normalize_zscore(scout_signals.news_volume_zscore, stddev_threshold=3.0)
    
    # Clamp social sentiment (already [-1, +1] from Scout)
    norm_social_sentiment = _clamp(scout_signals.social_sentiment_net, -1.0, 1.0)
    
    # Clamp price momentum
    norm_price_momentum = _clamp(scout_signals.price_momentum, -1.0, 1.0)
    
    # Clamp GitHub velocity
    norm_github_velocity = _clamp(scout_signals.github_velocity_delta, -1.0, 1.0)
    
    # Linguist certainty acts as a confidence multiplier on scout signals
    certainty_mult = linguist_scores.certainty_score  # [0, 1]
    
    # Regulatory whisper is a penalty (reduces net bullish conviction)
    regulatory_penalty = -linguist_scores.regulatory_whisper_severity  # [-1, 0]
    
    # Linguistic drift acts as a modifier (accelerates or dampens sentiment direction)
    drift_modifier = linguist_scores.linguistic_drift  # [-1, +1]
    
    # Weighted combination of scout signals
    scout_net = (
        weights['price_momentum'] * norm_price_momentum +
        weights['news_volume'] * norm_news_volume +
        weights['social_sentiment'] * norm_social_sentiment +
        (weights.get('github_velocity', 0.0) * norm_github_velocity)
    )
    
    # Apply certainty as a confidence gate: high certainty preserves signal,
    # low certainty dampens it toward zero
    gated_scout = scout_net * certainty_mult
    
    # Apply regulatory penalty (risk mitigation)
    risk_adjusted = gated_scout + weights['regulatory_risk'] * regulatory_penalty
    
    # Apply linguistic drift as a small directional boost/dampen
    drifted = risk_adjusted + (weights.get('linguistic_drift', 0.05) * drift_modifier * certainty_mult)
    
    # Clamp final net score to [-1, +1]
    net_score = _clamp(drifted, -1.0, 1.0)
    
    # Urgency boost: combination of recent volume spike and high certainty
    urgency_boost = max(0.0, abs(norm_news_volume) * 0.6 + certainty_mult * 0.4)
    
    # Conviction confidence: how stable the signal is
    # High certainty + low drift uncertainty = high conviction
    conviction_confidence = (
        certainty_mult * 0.5 +
        (1.0 - abs(drift_modifier)) * 0.3 +
        (1.0 - linguist_scores.regulatory_whisper_severity) * 0.2
    )
    conviction_confidence = _clamp(conviction_confidence, 0.0, 1.0)
    
    # Build component breakdown for audit trail
    component_breakdown = {
        'price_momentum_contrib': norm_price_momentum * weights['price_momentum'] * certainty_mult,
        'news_volume_contrib': norm_news_volume * weights['news_volume'] * certainty_mult,
        'social_sentiment_contrib': norm_social_sentiment * weights['social_sentiment'] * certainty_mult,
        'github_velocity_contrib': norm_github_velocity * weights.get('github_velocity', 0.0) * certainty_mult,
        'regulatory_penalty_contrib': regulatory_penalty * weights['regulatory_risk'],
        'linguistic_drift_contrib': drift_modifier * weights.get('linguistic_drift', 0.05) * certainty_mult,
        'scout_signals': scout_signals.to_dict(),
        'linguist_scores': linguist_scores.to_dict(),
    }
    
    # Interpret net score into human-readable label
    if net_score
