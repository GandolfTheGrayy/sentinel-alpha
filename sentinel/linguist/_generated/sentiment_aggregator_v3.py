"""
Sentiment Aggregator for Sentinel.

This module combines Scout signals (price momentum, news volume, social sentiment)
and Linguist scores (certainty, linguistic drift, regulatory whispers) into a
composite SentimentResidual score. The weighted formula synthesizes disparate
signals into a normalized [-1.0, 1.0] sentiment estimate, feeding Judge predictions.

The SentimentResidual represents the gap between market consensus (embedded in
price/volume) and linguistic/sentiment indicators—positive residuals suggest
upside surprise potential, negative ones downside risk.
"""

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoutSignals:
    """Container for raw Scout-derived market signals."""

    price_momentum: float
    """24h price change as normalized zscore [-inf, +inf]; clipped to [-3, 3]."""

    news_volume_zscore: float
    """Z-score of news mention frequency vs. 30d rolling average."""

    social_sentiment: float
    """Aggregated Reddit/HN upvote ratio sentiment on [-1, 1]."""

    social_volume_zscore: float
    """Z-score of social post count vs. baseline; clipped to [-3, 3]."""


@dataclass
class LinguistScores:
    """Container for Linguist-derived reasoning signals."""

    certainty: float
    """[0, 1] confidence that sentiment direction is well-supported."""

    linguistic_drift: float
    """[-1, 1] tone shift vs. 30d rolling baseline; positive = improving."""

    regulatory_whispers: float
    """[-1, 1] sentiment polarity of regulatory language; -1 = hostile."""


@dataclass
class SentimentResidual:
    """Composite sentiment estimate and diagnostics."""

    score: float
    """[-1.0, 1.0] normalized sentiment; >0 bullish, <0 bearish."""

    components: dict
    """Breakdown: {'scout_signal': float, 'linguist_signal': float, ...}."""

    explanation: str
    """Human-readable summary of dominant drivers."""

    uncertainty: float
    """[0, 1] epistemic uncertainty; high when signals conflict."""


def aggregate_sentiment(
    scout: ScoutSignals, linguist: LinguistScores, weights: Optional[dict] = None
) -> SentimentResidual:
    """
    Synthesize Scout and Linguist signals into a composite SentimentResidual.

    Formula:
      scout_component = (
        w_mom * clip(price_momentum, -3, 3) / 3
        + w_news * clip(news_volume_zscore, -3, 3) / 3
        + w_social * social_sentiment
        + w_social_vol * clip(social_volume_zscore, -3, 3) / 3
      ) / sum(w_*)

      linguist_component = (
        w_cert * certainty * (2 * linguistic_drift + regulatory_whispers) / 2
      )

      final_score = (
        w_scout * scout_component + w_linguist * linguist_component
      ) / (w_scout + w_linguist)

      uncertainty = (1 - certainty) * (1 - |correlation|)
        where correlation measures agreement between scout & linguist signals.

    Args:
        scout: ScoutSignals container with price, news, social metrics.
        linguist: LinguistScores with certainty, drift, regulatory sentiment.
        weights: Optional dict overriding defaults:
                  {'price_momentum': 0.25, 'news_volume_zscore': 0.20,
                   'social_sentiment': 0.30, 'social_volume_zscore': 0.25,
                   'certainty': 0.6, 'scout_weight': 0.5, 'linguist_weight': 0.5}

    Returns:
        SentimentResidual with score, components, explanation, uncertainty.
    """
    # Default weights: Scout signals balanced by frequency; Linguist gated by certainty.
    default_weights = {
        "price_momentum": 0.25,
        "news_volume_zscore": 0.20,
        "social_sentiment": 0.30,
        "social_volume_zscore": 0.25,
        "certainty_gate": 0.6,
        "scout_weight": 0.5,
        "linguist_weight": 0.5,
    }
    w = {**default_weights, **(weights or {})}

    # Normalize Scout signals: clip to [-3, 3], divide by 3 to get [-1, 1].
    mom_norm = min(max(scout.price_momentum, -3.0), 3.0) / 3.0
    news_norm = min(max(scout.news_volume_zscore, -3.0), 3.0) / 3.0
    social_norm = scout.social_sentiment  # Already [-1, 1]
    social_vol_norm = min(max(scout.social_volume_zscore, -3.0), 3.0) / 3.0

    # Scout component: weighted average of normalized signals.
    scout_sum = (
        w["price_momentum"] * mom_norm
        + w["news_volume_zscore"] * news_norm
        + w["social_sentiment"] * social_norm
        + w["social_volume_zscore"] * social_vol_norm
    )
    scout_weight_sum = (
        w["price_momentum"]
        + w["news_volume_zscore"]
        + w["social_sentiment"]
        + w["social_volume_zscore"]
    )
    scout_component = scout_sum / scout_weight_sum if scout_weight_sum > 0 else 0.0

    # Linguist component: certainty gates the blend of drift + regulatory sentiment.
    linguist_blend = (2.0 * scout.linguistic_drift + scout.regulatory_whispers) / 3.0
    linguist_component = w["certainty_gate"] * linguist.certainty * linguist_blend

    # Final aggregation: weighted average of Scout and Linguist components.
    final_score = (
        w["scout_weight"] * scout_component
        + w["linguist_weight"] * linguist_component
    ) / (w["scout_weight"] + w["linguist_weight"])

    # Clip final score to [-1, 1].
    final_score = min(max(final_score, -1.0), 1.0)

    # Compute uncertainty: high when certainty is low OR signals conflict.
    signal_agreement = 1.0 - abs(
        scout_component - linguist_component
    )  # [-1, 1] → [0, 2]
    signal_agreement = min(max(signal_agreement, 0.0), 1.0)
    uncertainty = (1.0 - linguist.certainty) * (1.0 - signal_agreement)

    # Build explanation string.
    scout_drivers = []
    if abs(mom_norm) > 0.3:
        scout_drivers.append(
            f"price_momentum={mom_norm:+.2f} ({'up' if mom_norm > 0 else 'down'})"
        )
    if abs(news_norm) > 0.3:
        scout_drivers.append(
            f"news_volume={news_norm:+.2f} ({'spike' if news_norm > 0 else 'quiet'})"
        )
    if abs(social_norm) > 0.3:
        scout_drivers.append(
            f"social_sentiment={social_norm:+.2f} ({'bullish' if social_norm > 0 else 'bearish'})"
        )

    linguist_drivers = []
    if linguist.certainty < 0.5:
        linguist_drivers.append(f"low_certainty={linguist.certainty:.2f}")
    if abs(linguist.linguistic_drift) > 0.2:
        linguist_drivers.append(
            f"drift={linguist.linguistic_drift:+.2f} ({'improving' if linguist.linguistic_drift > 0 else 'deteriorating'}
