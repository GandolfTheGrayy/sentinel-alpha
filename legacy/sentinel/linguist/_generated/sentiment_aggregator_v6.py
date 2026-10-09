"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout signals (price momentum, news volume, Reddit/HN sentiment,
GitHub health) with Linguist scores (certainty, linguistic drift, regulatory
whispers) into a composite SentimentResidual score via weighted formula.

This module synthesizes heterogeneous sentiment signals into a single
normalized confidence metric that feeds into Judge's prediction pipeline.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class ScoutSignals:
    """Container for raw Scout data signals."""

    price_momentum: float  # [-1, 1]: recent price velocity
    news_volume: float  # [0, 1]: normalized headline count spike
    reddit_sentiment: float  # [-1, 1]: aggregated Reddit/HN posts
    github_health: float  # [0, 1]: commit frequency, open issues ratio
    volatility_zscore: float  # [-3, 3]: market vol vs 30-day baseline


@dataclass
class LinguistScores:
    """Container for Linguist reasoning outputs."""

    certainty_score: float  # [0, 1]: confidence in language (Claude)
    linguistic_drift: float  # [-1, 1]: tone shift vs historical baseline
    regulatory_whispers: float  # [0, 1]: risk signal from SEC/legal language
    sentiment_polarity: float  # [-1, 1]: overall positive/negative tone


@dataclass
class SentimentResidual:
    """Composite sentiment metric with diagnostic breakdown."""

    composite_score: float  # [-1, 1]: final aggregated signal
    bull_components: dict  # signal_name → contribution
    bear_components: dict  # signal_name → contribution
    confidence: float  # [0, 1]: aggregator confidence
    component_breakdown: pd.DataFrame  # detailed scoring audit trail


def aggregate_sentiment(
    scout: ScoutSignals, linguist: LinguistScores, ticker: str
) -> SentimentResidual:
    """
    Combine Scout and Linguist signals into weighted SentimentResidual.

    Implements a multi-factor weighted formula where:
    - Scout signals (price, news, social, dev health) provide market context
    - Linguist scores (certainty, drift, regulatory) add reasoning confidence
    - Cross-signal harmonics detect convergence or divergence
    - Anomalies trigger confidence discounts

    Args:
        scout: Raw market/social signals from Scout module
        linguist: Reasoning scores from Linguist module
        ticker: Stock ticker for anomaly flagging

    Returns:
        SentimentResidual with composite score, component breakdown, confidence
    """

    # Normalize and validate all inputs to [-1, 1] or [0, 1]
    scout_norm = _normalize_scout_signals(scout)
    linguist_norm = _normalize_linguist_scores(linguist)

    # Primary weighted formula: Scout (40%) + Linguist (40%) + Harmonics (20%)
    scout_aggregate = _weighted_scout_aggregate(scout_norm)
    linguist_aggregate = _weighted_linguist_aggregate(linguist_norm)
    harmonic_score = _compute_harmonics(scout_norm, linguist_norm)

    # Final composite with clipping to [-1, 1]
    composite = (
        0.4 * scout_aggregate + 0.4 * linguist_aggregate + 0.2 * harmonic_score
    )
    composite = np.clip(composite, -1.0, 1.0)

    # Separate bull and bear contributions for audit trail
    bull_components = {}
    bear_components = {}

    for name, value in scout_norm.items():
        contribution = 0.4 * value * (1.0 / 5)  # 5 scout signals
        if contribution > 0:
            bull_components[f"scout_{name}"] = contribution
        else:
            bear_components[f"scout_{name}"] = contribution

    for name, value in linguist_norm.items():
        contribution = 0.4 * value * (1.0 / 4)  # 4 linguist scores
        if contribution > 0:
            bull_components[f"linguist_{name}"] = contribution
        else:
            bear_components[f"linguist_{name}"] = contribution

    if harmonic_score > 0:
        bull_components["harmonics"] = 0.2 * harmonic_score
    else:
        bear_components["harmonics"] = 0.2 * harmonic_score

    # Confidence = product of all signal confidences, discounted by anomalies
    base_confidence = _compute_confidence(scout_norm, linguist_norm)
    anomaly_discount = _detect_anomalies(scout_norm, linguist_norm, ticker)
    confidence = base_confidence * (1.0 - anomaly_discount)

    # Build component breakdown DataFrame
    component_df = _build_audit_dataframe(
        scout_norm, linguist_norm, harmonic_score, composite
    )

    return SentimentResidual(
        composite_score=composite,
        bull_components=bull_components,
        bear_components=bear_components,
        confidence=np.clip(confidence, 0.0, 1.0),
        component_breakdown=component_df,
    )


def _normalize_scout_signals(scout: ScoutSignals) -> dict:
    """
    Validate and normalize Scout signals to [-1, 1] or [0, 1].

    Args:
        scout: Raw Scout signals

    Returns:
        Dictionary of normalized values
    """
    return {
        "price_momentum": np.clip(scout.price_momentum, -1.0, 1.0),
        "news_volume": np.clip(scout.news_volume, 0.0, 1.0) * 2 - 1,  # Scale to [-1, 1]
        "reddit_sentiment": np.clip(scout.reddit_sentiment, -1.0, 1.0),
        "github_health": np.clip(scout.github_health, 0.0, 1.0) * 2 - 1,  # Scale to [-1, 1]
        "volatility_zscore": np.clip(scout.volatility_zscore / 3.0, -1.0, 1.0),
    }


def _normalize_linguist_scores(linguist: LinguistScores) -> dict:
    """
    Validate and normalize Linguist scores to [-1, 1] or [0, 1].

    Args:
        linguist: Raw Linguist scores

    Returns:
        Dictionary of normalized values
    """
    return {
        "certainty": (np.clip(linguist.certainty_score, 0.0, 1.0) * 2) - 1,
        "linguistic_drift": np.clip(linguist.linguistic_drift, -1.0, 1.0),
        "regulatory_whispers": (np.clip(linguist.regulatory_whispers, 0.0, 1.0) * 2 - 1) * -1,  # Invert: high regulatory risk = bear
        "polarity": np.clip(linguist.sentiment_polarity, -1.0, 1.0),
    }


def _weighted_scout_aggregate(scout_norm: dict) -> float:
    """
    Compute weighted aggregate of Scout signals.

    Weights: price_momentum (30%), reddit_sentiment (25%), news_volume (20%),
    github_health (15%), volatility (10%).

    Args:
        scout_norm: Normalized Scout signals

    Returns:
        Weighted aggregate in [-1, 1]
    """
    weights = {
        "price_momentum": 0.30,
        "reddit_sentiment": 0.25,
        "news_volume": 0.20,
        "github_health": 0.15,
        "volatility_zscore": 0.10,
    }
    return sum(scout_norm[key] * weights[key] for key in weights)


def _weighted_linguist_aggregate(linguist_norm: dict) -> float:
    """
    Compute weighted aggregate of Linguist scores.

    Weights:
