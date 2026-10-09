"""
Confidence Score Weighting System for Sentinel Historian.

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. It implements time-aware confidence
aggregation, allowing older but highly relevant historical precedents to
decay gracefully while recent signals remain sharp.

Used by sentinel/judge/predictor.py to weight historical context signals
when synthesizing final price-movement predictions.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional


@dataclass
class RagSignal:
    """A single RAG retrieval result with similarity and provenance metadata."""
    
    similarity_score: float  # [0.0, 1.0] from vector DB
    retrieved_at: datetime   # When this signal was sourced
    source: str              # e.g., "SEC_10Q", "NEWS", "REDDIT"
    ticker: str              # Company ticker
    text_snippet: str        # Actual retrieved text


@dataclass
class WeightedConfidence:
    """Final aggregated confidence with component breakdown."""
    
    final_score: float       # [0.0, 1.0] weighted & decayed
    raw_similarity_mean: float  # Before decay
    recency_decay_factor: float  # How much time decay was applied
    signal_count: int        # Number of signals aggregated
    decay_halflife_days: float  # Halflife used for exponential decay


def compute_recency_decay(
    retrieved_at: datetime,
    reference_time: Optional[datetime] = None,
    halflife_days: float = 180.0
) -> float:
    """
    Compute exponential decay factor for signal age.
    
    Returns a decay multiplier in [0.0, 1.0], with 1.0 meaning "just retrieved"
    and values approaching 0 as age approaches infinity. At halflife_days,
    the multiplier equals 0.5.
    """
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    age_seconds = (reference_time - retrieved_at).total_seconds()
    age_days = age_seconds / (24 * 3600)
    
    if age_days < 0:
        age_days = 0
    
    decay = math.exp(-0.693147 * age_days / halflife_days)
    return decay


def weight_signals_by_source(
    signals: List[RagSignal],
    source_weights: Optional[dict] = None
) -> List[float]:
    """
    Assign per-signal weights based on source trustworthiness.
    
    Returns a list of weights matching signal order, summing to 1.0
    if source_weights is provided, else all 1.0.
    """
    if source_weights is None:
        source_weights = {
            "SEC_10Q": 1.0,
            "SEC_8K": 0.95,
            "NEWS": 0.7,
            "REDDIT": 0.5,
            "GITHUB": 0.6,
            "EARNINGS_CALL": 0.9,
        }
    
    weights = []
    for signal in signals:
        w = source_weights.get(signal.source, 0.5)
        weights.append(w)
    
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        weights = [1.0 / len(signals)] * len(signals)
    
    return weights


def aggregate_with_decay(
    signals: List[RagSignal],
    reference_time: Optional[datetime] = None,
    halflife_days: float = 180.0,
    source_weights: Optional[dict] = None,
    min_signals: int = 1
) -> WeightedConfidence:
    """
    Aggregate RAG similarity scores with exponential recency decay.
    
    Combines source weighting and time decay to produce a final confidence.
    If fewer than min_signals are provided, returns 0.0 confidence.
    """
    if len(signals) < min_signals:
        return WeightedConfidence(
            final_score=0.0,
            raw_similarity_mean=0.0,
            recency_decay_factor=1.0,
            signal_count=0,
            decay_halflife_days=halflife_days
        )
    
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    source_ws = weight_signals_by_source(signals, source_weights)
    
    weighted_sum = 0.0
    decay_factors = []
    
    for signal, source_w in zip(signals, source_ws):
        decay = compute_recency_decay(signal.retrieved_at, reference_time, halflife_days)
        decay_factors.append(decay)
        weighted_sum += signal.similarity_score * source_w * decay
    
    raw_mean = sum(s.similarity_score for s in signals) / len(signals)
    mean_decay = sum(decay_factors) / len(decay_factors)
    
    final_score = min(1.0, max(0.0, weighted_sum))
    
    return WeightedConfidence(
        final_score=final_score,
        raw_similarity_mean=raw_mean,
        recency_decay_factor=mean_decay,
        signal_count=len(signals),
        decay_halflife_days=halflife_days
    )


def threshold_confidence(
    confidence: WeightedConfidence,
    min_threshold: float = 0.5
) -> bool:
    """
    Check if confidence score meets minimum threshold for actionability.
    
    Returns True if final_score >= min_threshold, False otherwise.
    """
    return confidence.final_score >= min_threshold


def explain_confidence(confidence: WeightedConfidence) -> str:
    """
    Produce a human-readable explanation of confidence breakdown.
    
    Returns a formatted string describing score, decay, and signal count.
    """
    return (
        f"WeightedConfidence: {confidence.final_score:.3f} "
        f"(raw_similarity={confidence.raw_similarity_mean:.3f}, "
        f"decay_factor={confidence.recency_decay_factor:.3f}, "
        f"signals={confidence.signal_count}, "
        f"halflife={confidence.decay_halflife_days:.1f}d)"
    )
