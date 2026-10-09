"""
Confidence Score Weighting System for Sentinel RAG Pipeline.

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. Used by the Judge to calibrate prediction
certainty based on historical event relevance and temporal distance.

Exports:
  - compute_recency_decay(): Exponential decay factor (0.0–1.0) based on days elapsed.
  - weight_rag_similarities(): Normalize and apply decay to RAG match scores.
  - combine_confidence_signals(): Merge multiple weighted sources into single score.
  - WeightedConfidenceResult: Dataclass holding final score + component breakdown.
"""

from dataclasses import dataclass
from typing import List, Tuple
import math
from datetime import datetime, timedelta


@dataclass
class WeightedConfidenceResult:
    """Holds final confidence score and component details for audit trail."""
    
    final_score: float
    """Merged confidence (0.0–1.0), ready for Judge use."""
    
    rag_score_weighted: float
    """RAG similarity after decay applied."""
    
    recency_decay_factor: float
    """Exponential decay multiplier applied to oldest event."""
    
    component_count: int
    """Number of signals merged."""
    
    details: str
    """Human-readable breakdown of score composition."""


def compute_recency_decay(
    event_date: datetime,
    reference_date: datetime | None = None,
    half_life_days: float = 90.0,
) -> float:
    """
    Compute exponential recency decay factor for historical events.
    
    Uses half-life decay: factor = 0.5^(elapsed_days / half_life_days).
    Older events get lower weights; baseline half-life is 90 days.
    
    Args:
        event_date: When the historical event occurred.
        reference_date: Anchor for "now" (defaults to datetime.utcnow()).
        half_life_days: Days until score drops to 0.5 (default 90).
    
    Returns:
        float in [0.0, 1.0]; 1.0 = same day, 0.5 = half-life, 0.0 = very old.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    elapsed = (reference_date - event_date).total_seconds() / 86400.0
    if elapsed < 0:
        elapsed = 0.0
    
    decay = 0.5 ** (elapsed / half_life_days)
    return max(0.0, min(1.0, decay))


def weight_rag_similarities(
    similarity_scores: List[float],
    event_dates: List[datetime],
    reference_date: datetime | None = None,
    half_life_days: float = 90.0,
) -> Tuple[List[float], float]:
    """
    Apply recency decay to raw RAG similarity scores.
    
    For each (similarity, event_date) pair, multiply similarity by its
    recency decay factor. Returns both the weighted list and mean weight.
    
    Args:
        similarity_scores: RAG cosine similarity or dot-product scores (0.0–1.0).
        event_dates: Corresponding dates for each similarity score.
        reference_date: Anchor date (defaults to now).
        half_life_days: Decay half-life in days.
    
    Returns:
        Tuple of (weighted_scores list, mean_weight float).
    """
    if not similarity_scores or not event_dates:
        return [], 0.0
    
    if len(similarity_scores) != len(event_dates):
        raise ValueError(
            f"Mismatch: {len(similarity_scores)} scores vs {len(event_dates)} dates"
        )
    
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    weighted = []
    decay_factors = []
    
    for sim, evt_date in zip(similarity_scores, event_dates):
        decay = compute_recency_decay(evt_date, reference_date, half_life_days)
        decay_factors.append(decay)
        weighted.append(sim * decay)
    
    mean_weight = sum(decay_factors) / len(decay_factors) if decay_factors else 0.0
    
    return weighted, mean_weight


def combine_confidence_signals(
    rag_weighted_scores: List[float],
    sentiment_scores: List[float] | None = None,
    regulation_flags: List[float] | None = None,
    weights: Tuple[float, float, float] | None = None,
) -> WeightedConfidenceResult:
    """
    Merge multiple confidence signals (RAG, sentiment, regulatory) into final score.
    
    Applies weighted averaging across available signals. Missing signals are
    skipped; weights are renormalized. Default: (RAG=0.5, Sentiment=0.3, Reg=0.2).
    
    Args:
        rag_weighted_scores: Recency-decayed RAG similarities (from weight_rag_similarities).
        sentiment_scores: Optional list of sentiment confidence scores (0.0–1.0).
        regulation_flags: Optional regulatory signal strengths (0.0–1.0).
        weights: Tuple (rag_weight, sentiment_weight, reg_weight). Defaults (0.5, 0.3, 0.2).
    
    Returns:
        WeightedConfidenceResult with final_score and breakdown.
    """
    if weights is None:
        weights = (0.5, 0.3, 0.2)
    
    rag_w, sent_w, reg_w = weights
    
    signals = {}
    signal_weights = {}
    
    if rag_weighted_scores:
        rag_mean = sum(rag_weighted_scores) / len(rag_weighted_scores)
        signals["rag"] = rag_mean
        signal_weights["rag"] = rag_w
    
    if sentiment_scores:
        sent_mean = sum(sentiment_scores) / len(sentiment_scores)
        signals["sentiment"] = sent_mean
        signal_weights["sentiment"] = sent_w
    
    if regulation_flags:
        reg_mean = sum(regulation_flags) / len(regulation_flags)
        signals["regulation"] = reg_mean
        signal_weights["regulation"] = reg_w
    
    if not signals:
        return WeightedConfidenceResult(
            final_score=0.0,
            rag_score_weighted=0.0,
            recency_decay_factor=0.0,
            component_count=0,
            details="No confidence signals provided.",
        )
    
    total_weight = sum(signal_weights.values())
    if total_weight == 0:
        total_weight = 1.0
    
    normalized_weights = {k: v / total_weight for k, v in signal_weights.items()}
    final = sum(signals[k] * normalized_weights[k] for k in signals)
    
    rag_component = signals.get("rag", 0.0)
    
    recency_avg = 1.0
    if rag_weighted_scores:
        rag_raw = [s / (d + 1e-9) for s, d in zip(rag_weighted_scores, [0.5] * len(rag_weighted_scores))]
        recency_avg = sum(rag_weighted_scores) / (sum(rag_raw) + 1e-9) if sum(rag_raw) > 0 else 1.0
    
    details = (
        f"Merged {len(signals)} signals: "
        + ", ".join(f"{k}={signals[k]:.3f}×{normalized_weights[k]:.2f}" for k in signals)
        + f" → {final:.3f}"
    )
    
    return WeightedConfidenceResult(
        final_score=min(1.0, max(0.0, final)),
        rag_score_weighted=rag_component,
        recency_decay_factor=recency
