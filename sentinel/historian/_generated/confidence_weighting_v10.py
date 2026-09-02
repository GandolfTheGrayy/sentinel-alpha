"""
Confidence score weighting system for Sentinel Historian.

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence metric. It accounts for how old a reference event
is and how semantically similar it is to the current query, weighting recent
and highly-similar signals more heavily in final prediction confidence.

Used by sentinel/judge/predictor.py to calibrate certainty when synthesizing
multiple historical precedents into a single price movement prediction.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class RagReference:
    """A single RAG-retrieved historical reference with metadata."""
    similarity_score: float
    """Cosine similarity [0.0, 1.0] from vector retrieval."""
    event_date: datetime
    """Date the historical event occurred."""
    content: str
    """Text snippet of the reference."""
    ticker: Optional[str] = None
    """Stock ticker if cross-company analogy."""


@dataclass
class WeightedConfidence:
    """Final confidence metric for a prediction."""
    overall_score: float
    """Combined confidence [0.0, 1.0]."""
    recency_decay_factor: float
    """How much age penalized this score [0.0, 1.0]."""
    similarity_avg: float
    """Average RAG similarity across references."""
    reference_count: int
    """Number of references used in calculation."""
    explanation: str
    """Human-readable breakdown of weighting logic."""


def recency_decay(
    event_date: datetime,
    reference_date: datetime = None,
    half_life_days: float = 365.0
) -> float:
    """
    Compute exponential decay factor for historical event age.
    
    By default, events from one year ago contribute 50% confidence.
    Uses exponential decay: decay = 2^(-age_days / half_life_days)
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    age = (reference_date - event_date).total_seconds() / 86400.0
    age = max(0.0, age)
    
    decay = math.pow(2.0, -age / half_life_days)
    return max(0.0, min(1.0, decay))


def compute_weighted_confidence(
    rag_references: list[RagReference],
    reference_date: datetime = None,
    half_life_days: float = 365.0,
    min_references: int = 1
) -> WeightedConfidence:
    """
    Combine RAG similarity + recency decay into a final confidence score.
    
    Algorithm:
    1. For each reference, compute weighted_score = similarity * recency_decay
    2. Average all weighted scores
    3. Apply reference count penalty (fewer refs = lower confidence ceiling)
    4. Return structured WeightedConfidence with explanation
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if not rag_references:
        return WeightedConfidence(
            overall_score=0.0,
            recency_decay_factor=1.0,
            similarity_avg=0.0,
            reference_count=0,
            explanation="No RAG references provided."
        )
    
    weighted_scores = []
    decay_factors = []
    
    for ref in rag_references:
        decay = recency_decay(ref.event_date, reference_date, half_life_days)
        weighted = ref.similarity_score * decay
        weighted_scores.append(weighted)
        decay_factors.append(decay)
    
    avg_weighted = sum(weighted_scores) / len(weighted_scores)
    avg_decay = sum(decay_factors) / len(decay_factors)
    avg_similarity = sum(ref.similarity_score for ref in rag_references) / len(rag_references)
    
    # Reference count penalty: confidence ceiling drops with fewer references.
    # At min_references (default 1), penalty = 1.0. At 5+ refs, penalty ≈ 1.0.
    ref_penalty = min(
        1.0,
        1.0 - (0.2 * max(0, min_references - len(rag_references)))
    )
    
    overall = avg_weighted * ref_penalty
    overall = max(0.0, min(1.0, overall))
    
    oldest_date = min(ref.event_date for ref in rag_references)
    newest_date = max(ref.event_date for ref in rag_references)
    date_range = (newest_date - oldest_date).days
    
    explanation = (
        f"WeightedConfidence: {overall:.3f} | "
        f"Avg Similarity: {avg_similarity:.3f} | "
        f"Avg Recency Decay: {avg_decay:.3f} | "
        f"Ref Count: {len(rag_references)} | "
        f"Date Range: {date_range}d (oldest={oldest_date.date()}, newest={newest_date.date()})"
    )
    
    return WeightedConfidence(
        overall_score=overall,
        recency_decay_factor=avg_decay,
        similarity_avg=avg_similarity,
        reference_count=len(rag_references),
        explanation=explanation
    )


def adjust_confidence_by_anomaly(
    base_confidence: WeightedConfidence,
    anomaly_flags: list[str]
) -> WeightedConfidence:
    """
    Reduce confidence if anomaly flags suggest current context is unusual.
    
    Each anomaly (e.g., "market_circuit_breaker", "earnings_blackout")
    reduces overall_score by 10% per flag, with floor at 0.1.
    """
    penalty_per_flag = 0.10
    penalty = min(0.9, len(anomaly_flags) * penalty_per_flag)
    adjusted_score = max(0.1, base_confidence.overall_score * (1.0 - penalty))
    
    flags_str = ", ".join(anomaly_flags) if anomaly_flags else "none"
    new_explanation = (
        base_confidence.explanation +
        f" | Anomaly Adjustment ({flags_str}): {base_confidence.overall_score:.3f} → {adjusted_score:.3f}"
    )
    
    return WeightedConfidence(
        overall_score=adjusted_score,
        recency_decay_factor=base_confidence.recency_decay_factor,
        similarity_avg=base_confidence.similarity_avg,
        reference_count=base_confidence.reference_count,
        explanation=new_explanation
    )


def confidence_to_certainty_label(confidence: float) -> str:
    """Map numeric confidence [0.0, 1.0] to verbal certainty label."""
    if confidence >= 0.85:
        return "Very High"
    elif confidence >= 0.70:
        return "High"
    elif confidence >= 0.55:
        return "Moderate"
    elif confidence >= 0.40:
        return "Low"
    else:
        return "Very Low"
