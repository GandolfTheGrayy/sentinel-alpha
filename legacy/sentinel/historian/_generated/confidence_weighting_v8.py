"""
Sentinel Historian — Confidence Score Weighting System

Combines RAG similarity scores with recency decay to produce a final
WeightedConfidence float. This module bridges RAG retrieval (which yields
cosine similarity scores) with temporal relevance, ensuring that older
historical events have diminished influence on current predictions while
maintaining strong signal from recent high-confidence matches.

Used by: sentinel/judge/predictor.py for final prediction confidence calibration.
"""

from datetime import datetime, timedelta
from typing import List, Tuple
import math


def recency_decay_factor(event_date: datetime, reference_date: datetime | None = None, half_life_days: float = 30.0) -> float:
    """
    Compute exponential decay weight based on days elapsed since event_date.
    
    Args:
        event_date: The datetime of the historical event.
        reference_date: Comparison point (default: now).
        half_life_days: Days until weight reaches 0.5 (default: 30).
    
    Returns:
        Decay factor in (0, 1], where 1.0 = today, 0.5 = half_life_days ago.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    days_elapsed = (reference_date - event_date).total_seconds() / 86400.0
    
    if days_elapsed < 0:
        days_elapsed = 0
    
    decay = math.exp(-0.693147 * days_elapsed / half_life_days)
    return max(0.01, min(1.0, decay))


def normalize_rag_scores(rag_scores: List[float]) -> List[float]:
    """
    Normalize RAG similarity scores to [0, 1] range using min-max scaling.
    
    Args:
        rag_scores: List of raw cosine similarity scores from ChromaDB.
    
    Returns:
        List of normalized scores in [0, 1].
    """
    if not rag_scores:
        return []
    
    min_score = min(rag_scores)
    max_score = max(rag_scores)
    
    if max_score == min_score:
        return [1.0] * len(rag_scores)
    
    normalized = [
        (score - min_score) / (max_score - min_score)
        for score in rag_scores
    ]
    return normalized


def compute_weighted_confidence(
    rag_scores: List[float],
    event_dates: List[datetime],
    reference_date: datetime | None = None,
    half_life_days: float = 30.0,
    rag_weight: float = 0.7,
    recency_weight: float = 0.3,
) -> float:
    """
    Combine normalized RAG scores with recency decay into a single WeightedConfidence.
    
    Args:
        rag_scores: List of cosine similarity scores from RAG retrieval.
        event_dates: List of datetime objects corresponding to each RAG score.
        reference_date: Reference point for recency calculation (default: now).
        half_life_days: Exponential decay half-life in days (default: 30).
        rag_weight: Importance of RAG similarity (default: 0.7).
        recency_weight: Importance of recency (default: 0.3).
    
    Returns:
        Weighted confidence score in [0, 1].
    """
    if not rag_scores or not event_dates:
        return 0.0
    
    if len(rag_scores) != len(event_dates):
        raise ValueError("rag_scores and event_dates must have equal length")
    
    normalized_scores = normalize_rag_scores(rag_scores)
    
    weighted_sum = 0.0
    weight_total = 0.0
    
    for norm_score, event_date in zip(normalized_scores, event_dates):
        decay = recency_decay_factor(event_date, reference_date, half_life_days)
        
        combined_score = (rag_weight * norm_score) + (recency_weight * decay)
        weighted_sum += combined_score
        weight_total += 1.0
    
    if weight_total == 0:
        return 0.0
    
    final_confidence = weighted_sum / weight_total
    return max(0.0, min(1.0, final_confidence))


def compute_per_match_confidence(
    rag_score: float,
    event_date: datetime,
    reference_date: datetime | None = None,
    half_life_days: float = 30.0,
    rag_weight: float = 0.7,
    recency_weight: float = 0.3,
) -> float:
    """
    Compute confidence for a single RAG match without batch normalization.
    
    Args:
        rag_score: Raw cosine similarity score (typically in [0, 1]).
        event_date: Datetime of the historical event.
        reference_date: Reference point for recency (default: now).
        half_life_days: Exponential decay half-life (default: 30).
        rag_weight: Weight of RAG similarity (default: 0.7).
        recency_weight: Weight of recency (default: 0.3).
    
    Returns:
        Confidence score in [0, 1].
    """
    decay = recency_decay_factor(event_date, reference_date, half_life_days)
    
    clamped_score = max(0.0, min(1.0, rag_score))
    
    confidence = (rag_weight * clamped_score) + (recency_weight * decay)
    return max(0.0, min(1.0, confidence))


def adaptive_half_life(ticker: str, volatility: float = 0.02) -> float:
    """
    Suggest an adaptive half-life based on ticker volatility profile.
    
    Args:
        ticker: Stock ticker symbol (informational only in this stub).
        volatility: Annualized volatility estimate (default: 0.02 = 2%).
    
    Returns:
        Suggested half-life in days.
    """
    base_half_life = 30.0
    
    volatility_scale = 1.0 + (volatility / 0.02)
    
    adaptive = base_half_life * volatility_scale
    return max(7.0, min(90.0, adaptive))


if __name__ == "__main__":
    now = datetime.utcnow()
    
    test_rag_scores = [0.85, 0.72, 0.91]
    test_dates = [
        now - timedelta(days=5),
        now - timedelta(days=15),
        now - timedelta(days=1),
    ]
    
    confidence = compute_weighted_confidence(
        test_rag_scores,
        test_dates,
        reference_date=now,
        half_life_days=30.0,
    )
    print(f"Batch WeightedConfidence: {confidence:.4f}")
    
    single_confidence = compute_per_match_confidence(
        0.88,
        now - timedelta(days=10),
        reference_date=now,
    )
    print(f"Single-match confidence: {single_confidence:.4f}")
    
    suggested_half_life = adaptive_half_life("AAPL", volatility=0.035)
    print(f"Adaptive half-life for AAPL: {suggested_half_life:.1f} days")
