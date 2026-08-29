"""
Sentinel Historian — Confidence Score Weighting System

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. It bridges the gap between raw embedding
similarity (from rag_query.py) and the Judge's prediction confidence.

Recency decay reflects the intuition that older market events are less
predictive of current price movements. Similarity scores reflect how closely
the current context matches historical precedents in the vector DB.

The final WeightedConfidence is a normalized float in [0.0, 1.0] that the
Judge uses to calibrate prediction strength and position sizing.
"""

import math
from datetime import datetime, timedelta
from typing import Optional


def recency_decay_factor(event_date: datetime, reference_date: Optional[datetime] = None, half_life_days: float = 90.0) -> float:
    """
    Compute exponential decay factor for historical event recency.
    
    Args:
        event_date: Timestamp of the historical event.
        reference_date: Current reference time (defaults to now).
        half_life_days: Days for similarity to decay to 50% (default: 90).
    
    Returns:
        Decay factor in [0.0, 1.0]; 1.0 = very recent, ~0.0 = very old.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    days_elapsed = (reference_date - event_date).total_seconds() / (24 * 3600)
    
    # Exponential decay: factor = 2^(-days_elapsed / half_life)
    if days_elapsed < 0:
        # Future events (shouldn't happen) get zero decay credit
        return 0.0
    
    decay = math.exp(-math.log(2) * days_elapsed / half_life_days)
    return max(0.0, min(1.0, decay))


def normalize_similarity_score(raw_similarity: float) -> float:
    """
    Normalize embedding similarity (typically cosine in [0, 1]) to confidence range.
    
    Args:
        raw_similarity: Raw cosine similarity or distance metric, ideally in [0, 1].
    
    Returns:
        Normalized score in [0.0, 1.0].
    """
    # Clamp to valid range and apply slight curve to emphasize high matches
    clamped = max(0.0, min(1.0, raw_similarity))
    # Optional: apply sigmoid-like curve to penalize weak matches
    # For now, linear normalization is sufficient
    return clamped


def compute_weighted_confidence(
    similarity_scores: list[float],
    event_dates: list[datetime],
    reference_date: Optional[datetime] = None,
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 90.0,
) -> float:
    """
    Compute final WeightedConfidence by combining similarity and recency.
    
    Args:
        similarity_scores: List of raw embedding similarities from RAG results.
        event_dates: Corresponding historical event timestamps.
        reference_date: Current time for decay calculation (defaults to now).
        similarity_weight: Contribution of similarity to final score (default: 0.6).
        recency_weight: Contribution of recency to final score (default: 0.4).
        half_life_days: Recency decay half-life in days (default: 90).
    
    Returns:
        WeightedConfidence float in [0.0, 1.0].
    
    Raises:
        ValueError: If lists have mismatched lengths or weights don't sum to ~1.0.
    """
    if not similarity_scores or not event_dates:
        return 0.0
    
    if len(similarity_scores) != len(event_dates):
        raise ValueError(
            f"similarity_scores ({len(similarity_scores)}) and "
            f"event_dates ({len(event_dates)}) must have equal length"
        )
    
    # Weights should sum to 1.0 (allow small floating-point tolerance)
    total_weight = similarity_weight + recency_weight
    if not (0.99 <= total_weight <= 1.01):
        raise ValueError(
            f"similarity_weight ({similarity_weight}) + recency_weight ({recency_weight}) "
            f"must sum to ~1.0, got {total_weight}"
        )
    
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    # Compute weighted average of normalized scores
    total_sim_score = 0.0
    total_recency_score = 0.0
    
    for sim, date in zip(similarity_scores, event_dates):
        norm_sim = normalize_similarity_score(sim)
        decay = recency_decay_factor(date, reference_date, half_life_days)
        
        total_sim_score += norm_sim
        total_recency_score += decay
    
    n = len(similarity_scores)
    avg_similarity = total_sim_score / n
    avg_recency = total_recency_score / n
    
    # Normalize weights (already done above, but explicit)
    w_sim = similarity_weight / (similarity_weight + recency_weight)
    w_rec = recency_weight / (similarity_weight + recency_weight)
    
    weighted_confidence = w_sim * avg_similarity + w_rec * avg_recency
    
    return max(0.0, min(1.0, weighted_confidence))


def combine_rag_predictions(
    rag_results: list[dict],
    reference_date: Optional[datetime] = None,
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 90.0,
) -> dict:
    """
    High-level wrapper: ingest RAG result dicts and emit WeightedConfidence + metadata.
    
    Each RAG result dict is expected to have:
      - "similarity": float (embedding cosine similarity, typically 0–1)
      - "event_date": datetime (when the historical event occurred)
      - "content": str (optional, for debugging)
      - "ticker": str (optional, for tracking)
    
    Args:
        rag_results: List of dicts returned from historian/rag_query.py.
        reference_date: Current time for decay (defaults to now).
        similarity_weight: Similarity contribution (default: 0.6).
        recency_weight: Recency contribution (default: 0.4).
        half_life_days: Recency half-life in days (default: 90).
    
    Returns:
        Dict with keys:
          - "weighted_confidence": float in [0.0, 1.0]
          - "num_events": int (count of RAG results used)
          - "avg_similarity": float (before weighting)
          - "avg_recency": float (before weighting)
          - "reference_date": datetime (used for computation)
    """
    if not rag_results:
        return {
            "weighted_confidence": 0.0,
            "num_events": 0,
            "avg_similarity": 0.0,
            "avg_recency": 0.0,
            "reference_date": reference_date or datetime.utcnow(),
        }
    
    similarities = [r.get("similarity", 0.0) for r in rag_results]
    dates = [r.get("event_date", datetime.utcnow()) for r in rag_results]
    
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    weighted_conf = compute_weighted_confidence(
        similarities, dates, reference_date, similarity_weight, recency_weight, half_life_days
    )
    
    # Compute component averages for transparency
    norm_sims = [normalize_similarity_score(s) for s in similarities]
    avg_sim = sum(norm_sims) / len(norm_sims) if norm_sims else 0.0
