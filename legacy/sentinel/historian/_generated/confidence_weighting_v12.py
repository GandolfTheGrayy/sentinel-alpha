"""
Confidence score weighting system for Sentinel Historian.

Combines RAG similarity scores with recency decay to produce a final WeightedConfidence
float. This module bridges the gap between raw vector similarity (how semantically
close a historical event is to the current query) and temporal relevance (how recent
that event was). The result is a normalized confidence score [0.0, 1.0] that the
Judge can use to calibrate prediction strength.

Integration:
  - Called by sentinel/historian/rag_query.py after similarity retrieval.
  - Passed to sentinel/judge/predictor.py to modulate prediction magnitude.
  - Logged in sentinel/judge/postmortem.py for audit trails.
"""

from datetime import datetime, timedelta
from typing import List, Tuple
import math


def compute_recency_decay(
    event_date: datetime,
    reference_date: datetime | None = None,
    half_life_days: float = 365.0,
) -> float:
    """
    Compute exponential decay factor for event recency; older events get lower weight.
    
    Uses half-life model: decay = 2^(-days_ago / half_life_days).
    At half_life_days, the factor is 0.5. At 2*half_life_days, it's 0.25.
    
    Args:
        event_date: Timestamp of the historical event.
        reference_date: Current date for decay calculation (default: today).
        half_life_days: Number of days for decay to reach 0.5 (default: 1 year).
    
    Returns:
        Decay factor in [0.0, 1.0]; clamped at 0.001 minimum to avoid log(0).
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    days_ago = (reference_date - event_date).total_seconds() / 86400.0
    days_ago = max(0.0, days_ago)
    
    decay = math.pow(2.0, -days_ago / half_life_days)
    return max(0.001, min(1.0, decay))


def normalize_similarity_scores(
    raw_similarities: List[float],
    method: str = "softmax",
) -> List[float]:
    """
    Normalize raw cosine similarity scores to [0.0, 1.0] distribution.
    
    Methods:
      - 'softmax': Exponential normalization (default); spreads mass more evenly.
      - 'minmax': Simple min-max scaling.
      - 'sigmoid': Sigmoid squashing around 0.5 baseline.
    
    Args:
        raw_similarities: List of cosine similarity scores (typically [0, 1]).
        method: Normalization strategy ('softmax', 'minmax', 'sigmoid').
    
    Returns:
        Normalized scores summing to 1.0 (softmax) or individually in [0, 1].
    """
    if not raw_similarities:
        return []
    
    if method == "softmax":
        # Exponential rescaling + normalization.
        # Boost high scores, suppress low ones.
        exp_scores = [math.exp(2.0 * s) for s in raw_similarities]
        total = sum(exp_scores)
        return [e / total for e in exp_scores] if total > 0 else raw_similarities
    
    elif method == "minmax":
        min_sim = min(raw_similarities)
        max_sim = max(raw_similarities)
        if max_sim == min_sim:
            return [0.5] * len(raw_similarities)
        return [(s - min_sim) / (max_sim - min_sim) for s in raw_similarities]
    
    elif method == "sigmoid":
        # Squash around 0.5; scores near 0.5 get boosted.
        return [1.0 / (1.0 + math.exp(-8.0 * (s - 0.5))) for s in raw_similarities]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def weighted_confidence_from_retrieval(
    retrieval_results: List[Tuple[str, float, datetime]],
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 365.0,
    reference_date: datetime | None = None,
) -> float:
    """
    Combine RAG similarity scores and recency decay into a single confidence float.
    
    Each retrieved item contributes a score: (normalized_similarity * similarity_weight)
    + (recency_decay * recency_weight). Then aggregate via weighted mean.
    
    Args:
        retrieval_results: List of (content, similarity, event_date) tuples from RAG.
        similarity_weight: Relative importance of semantic match [0, 1].
        recency_weight: Relative importance of temporal proximity [0, 1].
        half_life_days: Recency decay half-life in days.
        reference_date: Current date for decay (default: today).
    
    Returns:
        Weighted confidence score in [0.0, 1.0].
    """
    if not retrieval_results:
        return 0.0
    
    # Normalize the weights.
    total_weight = similarity_weight + recency_weight
    sim_w = similarity_weight / total_weight
    rec_w = recency_weight / total_weight
    
    # Extract similarities and dates.
    similarities = [r[1] for r in retrieval_results]
    dates = [r[2] for r in retrieval_results]
    
    # Normalize similarities.
    norm_sims = normalize_similarity_scores(similarities, method="softmax")
    
    # Compute recency decays.
    recency_decays = [
        compute_recency_decay(d, reference_date, half_life_days)
        for d in dates
    ]
    
    # Per-item composite score.
    composite_scores = [
        (norm_sims[i] * sim_w) + (recency_decays[i] * rec_w)
        for i in range(len(retrieval_results))
    ]
    
    # Aggregate: mean of composites.
    confidence = sum(composite_scores) / len(composite_scores)
    return max(0.0, min(1.0, confidence))


def adaptive_confidence_with_count_penalty(
    retrieval_results: List[Tuple[str, float, datetime]],
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 365.0,
    reference_date: datetime | None = None,
    min_count_for_full_score: int = 5,
) -> float:
    """
    Compute confidence but penalize low retrieval counts (fewer matches = less certain).
    
    If fewer than min_count_for_full_score results are retrieved, the confidence
    is scaled down by (count / min_count). This reflects lower conviction when
    the historical corpus offers limited precedent.
    
    Args:
        retrieval_results: List of (content, similarity, event_date) tuples from RAG.
        similarity_weight: Relative importance of semantic match.
        recency_weight: Relative importance of recency.
        half_life_days: Recency decay half-life.
        reference_date: Current date for decay.
        min_count_for_full_score: Number of results needed to avoid penalty.
    
    Returns:
        Penalized confidence score in [0.0, 1.0].
    """
    base_confidence = weighted_confidence_from_retrieval(
        retrieval_results,
        similarity_weight,
        recency_weight,
        half_life_days,
        reference_date,
    )
    
    count = len(retrieval_results)
    if count < min_count_for_full_score:
        penalty = count / min_count_for_full_score
        return base_confidence * penalty
    
    return base_confidence


def confidence_calibration_report(
    retrieval
