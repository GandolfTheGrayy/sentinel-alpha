"""
Confidence score weighting system for Sentinel Historian.

Combines RAG similarity scores with recency decay to produce final WeightedConfidence scores.
This module bridges vector search results (from rag_query.py) with temporal relevance,
ensuring recent, high-similarity historical events receive appropriate weight in predictions.

Exported: WeightedConfidence dataclass, compute_weighted_confidence(), decay_by_days().
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import math


@dataclass
class WeightedConfidence:
    """
    Final confidence score combining RAG similarity and recency decay.
    
    Attributes:
        base_similarity: Raw vector similarity [0.0, 1.0] from ChromaDB search.
        days_ago: Integer days between reference event and evaluation date.
        recency_decay_factor: Exponential decay multiplier [0.0, 1.0].
        final_weight: base_similarity * recency_decay_factor.
        reference_date: ISO string of the historical event date.
        query_date: ISO string of the date this weight was computed.
    """
    base_similarity: float
    days_ago: int
    recency_decay_factor: float
    final_weight: float
    reference_date: str
    query_date: str


def decay_by_days(
    days_elapsed: int,
    half_life_days: float = 365.0,
    floor_factor: float = 0.05
) -> float:
    """
    Compute exponential decay factor for historical event recency.
    
    Uses half-life model: factor(t) = 0.5^(t / half_life).
    Clamps result to [floor_factor, 1.0] to prevent total erasure of old events.
    
    Args:
        days_elapsed: Days since reference event (non-negative).
        half_life_days: Days at which decay = 0.5 (default 1 year).
        floor_factor: Minimum decay factor to prevent underweighting ancient data.
    
    Returns:
        Decay multiplier in range [floor_factor, 1.0].
    """
    if days_elapsed < 0:
        raise ValueError("days_elapsed must be non-negative")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if not (0.0 <= floor_factor <= 1.0):
        raise ValueError("floor_factor must be in [0.0, 1.0]")
    
    exponent = days_elapsed / half_life_days
    decay = 0.5 ** exponent
    return max(floor_factor, min(1.0, decay))


def compute_weighted_confidence(
    base_similarity: float,
    reference_date: datetime,
    query_date: Optional[datetime] = None,
    half_life_days: float = 365.0,
    floor_factor: float = 0.05
) -> WeightedConfidence:
    """
    Combine RAG similarity with recency decay to produce final WeightedConfidence.
    
    Args:
        base_similarity: Raw ChromaDB cosine similarity [0.0, 1.0].
        reference_date: datetime of historical reference event.
        query_date: datetime of current evaluation (defaults to now).
        half_life_days: Recency decay half-life in days.
        floor_factor: Minimum decay multiplier.
    
    Returns:
        WeightedConfidence dataclass with final_weight = base_similarity * decay_factor.
    
    Raises:
        ValueError: If base_similarity not in [0.0, 1.0] or reference_date is in future.
    """
    if not (0.0 <= base_similarity <= 1.0):
        raise ValueError("base_similarity must be in [0.0, 1.0]")
    
    if query_date is None:
        query_date = datetime.utcnow()
    
    if reference_date > query_date:
        raise ValueError("reference_date cannot be in the future relative to query_date")
    
    days_ago = (query_date - reference_date).days
    decay_factor = decay_by_days(
        days_elapsed=days_ago,
        half_life_days=half_life_days,
        floor_factor=floor_factor
    )
    final_weight = base_similarity * decay_factor
    
    return WeightedConfidence(
        base_similarity=base_similarity,
        days_ago=days_ago,
        recency_decay_factor=decay_factor,
        final_weight=final_weight,
        reference_date=reference_date.isoformat(),
        query_date=query_date.isoformat()
    )


def batch_weighted_confidence(
    similarities_and_dates: list[tuple[float, datetime]],
    query_date: Optional[datetime] = None,
    half_life_days: float = 365.0,
    floor_factor: float = 0.05
) -> list[WeightedConfidence]:
    """
    Compute WeightedConfidence for multiple RAG results in one call.
    
    Args:
        similarities_and_dates: List of (similarity, reference_date) tuples.
        query_date: Evaluation datetime (defaults to now).
        half_life_days: Recency decay half-life in days.
        floor_factor: Minimum decay multiplier.
    
    Returns:
        List of WeightedConfidence objects in input order.
    """
    if query_date is None:
        query_date = datetime.utcnow()
    
    results = []
    for similarity, ref_date in similarities_and_dates:
        wc = compute_weighted_confidence(
            base_similarity=similarity,
            reference_date=ref_date,
            query_date=query_date,
            half_life_days=half_life_days,
            floor_factor=floor_factor
        )
        results.append(wc)
    return results


def normalize_weighted_scores(
    confidences: list[WeightedConfidence]
) -> list[WeightedConfidence]:
    """
    Normalize final_weight across a batch to sum to 1.0 (probability distribution).
    
    Args:
        confidences: List of WeightedConfidence objects.
    
    Returns:
        New list with final_weight normalized; other fields unchanged.
    """
    if not confidences:
        return []
    
    total = sum(wc.final_weight for wc in confidences)
    if total == 0.0:
        return confidences
    
    normalized = []
    for wc in confidences:
        normalized_weight = wc.final_weight / total
        normalized.append(WeightedConfidence(
            base_similarity=wc.base_similarity,
            days_ago=wc.days_ago,
            recency_decay_factor=wc.recency_decay_factor,
            final_weight=normalized_weight,
            reference_date=wc.reference_date,
            query_date=wc.query_date
        ))
    return normalized
