"""
Confidence score weighting system for Sentinel Historian.

Combines RAG similarity scores with recency decay to produce a final
WeightedConfidence float. Used by Judge to modulate prediction strength
based on the relevance and freshness of historical precedents retrieved
from ChromaDB.

Key insight: A high-similarity match from 5 years ago should carry less
weight than a moderate-similarity match from last month. This module
balances both signals via configurable decay curves and normalization.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence


@dataclass
class SimilarityRecord:
    """Represents a single retrieved document with similarity and age metadata."""
    similarity_score: float
    timestamp: datetime
    source: str


def recency_decay(
    timestamp: datetime,
    reference_date: datetime,
    half_life_days: float = 180.0,
) -> float:
    """
    Compute exponential decay factor for document age.

    Decay follows: exp(-ln(2) * age_days / half_life_days).
    At half_life_days, decay = 0.5. Beyond that, exponentially smaller.
    Returns value in (0, 1], where 1 = today, 0 = infinitely old.
    """
    age_days = (reference_date - timestamp).total_seconds() / (24 * 3600)
    age_days = max(0, age_days)
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return decay


def normalize_similarity(scores: Sequence[float]) -> Sequence[float]:
    """
    Min-max normalize similarity scores to [0, 1] range.

    If all scores are identical or empty, returns uniform [1.0, 1.0, ...].
    Ensures no division by zero.
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def compute_weighted_confidence(
    records: Sequence[SimilarityRecord],
    reference_date: datetime | None = None,
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 180.0,
) -> float:
    """
    Combine RAG similarity scores with recency decay into a single confidence float.

    Args:
        records: Sequence of SimilarityRecord objects from RAG retrieval.
        reference_date: Baseline date for age calculation (default: now).
        similarity_weight: Coefficient for normalized similarity (0–1).
        recency_weight: Coefficient for recency decay (0–1).
        half_life_days: Exponential decay half-life in days.

    Returns:
        Weighted confidence score in [0, 1].
        0 = no records or all very old/dissimilar.
        1 = perfect recent match.
    """
    if not records:
        return 0.0

    if reference_date is None:
        reference_date = datetime.utcnow()

    similarities = [r.similarity_score for r in records]
    normalized_sims = normalize_similarity(similarities)

    decays = [
        recency_decay(r.timestamp, reference_date, half_life_days)
        for r in records
    ]

    combined_scores = [
        similarity_weight * norm_sim + recency_weight * decay
        for norm_sim, decay in zip(normalized_sims, decays)
    ]

    average_confidence = sum(combined_scores) / len(combined_scores)
    return min(1.0, max(0.0, average_confidence))


def confidence_with_max_only(
    records: Sequence[SimilarityRecord],
    reference_date: datetime | None = None,
    half_life_days: float = 180.0,
) -> float:
    """
    Return confidence based on single best-match record (highest similarity).

    Useful for Judge scenarios where one strong precedent outweighs many weak ones.
    Returns recency-decayed similarity of the top match.
    """
    if not records:
        return 0.0

    if reference_date is None:
        reference_date = datetime.utcnow()

    best = max(records, key=lambda r: r.similarity_score)
    decay = recency_decay(best.timestamp, reference_date, half_life_days)
    weighted = best.similarity_score * decay
    return min(1.0, max(0.0, weighted))


def confidence_percentile(
    records: Sequence[SimilarityRecord],
    percentile: float = 75.0,
    reference_date: datetime | None = None,
    half_life_days: float = 180.0,
) -> float:
    """
    Return confidence based on percentile-ranked record (e.g., 75th percentile).

    Balances top match with broader signal. Useful for robust weighting.
    """
    if not records:
        return 0.0

    if reference_date is None:
        reference_date = datetime.utcnow()

    sorted_by_sim = sorted(records, key=lambda r: r.similarity_score)
    idx = int(len(sorted_by_sim) * percentile / 100.0)
    idx = min(idx, len(sorted_by_sim) - 1)
    selected = sorted_by_sim[idx]

    decay = recency_decay(selected.timestamp, reference_date, half_life_days)
    weighted = selected.similarity_score * decay
    return min(1.0, max(0.0, weighted))


if __name__ == "__main__":
    now = datetime.utcnow()
    records = [
        SimilarityRecord(0.95, now - timedelta(days=5), "recent_10k"),
        SimilarityRecord(0.87, now - timedelta(days=90), "prior_earnings"),
        SimilarityRecord(0.72, now - timedelta(days=365), "historical_sec"),
        SimilarityRecord(0.65, now - timedelta(days=730), "old_news"),
    ]

    conf_avg = compute_weighted_confidence(records, reference_date=now)
    print(f"Average-based confidence: {conf_avg:.4f}")

    conf_max = confidence_with_max_only(records, reference_date=now)
    print(f"Max-only confidence: {conf_max:.4f}")

    conf_p75 = confidence_percentile(records, percentile=75.0, reference_date=now)
    print(f"75th-percentile confidence: {conf_p75:.4f}")
