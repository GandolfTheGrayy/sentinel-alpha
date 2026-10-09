"""
Confidence Score Weighting System for Sentinel Historian.

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. Used by the Judge to calibrate prediction
certainty based on both semantic relevance (via ChromaDB) and temporal
freshness of historical precedents.

Exports:
  - compute_weighted_confidence(): Fuse similarity + recency into [0,1] score
  - apply_recency_decay(): Time-based penalty for stale historical data
  - normalize_similarities(): Vectorize similarity array to [0,1] range
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import numpy as np


def apply_recency_decay(
    document_timestamp: datetime,
    reference_date: Optional[datetime] = None,
    half_life_days: float = 180.0,
) -> float:
    """
    Apply exponential decay to a document's confidence based on age.

    Args:
        document_timestamp: When the historical document was created.
        reference_date: Current date (defaults to now). Used for testing.
        half_life_days: Days until similarity score decays to 50%. Default 180.

    Returns:
        Decay factor in [0, 1]. Newer docs → 1.0, older docs → approaching 0.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    age_days = (reference_date - document_timestamp).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0.0

    decay = math.exp(-0.693147 * age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def normalize_similarities(
    similarity_scores: List[float],
) -> np.ndarray:
    """
    Normalize raw similarity scores (e.g., cosine distances) to [0, 1] range.

    Args:
        similarity_scores: List of raw similarity values from ChromaDB.

    Returns:
        NumPy array of normalized scores in [0, 1].
    """
    if not similarity_scores:
        return np.array([])

    arr = np.array(similarity_scores, dtype=np.float64)

    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val - min_val < 1e-9:
        return np.ones_like(arr)

    normalized = (arr - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0)


def compute_weighted_confidence(
    similarity_scores: List[float],
    document_timestamps: List[datetime],
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 180.0,
    reference_date: Optional[datetime] = None,
) -> float:
    """
    Combine RAG similarity scores with recency decay into a single confidence.

    Fuses two signals:
      1. Semantic relevance (normalized similarity scores from ChromaDB).
      2. Temporal freshness (exponential decay based on document age).

    Weights are normalized internally, so they need not sum to 1.0.

    Args:
        similarity_scores: List of cosine/distance scores from RAG query.
        document_timestamps: Corresponding creation dates of retrieved docs.
        similarity_weight: Importance of semantic relevance (default 0.6).
        recency_weight: Importance of temporal freshness (default 0.4).
        half_life_days: Days for recency decay to reach 50% (default 180).
        reference_date: Current date for decay calc (defaults to now).

    Returns:
        Final WeightedConfidence in [0, 1]. Higher = more trustworthy signal.

    Raises:
        ValueError: If lists are empty or length-mismatched.
    """
    if not similarity_scores or not document_timestamps:
        raise ValueError("similarity_scores and document_timestamps must not be empty")

    if len(similarity_scores) != len(document_timestamps):
        raise ValueError(
            f"Length mismatch: {len(similarity_scores)} scores vs "
            f"{len(document_timestamps)} timestamps"
        )

    if reference_date is None:
        reference_date = datetime.utcnow()

    normalized_sims = normalize_similarities(similarity_scores)

    recency_decays = np.array(
        [
            apply_recency_decay(ts, reference_date, half_life_days)
            for ts in document_timestamps
        ],
        dtype=np.float64,
    )

    total_weight = similarity_weight + recency_weight
    norm_sim_weight = similarity_weight / total_weight
    norm_rec_weight = recency_weight / total_weight

    combined_scores = (
        norm_sim_weight * normalized_sims + norm_rec_weight * recency_decays
    )

    final_confidence = float(np.mean(combined_scores))
    return max(0.0, min(1.0, final_confidence))


def batch_compute_weighted_confidence(
    batch_results: List[Tuple[List[float], List[datetime]]],
    similarity_weight: float = 0.6,
    recency_weight: float = 0.4,
    half_life_days: float = 180.0,
    reference_date: Optional[datetime] = None,
) -> List[float]:
    """
    Compute WeightedConfidence for multiple RAG query results in parallel.

    Args:
        batch_results: List of (similarity_scores, timestamps) tuples.
        similarity_weight: Importance of semantic relevance.
        recency_weight: Importance of temporal freshness.
        half_life_days: Decay half-life in days.
        reference_date: Reference date for all decay calculations.

    Returns:
        List of WeightedConfidence floats, one per batch item.
    """
    confidences = []
    for sims, timestamps in batch_results:
        try:
            conf = compute_weighted_confidence(
                sims,
                timestamps,
                similarity_weight=similarity_weight,
                recency_weight=recency_weight,
                half_life_days=half_life_days,
                reference_date=reference_date,
            )
            confidences.append(conf)
        except ValueError:
            confidences.append(0.0)

    return confidences
