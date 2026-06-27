"""
Sentinel Historian: Confidence Score Weighting System

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. It bridges the Historian's vector retrieval
(raw similarity) with the Judge's prediction pipeline, ensuring that older
or less-relevant context is down-weighted in real-time sentiment analysis.

Used by: sentinel/judge/predictor.py (to calibrate prediction certainty)
         sentinel/historian/rag_query.py (to annotate retrieval results)
"""

import math
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoreComponent:
    """A single similarity score with metadata for weighting."""
    similarity: float
    timestamp: datetime
    source_type: str  # "sec_filing", "news", "reddit", "github"
    relevance_label: Optional[str] = None


@dataclass
class WeightedConfidence:
    """Final confidence output combining similarity and recency."""
    final_score: float
    similarity_component: float
    recency_component: float
    source_weights: dict
    num_sources: int


def recency_decay(
    timestamp: datetime,
    reference_time: Optional[datetime] = None,
    half_life_days: float = 30.0
) -> float:
    """
    Exponential decay factor based on age of data relative to reference time.
    
    Decay follows: decay = exp(-ln(2) * age_days / half_life_days)
    At half_life_days, decay = 0.5. Older data decays toward 0.
    
    Args:
        timestamp: When the signal was collected.
        reference_time: Current time (defaults to now).
        half_life_days: Days until decay reaches 0.5.
    
    Returns:
        Decay multiplier in [0, 1].
    """
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    age = reference_time - timestamp
    age_days = age.total_seconds() / (24 * 3600)
    
    if age_days < 0:
        age_days = 0
    
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def source_weight(source_type: str) -> float:
    """
    Assign intrinsic credibility weight to a source category.
    
    Args:
        source_type: One of "sec_filing", "news", "reddit", "github".
    
    Returns:
        Weight in [0, 1]; higher = more trusted.
    """
    weights = {
        "sec_filing": 1.0,
        "news": 0.85,
        "reddit": 0.60,
        "github": 0.75,
    }
    return weights.get(source_type, 0.50)


def normalize_similarity(raw_sim: float, clip_min: float = 0.0) -> float:
    """
    Normalize raw cosine/embedding similarity to [0, 1] range.
    
    Args:
        raw_sim: Raw similarity score (often from ChromaDB).
        clip_min: Minimum threshold; scores below are clipped to 0.
    
    Returns:
        Normalized similarity in [0, 1].
    """
    norm = max(clip_min, raw_sim)
    return min(1.0, norm)


def combine_weighted_confidence(
    components: List[ScoreComponent],
    reference_time: Optional[datetime] = None,
    half_life_days: float = 30.0,
    similarity_weight: float = 0.65,
    recency_weight: float = 0.35
) -> WeightedConfidence:
    """
    Combine multiple RAG/sentiment signals into a single confidence score.
    
    Merges similarity (from embeddings) with recency decay (temporal freshness)
    and source credibility (SEC > news > github > reddit) into a final float.
    
    Args:
        components: List of ScoreComponent objects to fuse.
        reference_time: Current time for age calculation (defaults to now).
        half_life_days: Decay half-life in days.
        similarity_weight: Fraction of final score from similarity [0, 1].
        recency_weight: Fraction of final score from recency [0, 1].
    
    Returns:
        WeightedConfidence object with final_score, components, and breakdown.
    
    Raises:
        ValueError: If components list is empty.
    """
    if not components:
        raise ValueError("Must provide at least one ScoreComponent.")
    
    if reference_time is None:
        reference_time = datetime.utcnow()
    
    # Normalize similarity and apply source weights
    weighted_sims = []
    source_tally = {}
    
    for comp in components:
        norm_sim = normalize_similarity(comp.similarity)
        src_weight = source_weight(comp.source_type)
        weighted_sim = norm_sim * src_weight
        weighted_sims.append(weighted_sim)
        
        source_tally[comp.source_type] = source_tally.get(comp.source_type, 0) + 1
    
    # Average weighted similarity
    avg_similarity = sum(weighted_sims) / len(weighted_sims) if weighted_sims else 0.0
    
    # Compute recency: average decay across all components
    decays = [
        recency_decay(comp.timestamp, reference_time, half_life_days)
        for comp in components
    ]
    avg_recency = sum(decays) / len(decays) if decays else 0.0
    
    # Blend similarity and recency
    total_weight = similarity_weight + recency_weight
    if total_weight <= 0:
        raise ValueError("similarity_weight and recency_weight cannot both be zero.")
    
    similarity_component = (avg_similarity * similarity_weight) / total_weight
    recency_component = (avg_recency * recency_weight) / total_weight
    final_score = similarity_component + recency_component
    
    return WeightedConfidence(
        final_score=final_score,
        similarity_component=similarity_component,
        recency_component=recency_component,
        source_weights={st: source_weight(st) for st in source_tally.keys()},
        num_sources=len(source_tally)
    )


def apply_confidence_threshold(
    confidence: WeightedConfidence,
    threshold: float = 0.50
) -> bool:
    """
    Check if confidence score meets minimum threshold for signal inclusion.
    
    Args:
        confidence: WeightedConfidence object.
        threshold: Minimum acceptable final_score [0, 1].
    
    Returns:
        True if final_score >= threshold, else False.
    """
    return confidence.final_score >= threshold


def confidence_to_certainty_level(confidence: WeightedConfidence) -> str:
    """
    Map numeric confidence to human-readable certainty level.
    
    Args:
        confidence: WeightedConfidence object.
    
    Returns:
        One of: "very_low", "low", "moderate", "high", "very_high".
    """
    score = confidence.final_score
    if score >= 0.85:
        return "very_high"
    elif score >= 0.70:
        return "high"
    elif score >= 0.50:
        return "moderate"
    elif score >= 0.30:
        return "low"
    else:
        return "very_low"
