"""
Sentinel Historian: Confidence Score Weighting System

This module synthesizes RAG similarity scores with temporal decay to produce
a final WeightedConfidence float. It combines:
  - Vector similarity scores from ChromaDB RAG lookups (0.0–1.0)
  - Recency decay: older matches contribute less weight
  - Source credibility multipliers (SEC filings > news > social sentiment)
  - Aggregate confidence for final prediction scoring

Used by Judge modules to calibrate prediction certainty before Discord output.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import list


@dataclass
class RAGMatch:
    """Single document match from ChromaDB RAG query."""
    similarity_score: float
    source_type: str  # "sec_filing" | "news" | "sentiment"
    published_date: datetime
    document_id: str


def recency_decay(published_date: datetime, reference_date: datetime | None = None, half_life_days: float = 30.0) -> float:
    """
    Exponential decay function: older documents lose weight.
    
    At half_life_days, decay = 0.5. After 2*half_life_days ≈ 0.25.
    """
    if reference_date is None:
        reference_date = datetime.now()
    age_days = (reference_date - published_date).total_seconds() / 86400.0
    if age_days < 0:
        age_days = 0  # future-dated documents get full weight
    return 2.0 ** (-age_days / half_life_days)


def source_credibility_multiplier(source_type: str) -> float:
    """Return credibility weight: SEC filings > news > sentiment."""
    weights = {
        "sec_filing": 1.0,
        "news": 0.7,
        "sentiment": 0.5,
    }
    return weights.get(source_type, 0.5)


def compute_weighted_confidence(
    rag_matches: list[RAGMatch],
    reference_date: datetime | None = None,
    half_life_days: float = 30.0
) -> float:
    """
    Aggregate weighted confidence from a list of RAG matches.
    
    Returns a float in [0.0, 1.0] representing final prediction confidence.
    Empty input returns 0.0.
    """
    if not rag_matches:
        return 0.0
    
    if reference_date is None:
        reference_date = datetime.now()
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for match in rag_matches:
        # Similarity is already 0–1 from ChromaDB
        similarity = max(0.0, min(1.0, match.similarity_score))
        
        # Decay older documents
        decay = recency_decay(match.published_date, reference_date, half_life_days)
        
        # Apply source credibility
        credibility = source_credibility_multiplier(match.source_type)
        
        # Composite weight
        weight = similarity * decay * credibility
        weighted_sum += weight
        total_weight += weight
    
    if total_weight == 0.0:
        return 0.0
    
    return weighted_sum / total_weight


def compute_confidence_with_decay_curve(
    rag_matches: list[RAGMatch],
    reference_date: datetime | None = None,
    half_life_days: float = 30.0,
    min_similarity_threshold: float = 0.3
) -> dict:
    """
    Advanced weighting: return detailed breakdown (debug + final score).
    
    Filters out low-similarity matches and returns confidence + component scores.
    """
    if not rag_matches:
        return {
            "final_confidence": 0.0,
            "num_matches": 0,
            "num_filtered": 0,
            "breakdown": [],
        }
    
    if reference_date is None:
        reference_date = datetime.now()
    
    breakdown = []
    total_weight = 0.0
    weighted_sum = 0.0
    filtered_count = 0
    
    for match in rag_matches:
        similarity = max(0.0, min(1.0, match.similarity_score))
        
        # Filter low-confidence matches
        if similarity < min_similarity_threshold:
            filtered_count += 1
            continue
        
        decay = recency_decay(match.published_date, reference_date, half_life_days)
        credibility = source_credibility_multiplier(match.source_type)
        weight = similarity * decay * credibility
        
        breakdown.append({
            "document_id": match.document_id,
            "source_type": match.source_type,
            "similarity": similarity,
            "decay": decay,
            "credibility": credibility,
            "weight": weight,
        })
        
        weighted_sum += weight
        total_weight += weight
    
    final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    return {
        "final_confidence": final_confidence,
        "num_matches": len(rag_matches),
        "num_filtered": filtered_count,
        "breakdown": breakdown,
    }


def interpolate_confidence_range(
    low_confidence: float,
    high_confidence: float,
    target_confidence: float
) -> float:
    """
    Scale target_confidence to [low, high] range for prediction output.
    
    Useful for mapping raw confidence to a strategy-specific range.
    """
    target_confidence = max(0.0, min(1.0, target_confidence))
    return low_confidence + (target_confidence * (high_confidence - low_confidence))
