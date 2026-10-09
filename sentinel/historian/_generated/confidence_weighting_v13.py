"""
Confidence Score Weighting System for Sentinel Historian.

This module combines RAG similarity scores with recency decay to produce
a final WeightedConfidence float. Used by Judge to calibrate prediction
certainty based on how relevant and temporally proximate historical
events are to the current market signal.

Core insight: A 0.95 similarity score from 5 years ago is worth less than
a 0.85 similarity from last week. This module quantifies that decay.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List


@dataclass
class HistoricalMatch:
    """Represents a single RAG retrieval result with metadata."""
    similarity_score: float
    """Embedding cosine similarity [0.0, 1.0]."""
    event_date: datetime
    """When the historical event occurred."""
    source: str
    """E.g., 'SEC_8K', 'NEWS', 'REDDIT'."""
    snippet: str
    """The matched text or headline."""


def recency_decay(
    event_date: datetime,
    reference_date: datetime | None = None,
    half_life_days: int = 365,
) -> float:
    """
    Compute exponential decay weight based on event age.

    Args:
        event_date: When the historical event occurred.
        reference_date: Anchor point (defaults to now). Used for testing.
        half_life_days: Days until weight drops to 0.5. Default 365 (1 year).

    Returns:
        Decay weight in (0.0, 1.0]. At half_life_days, returns 0.5.
    """
    if reference_date is None:
        reference_date = datetime.now()

    if event_date > reference_date:
        return 1.0

    age_days = max(0, (reference_date - event_date).days)
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    return max(0.01, min(1.0, decay))


def similarity_confidence(raw_similarity: float) -> float:
    """
    Transform raw embedding similarity [0.0, 1.0] into confidence weight.

    Args:
        raw_similarity: Cosine similarity from embedding comparison.

    Returns:
        Confidence weight [0.0, 1.0]. Uses sigmoid-like squashing
        to amplify high-similarity matches.
    """
    raw_similarity = max(0.0, min(1.0, raw_similarity))

    if raw_similarity < 0.5:
        return raw_similarity ** 2

    return 1.0 - (1.0 - raw_similarity) ** 2


def source_credibility_bonus(source: str) -> float:
    """
    Return credibility multiplier for a given source type.

    Args:
        source: One of 'SEC_8K', 'SEC_10Q', 'NEWS', 'REDDIT', 'GITHUB'.

    Returns:
        Multiplier [0.7, 1.3] applied to final confidence.
    """
    bonuses = {
        "SEC_8K": 1.3,
        "SEC_10Q": 1.25,
        "NEWS": 1.0,
        "REDDIT": 0.85,
        "GITHUB": 0.9,
    }
    return bonuses.get(source.upper(), 1.0)


def compute_weighted_confidence(
    matches: List[HistoricalMatch],
    reference_date: datetime | None = None,
    half_life_days: int = 365,
    top_k: int | None = None,
) -> float:
    """
    Combine RAG similarity + recency into a single WeightedConfidence score.

    Args:
        matches: List of HistoricalMatch objects from RAG retrieval.
        reference_date: Anchor for recency calculation (defaults to now).
        half_life_days: Decay half-life in days.
        top_k: If set, only use top k matches by similarity. Else use all.

    Returns:
        WeightedConfidence float in [0.0, 1.0]. Represents how much
        historical precedent supports the current prediction.
    """
    if not matches:
        return 0.0

    if top_k is not None:
        matches = sorted(matches, key=lambda m: m.similarity_score, reverse=True)[
            :top_k
        ]

    if reference_date is None:
        reference_date = datetime.now()

    total_weight = 0.0
    total_confidence = 0.0

    for match in matches:
        sim_conf = similarity_confidence(match.similarity_score)
        recency = recency_decay(match.event_date, reference_date, half_life_days)
        source_bonus = source_credibility_bonus(match.source)

        combined = sim_conf * recency * source_bonus
        total_weight += combined
        total_confidence += combined

    if not matches:
        return 0.0

    avg_weight = total_weight / len(matches)
    return min(1.0, max(0.0, avg_weight))


def confidence_tier(weighted_confidence: float) -> str:
    """
    Categorize confidence into human-readable tiers.

    Args:
        weighted_confidence: Output from compute_weighted_confidence().

    Returns:
        Tier string: 'HIGH', 'MEDIUM', 'LOW', or 'INSUFFICIENT'.
    """
    if weighted_confidence >= 0.75:
        return "HIGH"
    elif weighted_confidence >= 0.5:
        return "MEDIUM"
    elif weighted_confidence >= 0.25:
        return "LOW"
    else:
        return "INSUFFICIENT"


def explain_confidence(
    matches: List[HistoricalMatch],
    reference_date: datetime | None = None,
) -> dict:
    """
    Return detailed breakdown of how confidence score was computed.

    Args:
        matches: List of HistoricalMatch objects.
        reference_date: Anchor for recency (defaults to now).

    Returns:
        Dict with keys: 'weighted_confidence', 'tier', 'components'.
        'components' is a list of dicts, one per match, showing
        sim_conf, recency, source_bonus, and combined scores.
    """
    if not matches:
        return {
            "weighted_confidence": 0.0,
            "tier": "INSUFFICIENT",
            "components": [],
        }

    if reference_date is None:
        reference_date = datetime.now()

    components = []
    for match in matches:
        sim_conf = similarity_confidence(match.similarity_score)
        recency = recency_decay(match.event_date, reference_date)
        source_bonus = source_credibility_bonus(match.source)
        combined = sim_conf * recency * source_bonus

        components.append(
            {
                "source": match.source,
                "event_date": match.event_date.isoformat(),
                "similarity": match.similarity_score,
                "sim_confidence": sim_conf,
                "recency_weight": recency,
                "source_bonus": source_bonus,
                "combined_score": combined,
                "snippet": match.snippet[:100],
            }
        )

    final_confidence = compute_weighted_confidence(matches, reference_date)
    tier = confidence_tier(final_confidence)

    return {
        "weighted_confidence": final_confidence,
        "tier": tier,
        "components": components,
    }
