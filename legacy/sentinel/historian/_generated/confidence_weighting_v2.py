"""
Confidence score weighting system for Sentinel's RAG pipeline.

This module combines RAG similarity scores with recency decay to produce a
final WeightedConfidence float. Used by historian/rag_query.py to calibrate
how much trust we place in historical context retrieved from ChromaDB.

The weighting model accounts for:
  - Similarity score (0–1) from the embedding search.
  - Recency decay: older events are penalized exponentially.
  - Event type credibility: SEC filings > news > social media.
  - Market volatility context: high-vol events weighted differently.

Final score ranges 0–1, where 1.0 = maximum confidence in the retrieved signal.
"""

from datetime import datetime, timedelta
from typing import TypedDict
import math


class ScoredDocument(TypedDict):
    """Schema for a RAG result with metadata."""
    text: str
    similarity: float
    source: str  # 'sec', 'news', 'reddit', 'github', etc.
    timestamp: datetime  # publication or filing date
    event_type: str  # '8-K', 'headline', 'earnings_whisper', etc.


class WeightedConfidence(TypedDict):
    """Final confidence output for a historical signal."""
    score: float  # 0–1
    component_scores: dict  # { 'similarity': X, 'recency': Y, 'credibility': Z }
    explanation: str


def similarity_weight(similarity: float) -> float:
    """Apply non-linear scaling to embedding similarity (0–1 → 0–1)."""
    # Squash very low similarities; boost high ones slightly
    if similarity < 0.3:
        return 0.0
    return min(1.0, similarity ** 0.8)


def recency_decay(timestamp: datetime, reference_date: datetime | None = None) -> float:
    """
    Exponential decay: recent events ~ 1.0, old events → 0.
    
    Half-life = 90 days. Event older than 2 years → ~0.01.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    days_ago = (reference_date - timestamp).days
    if days_ago < 0:
        return 0.0  # future timestamp
    
    half_life = 90  # days
    decay = math.exp(-0.693 * days_ago / half_life)
    return max(0.01, min(1.0, decay))


def credibility_weight(source: str, event_type: str) -> float:
    """
    Assign source + event-type credibility (0–1).
    
    SEC filings (8-K, 10-Q, insider trades) > news headlines > social sentiment.
    """
    source_weights = {
        'sec': 0.95,
        'earnings_call': 0.90,
        'news': 0.75,
        'reddit': 0.50,
        'github': 0.60,
        'twitter': 0.40,
    }
    
    event_weights = {
        '8-K': 0.98,
        '10-Q': 0.95,
        '10-K': 0.95,
        'insider_trade': 0.85,
        'earnings_whisper': 0.70,
        'headline': 0.70,
        'developer_sentiment': 0.65,
        'social': 0.50,
    }
    
    s = source_weights.get(source, 0.5)
    e = event_weights.get(event_type, 0.5)
    return (s + e) / 2


def combine_weights(
    similarity: float,
    recency: float,
    credibility: float,
    volatility_multiplier: float = 1.0
) -> float:
    """
    Harmonic-mean-inspired blend: credibility & recency gating, similarity as signal strength.
    
    In high-volatility markets, we slightly boost historical confidence (more signals needed).
    """
    # Credibility and recency act as gates; similarity is the signal.
    # If recency is very low, the score plummets regardless of similarity.
    gated = credibility * recency * similarity
    
    # Volatility multiplier: in volatile markets, historical events carry slightly more weight
    # (we need all available signals). Max boost = 1.2x.
    final = gated * min(1.2, volatility_multiplier)
    
    return max(0.0, min(1.0, final))


def compute_weighted_confidence(
    doc: ScoredDocument,
    reference_date: datetime | None = None,
    volatility_multiplier: float = 1.0,
) -> WeightedConfidence:
    """
    Compute final confidence score for a RAG-retrieved document.
    
    Args:
        doc: Retrieved document with similarity, source, timestamp, event_type.
        reference_date: Date to measure recency from (default: now).
        volatility_multiplier: Market volatility factor (1.0–1.2).
    
    Returns:
        WeightedConfidence dict with score, component breakdown, and explanation.
    """
    if reference_date is None:
        reference_date = datetime.utcnow()
    
    sim = similarity_weight(doc['similarity'])
    rec = recency_decay(doc['timestamp'], reference_date)
    cred = credibility_weight(doc['source'], doc['event_type'])
    
    final = combine_weights(sim, rec, cred, volatility_multiplier)
    
    explanation = (
        f"Source={doc['source']} ({cred:.2f}), "
        f"Recency={rec:.2f} (from {doc['timestamp'].date()}), "
        f"Similarity={sim:.2f}, "
        f"VolMult={volatility_multiplier:.2f} → {final:.3f}"
    )
    
    return WeightedConfidence(
        score=final,
        component_scores={
            'similarity': sim,
            'recency': rec,
            'credibility': cred,
            'volatility_multiplier': volatility_multiplier,
        },
        explanation=explanation,
    )


def batch_weight_rag_results(
    docs: list[ScoredDocument],
    reference_date: datetime | None = None,
    volatility_multiplier: float = 1.0,
) -> list[WeightedConfidence]:
    """
    Apply weighting to a batch of RAG results (e.g., top-5 ChromaDB hits).
    
    Args:
        docs: List of retrieved documents.
        reference_date: Reference date for recency decay.
        volatility_multiplier: Market volatility factor.
    
    Returns:
        List of WeightedConfidence dicts, sorted by score (highest first).
    """
    weighted = [
        compute_weighted_confidence(doc, reference_date, volatility_multiplier)
        for doc in docs
    ]
    return sorted(weighted, key=lambda x: x['score'], reverse=True)


def aggregate_confidence(
    weighted_results: list[WeightedConfidence],
    method: str = 'weighted_mean',
) -> float:
    """
    Aggregate multiple weighted confidences into a single signal strength (0–1).
    
    Args:
        weighted_results: List of WeightedConfidence dicts.
        method: 'weighted_mean' (default) or 'max'.
    
    Returns:
        Aggregated confidence score (0–1).
    """
    if not weighted_results:
        return 0.0
    
    if method == 'max':
        return max((r['score'] for r in weighted_results), default=0.0)
    
    # weighted_mean: higher-scored results dominate.
    scores = [r['score'] for r in weighted_results]
    total_weight = sum(scores)
    if total_weight == 0:
        return 0.0
    return sum(s * s for s in scores) / total_weight


if __name__ == '__main__':
    # Smoke test: create mock RAG
