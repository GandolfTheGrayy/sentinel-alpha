"""
Regulatory Whispers Detector — Sentinel Linguist Pillar

Scans SEC filings for hedging language patterns (e.g., 'may', 'subject to',
'could materially') and scores their density. High hedging density signals
management uncertainty and reduced confidence in forward guidance.

Integrates with Sentinel's Linguist reasoning pipeline to flag regulatory
caution signals that may precede downside moves.
"""

import re
from typing import TypedDict, Optional
from collections import defaultdict


class HedgingPattern(TypedDict):
    """Type definition for hedging language pattern metadata."""
    pattern: str
    weight: float
    category: str


class WhispersScore(TypedDict):
    """Type definition for regulatory whispers analysis output."""
    total_hedging_count: int
    hedging_density: float
    category_breakdown: dict[str, int]
    weighted_score: float
    confidence: float
    interpretation: str


# Curated hedging language patterns by category and severity
HEDGING_LEXICON: list[HedgingPattern] = [
    # Uncertainty qualifiers (high weight)
    {"pattern": r"\bmay\b", "weight": 1.0, "category": "uncertainty"},
    {"pattern": r"\bmight\b", "weight": 1.0, "category": "uncertainty"},
    {"pattern": r"\bcould\b", "weight": 0.9, "category": "uncertainty"},
    {"pattern": r"\bpotentially\b", "weight": 0.95, "category": "uncertainty"},
    {"pattern": r"\bpossibly\b", "weight": 0.9, "category": "uncertainty"},
    
    # Risk caveats (medium-high weight)
    {"pattern": r"\bsubject to\b", "weight": 0.85, "category": "risk_caveat"},
    {"pattern": r"\bsubject to material risk", "weight": 1.1, "category": "risk_caveat"},
    {"pattern": r"\bsubject to\s+(?:significant|substantial|material)", "weight": 1.1, "category": "risk_caveat"},
    {"pattern": r"\bcontingent\s+(?:on|upon)\b", "weight": 0.85, "category": "risk_caveat"},
    {"pattern": r"\bdepending on", "weight": 0.75, "category": "risk_caveat"},
    
    # Materiality hedges (high weight)
    {"pattern": r"\bcould materially\b", "weight": 1.2, "category": "materiality"},
    {"pattern": r"\bmaterially\s+(?:adverse|negative|impair)", "weight": 1.15, "category": "materiality"},
    {"pattern": r"\bif.*materialize", "weight": 1.0, "category": "materiality"},
    {"pattern": r"\bimpair\b", "weight": 0.9, "category": "materiality"},
    
    # Assumption-based language (medium weight)
    {"pattern": r"\bassuming\b", "weight": 0.7, "category": "assumption"},
    {"pattern": r"\bif conditions\b", "weight": 0.75, "category": "assumption"},
    {"pattern": r"\bdepending\b", "weight": 0.7, "category": "assumption"},
    {"pattern": r"\bsubject to\s+(?:changes|variation)", "weight": 0.8, "category": "assumption"},
    
    # Risk and uncertainty language (medium weight)
    {"pattern": r"\brisks?\b", "weight": 0.6, "category": "risk_language"},
    {"pattern": r"\buncertain(?:ty|ties)\b", "weight": 0.7, "category": "risk_language"},
    {"pattern": r"\bunexpected\b", "weight": 0.65, "category": "risk_language"},
    {"pattern": r"\badverse", "weight": 0.7, "category": "risk_language"},
    {"pattern": r"\bnegative\s+(?:impact|effect|outcome)", "weight": 0.75, "category": "risk_language"},
    
    # Limitation and caution language (low-medium weight)
    {"pattern": r"\blimitations?\b", "weight": 0.5, "category": "limitation"},
    {"pattern": r"\bcaution\b", "weight": 0.6, "category": "limitation"},
    {"pattern": r"\bshould not\b", "weight": 0.7, "category": "limitation"},
    {"pattern": r"\bno assurance\b", "weight": 0.9, "category": "limitation"},
]


def _preprocess_text(text: str) -> str:
    """Normalize text for pattern matching: lowercase, minimal whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _count_pattern_matches(text: str, pattern: str) -> int:
    """Count non-overlapping occurrences of regex pattern in text."""
    try:
        return len(re.findall(pattern, text, re.IGNORECASE))
    except re.error:
        return 0


def detect_regulatory_whispers(filing_text: str) -> WhispersScore:
    """
    Scan SEC filing text for hedging language and return whispers score.
    
    Args:
        filing_text: Full text of SEC filing (8-K, 10-Q, 10-K, etc.)
    
    Returns:
        WhispersScore dict with hedging counts, density, category breakdown,
        weighted score, confidence, and interpretation.
    """
    if not filing_text or not isinstance(filing_text, str):
        return WhispersScore(
            total_hedging_count=0,
            hedging_density=0.0,
            category_breakdown={},
            weighted_score=0.0,
            confidence=0.0,
            interpretation="Invalid input: empty or non-string filing text."
        )
    
    processed_text = _preprocess_text(filing_text)
    word_count = len(processed_text.split())
    
    if word_count < 50:
        return WhispersScore(
            total_hedging_count=0,
            hedging_density=0.0,
            category_breakdown={},
            weighted_score=0.0,
            confidence=0.0,
            interpretation="Insufficient text: filing too short for reliable analysis."
        )
    
    # Count matches per pattern
    total_hedging_count = 0
    weighted_sum = 0.0
    category_counts: dict[str, int] = defaultdict(int)
    
    for hedge_pattern in HEDGING_LEXICON:
        pattern_str = hedge_pattern["pattern"]
        weight = hedge_pattern["weight"]
        category = hedge_pattern["category"]
        
        count = _count_pattern_matches(processed_text, pattern_str)
        if count > 0:
            total_hedging_count += count
            weighted_sum += count * weight
            category_counts[category] += count
    
    # Calculate density metrics
    hedging_density = total_hedging_count / word_count if word_count > 0 else 0.0
    weighted_score = weighted_sum / word_count if word_count > 0 else 0.0
    
    # Normalize weighted score to 0-1 range (empirically calibrated)
    # Typical 10-K has ~0.01-0.05 weighted density; cap at 0.1 for normalization
    normalized_weighted_score = min(weighted_score / 0.1, 1.0)
    
    # Confidence: higher with more matches and longer text
    confidence = min(
        (total_hedging_count / 10.0) * (word_count / 5000.0),
        1.0
    )
    confidence = max(confidence, 0.0)
    
    # Generate interpretation
    interpretation = _interpret_whispers_score(
        normalized_weighted_score,
        hedging_density,
        total_hedging_count,
        category_counts
    )
    
    return WhispersScore(
        total_hedging_count=total
