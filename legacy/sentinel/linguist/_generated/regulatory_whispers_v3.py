"""Regulatory Whispers detector for Sentinel Sentiment Engine.

Scans SEC filings for hedging language patterns (e.g. 'may', 'subject to',
'could materially') and scores their density. Higher density indicates elevated
regulatory caution or uncertainty, which may precede market moves.

Used by linguist pillar to flag filing-level tone shifts and feed into
Judge predictions via RAG context enrichment.
"""

import re
from typing import Dict, List, Tuple
import sqlite3


# Hedging language patterns organized by severity tier
HEDGING_PATTERNS = {
    "high_certainty": [
        r"\bmay\b",
        r"\bcould\b",
        r"\bmight\b",
        r"\bpotentially\b",
        r"\bpossibly\b",
        r"\blikely\b",
        r"\bappears\b",
        r"\bsuggests\b",
    ],
    "regulatory_caution": [
        r"\bsubject\s+to\b",
        r"\bsubject\s+to\s+(?:the\s+)?risks?\b",
        r"\bcould\s+materially\b",
        r"\bmay\s+materially\b",
        r"\bcontinuing\s+uncertainty\b",
        r"\bto\s+the\s+extent\b",
        r"\bas\s+(?:a\s+)?condition\b",
        r"\bcontingent\s+(?:upon|on)\b",
    ],
    "risk_language": [
        r"\brisk\b",
        r"\brisk[s]?\b",
        r"\bunable\s+to\b",
        r"\bfailure\s+(?:to|of)\b",
        r"\badverse\b",
        r"\bdisruption\b",
        r"\blitigation\b",
        r"\bregulatory\s+(?:action|change|risk)\b",
        r"\bcompliance\b",
    ],
    "uncertainty_markers": [
        r"\buncertain\b",
        r"\buncertainty\b",
        r"\bvariable\b",
        r"\bunpredictable\b",
        r"\bunexpected\b",
        r"\billusor(?:y|iness)\b",
    ],
}

# Severity weights: higher weight = more concerning
SEVERITY_WEIGHTS = {
    "high_certainty": 1.0,
    "regulatory_caution": 2.5,
    "risk_language": 1.5,
    "uncertainty_markers": 1.8,
}


def tokenize_text(text: str) -> List[str]:
    """Normalize and tokenize text into words."""
    text = text.lower()
    tokens = re.findall(r"\b[\w\-]+\b", text)
    return tokens


def compile_patterns() -> Dict[str, List]:
    """Pre-compile regex patterns for efficiency."""
    compiled = {}
    for category, patterns in HEDGING_PATTERNS.items():
        compiled[category] = [re.compile(p, re.IGNORECASE) for p in patterns]
    return compiled


def detect_hedging_in_text(text: str) -> Dict[str, int]:
    """Count hedging patterns in text, grouped by category.
    
    Returns a dict mapping category name to match count.
    """
    compiled = compile_patterns()
    counts = {cat: 0 for cat in HEDGING_PATTERNS.keys()}
    
    for category, patterns in compiled.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            counts[category] += len(matches)
    
    return counts


def compute_hedging_score(text: str, normalize_by_length: bool = True) -> float:
    """Compute weighted hedging score (0.0 to 1.0+).
    
    Higher score indicates more hedging language and regulatory caution.
    If normalize_by_length, score is per-1000-words to control for filing length.
    """
    counts = detect_hedging_in_text(text)
    
    # Weighted sum
    weighted_sum = sum(
        counts[cat] * SEVERITY_WEIGHTS.get(cat, 1.0)
        for cat in counts.keys()
    )
    
    if normalize_by_length:
        word_count = len(tokenize_text(text))
        if word_count == 0:
            return 0.0
        # Normalize to per-1000-words
        hedging_density = (weighted_sum / word_count) * 1000.0
        return min(hedging_density, 5.0)  # Cap at 5.0 for interpretability
    else:
        return weighted_sum


def analyze_filing(filing_text: str, ticker: str = "") -> Dict:
    """Analyze a single SEC filing for regulatory whispers.
    
    Returns dict with keys:
      - ticker: stock ticker
      - raw_score: unweighted hedging count
      - normalized_score: per-1000-words hedging density
      - category_breakdown: dict of counts by hedging category
      - risk_level: 'low', 'medium', 'high', 'critical' based on score
    """
    if not filing_text or not isinstance(filing_text, str):
        return {
            "ticker": ticker,
            "raw_score": 0,
            "normalized_score": 0.0,
            "category_breakdown": {},
            "risk_level": "unknown",
        }
    
    counts = detect_hedging_in_text(filing_text)
    normalized = compute_hedging_score(filing_text, normalize_by_length=True)
    
    # Risk level assignment
    if normalized >= 3.0:
        risk_level = "critical"
    elif normalized >= 2.0:
        risk_level = "high"
    elif normalized >= 1.0:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    raw_score = sum(counts.values())
    
    return {
        "ticker": ticker,
        "raw_score": raw_score,
        "normalized_score": round(normalized, 3),
        "category_breakdown": {k: int(v) for k, v in counts.items()},
        "risk_level": risk_level,
    }


def compare_filings(filing_a: str, filing_b: str, ticker: str = "") -> Dict:
    """Compare hedging density between two filings (drift detection).
    
    Returns dict with:
      - score_a, score_b: normalized scores for each filing
      - drift: absolute change (score_b - score_a)
      - drift_pct: percentage change
      - direction: 'increasing_caution', 'decreasing_caution', or 'stable'
    """
    score_a = compute_hedging_score(filing_a, normalize_by_length=True)
    score_b = compute_hedging_score(filing_b, normalize_by_length=True)
    
    drift = score_b - score_a
    drift_pct = (drift / max(score_a, 0.1)) * 100.0 if score_a > 0 else 0.0
    
    if drift > 0.2:
        direction = "increasing_caution"
    elif drift < -0.2:
        direction = "decreasing_caution"
    else:
        direction = "stable"
    
    return {
        "ticker": ticker,
        "score_a": round(score_a, 3),
        "score_b": round(score_b, 3),
        "drift": round(drift, 3),
        "drift_pct": round(drift_pct, 1),
        "direction": direction,
    }


def batch_analyze_filings(filings: List[Tuple[str, str]]) -> List[Dict]:
    """Analyze multiple (text, ticker) tuples in batch.
    
    Returns list of analysis dicts (one per filing).
    """
    results = []
    for filing_text, ticker in fi
