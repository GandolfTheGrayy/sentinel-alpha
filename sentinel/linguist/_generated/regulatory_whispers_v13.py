"""
Regulatory Whispers Detector for Sentinel Sentiment Engine.

Scans SEC filings for hedging language patterns (e.g. 'may', 'subject to', 
'could materially') and scores their density as a proxy for management caution.
Integrates with the Linguist pillar to augment certainty scoring with 
regulatory risk signals.

Exported functions:
  - scan_filing_text: Analyzes raw SEC text and returns hedge density + matches.
  - compute_whisper_score: Converts hedge density into a 0-1 confidence penalty.
"""

import re
from typing import Dict, List, Tuple


# Hedging language patterns organized by severity.
HEDGING_PATTERNS = {
    "strong_hedge": [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bif\b",
        r"\bdependent\s+on\b",
    ],
    "material_risk": [
        r"\bsubject\s+to\b",
        r"\bcould\s+materially\b",
        r"\bmay\s+materially\b",
        r"\brisks?\s+include\b",
        r"\badverse\s+effects?\b",
    ],
    "caution_language": [
        r"\buncertain(?:ty|ties)?\b",
        r"\bunpredictable\b",
        r"\bvolatile\b",
        r"\bunforeseen\b",
        r"\bno\s+guarantee\b",
    ],
    "contingency": [
        r"\bunless\b",
        r"\bexcept\s+(?:where|as|if)\b",
        r"\bto\s+the\s+extent\b",
        r"\bsubject\s+to\s+(?:certain\s+)?conditions\b",
        r"\bcontingent\s+on\b",
    ],
}

# Weights for each severity tier when computing aggregate score.
SEVERITY_WEIGHTS = {
    "strong_hedge": 1.0,
    "material_risk": 2.0,
    "caution_language": 1.5,
    "contingency": 1.0,
}


def scan_filing_text(
    text: str, case_sensitive: bool = False
) -> Dict[str, any]:
    """
    Scan SEC filing text for hedging language patterns and return density metrics.

    Args:
        text: Raw SEC filing content (typically from 8-K, 10-Q, or 10-K).
        case_sensitive: If False (default), patterns match case-insensitively.

    Returns:
        Dict with keys:
          - "total_hedges": Total count of hedging matches across all tiers.
          - "by_severity": Dict mapping severity tier to match count.
          - "matches": List of (pattern_type, match_text, start_pos, end_pos) tuples.
          - "word_count": Total words in input text (for density normalization).
          - "hedge_density": Hedges per 1000 words.
    """
    if not text:
        return {
            "total_hedges": 0,
            "by_severity": {tier: 0 for tier in HEDGING_PATTERNS},
            "matches": [],
            "word_count": 0,
            "hedge_density": 0.0,
        }

    # Normalize text: strip extra whitespace, convert case if needed.
    normalized = text.strip()
    if not case_sensitive:
        normalized = normalized.lower()

    flags = 0 if case_sensitive else re.IGNORECASE

    # Count total words (rough estimate).
    word_count = len(normalized.split())

    matches: List[Tuple[str, str, int, int]] = []
    severity_counts: Dict[str, int] = {tier: 0 for tier in HEDGING_PATTERNS}
    total_hedges = 0

    # Scan each severity tier.
    for tier, patterns in HEDGING_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, flags):
                matches.append((tier, match.group(), match.start(), match.end()))
                severity_counts[tier] += 1
                total_hedges += 1

    # Compute density: hedges per 1000 words.
    hedge_density = (total_hedges / word_count * 1000) if word_count > 0 else 0.0

    return {
        "total_hedges": total_hedges,
        "by_severity": severity_counts,
        "matches": matches,
        "word_count": word_count,
        "hedge_density": hedge_density,
    }


def compute_whisper_score(hedge_density: float) -> float:
    """
    Convert hedge density (hedges per 1000 words) into a confidence penalty (0-1).

    Higher density → lower score (more caution). Calibrated empirically:
      - 0-5 hedges/1000 words: near-neutral (0.05 penalty).
      - 5-15 hedges/1000 words: moderate caution (linear 0.05-0.25).
      - 15+ hedges/1000 words: high caution, capped at 0.50 penalty.

    Args:
        hedge_density: Hedges per 1000 words from scan_filing_text.

    Returns:
        Float in [0.0, 0.5] representing confidence penalty to subtract from
        base sentiment score. 0.0 = no penalty, 0.5 = strong caution signal.
    """
    if hedge_density < 5:
        return 0.05
    elif hedge_density < 15:
        # Linear interpolation: 5→0.05, 15→0.25
        return 0.05 + (hedge_density - 5) / 10 * 0.20
    else:
        # Asymptotic approach to 0.50 cap
        return min(0.25 + (hedge_density - 15) / 50 * 0.25, 0.50)


def weighted_hedge_score(scan_result: Dict[str, any]) -> float:
    """
    Compute severity-weighted hedge score from scan results.

    Multiplies each severity tier count by its weight, sums, and normalizes
    by word count to produce a 0-1 risk intensity metric.

    Args:
        scan_result: Output dict from scan_filing_text.

    Returns:
        Float in [0.0, 1.0]. Higher = more severe/weighted hedging language.
    """
    weighted_sum = 0.0
    for tier, count in scan_result["by_severity"].items():
        weighted_sum += count * SEVERITY_WEIGHTS[tier]

    word_count = scan_result["word_count"]
    if word_count == 0:
        return 0.0

    # Normalize by word count and cap at 1.0.
    raw_score = weighted_sum / word_count
    return min(raw_score, 1.0)


def summarize_top_patterns(
    scan_result: Dict[str, any], top_n: int = 5
) -> List[Tuple[str, str]]:
    """
    Extract the most common unique hedging phrases from scan results.

    Args:
        scan_result: Output dict from scan_filing_text.
        top_n: Number of top patterns to return.

    Returns:
        List of (severity_tier, unique_phrase) tuples, sorted by frequency.
    """
    phrase_counts: Dict[str, str] = {}
    for tier, phrase, _, _ in scan_result["matches"]:
        key = phrase.lower()
        if key not in phrase_counts:
            phrase_counts[key] = tier

    # Sort by frequency (number of occurrences) and return top N.
    sorted_phrases = sorted(
        phrase_counts.items(),
        key=lambda x: sum(1 for t, p, _, _ in scan_result["matches"] if p.lower() == x[0]),
        reverse=True,
    )

    return
