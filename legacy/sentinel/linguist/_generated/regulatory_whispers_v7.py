"""
Regulatory Whispers Detector — scans SEC filings for hedging language patterns.

This module identifies and scores the density of cautionary/hedging linguistic
markers in SEC filings (8-K, 10-Q, 10-K). High hedging density may signal
management uncertainty, regulatory pressure, or hidden risks — useful for
contrarian or risk-adjusted sentiment signals in the Sentinel pipeline.

Part of the Linguist pillar: feeds raw filing text → hedging scores → Judge.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RegulatoryWhisper:
    """Container for a single hedging pattern match and its context."""
    
    pattern: str
    category: str
    match_text: str
    position: int
    context_window: str


@dataclass
class HedgingAnalysis:
    """Aggregate hedging analysis for a single filing or document."""
    
    total_words: int
    hedging_matches: int
    hedging_density: float
    category_breakdown: Dict[str, int]
    top_patterns: List[Tuple[str, int]]
    risk_score: float
    whispers: List[RegulatoryWhisper]


# Core hedging language patterns organized by category.
HEDGING_PATTERNS = {
    "uncertainty_markers": [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\blikely\b",
        r"\bappears\b",
        r"\bseems\b",
        r"\bestimated\b",
        r"\bapproximately\b",
    ],
    "material_risk_language": [
        r"\bmaterially\b",
        r"\bmaterial\s+risk",
        r"\bmaterial\s+adverse",
        r"\bmaterial\s+effect",
        r"\bsubstantial\s+risk",
        r"\bsignificant\s+risk",
    ],
    "contingency_markers": [
        r"\bsubject\s+to\b",
        r"\bcontingent\s+on\b",
        r"\bcontingent\s+upon\b",
        r"\bdependent\s+on\b",
        r"\bif\s+and\s+when\b",
        r"\bto\s+the\s+extent\b",
        r"\bconditioned\s+upon\b",
    ],
    "limitation_qualifiers": [
        r"\blimited\b",
        r"\bunless\b",
        r"\bexcept\b",
        r"\bexcluding\b",
        r"\bexcept\s+for\b",
        r"\bother\s+than\b",
        r"\bnotwithstanding\b",
    ],
    "future_uncertainty": [
        r"\bno\s+assurance\b",
        r"\bno\s+guarantee\b",
        r"\bcannot\s+assure\b",
        r"\bunable\s+to\s+predict\b",
        r"\bunpredictable\b",
        r"\bunforeseen\b",
        r"\bforthcoming\b",
    ],
    "compliance_and_regulatory": [
        r"\bcomply\s+with\b",
        r"\bcompliance\s+with\b",
        r"\bregulatory\s+approval\b",
        r"\bgovernmental\s+approval\b",
        r"\bsubject\s+to\s+regulation\b",
        r"\bregulatory\s+changes\b",
        r"\blicensing\s+requirements\b",
    ],
}


def _normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace in text for pattern matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_context_window(
    text: str, match_pos: int, window_chars: int = 100
) -> str:
    """Extract surrounding context around a match position."""
    start = max(0, match_pos - window_chars)
    end = min(len(text), match_pos + window_chars)
    return text[start:end].replace("\n", " ").strip()


def _count_words(text: str) -> int:
    """Count approximate word count in text."""
    return len(text.split())


def detect_hedging_patterns(
    filing_text: str, context_window: int = 100
) -> HedgingAnalysis:
    """
    Scan SEC filing text for hedging language and return aggregate analysis.
    
    Args:
        filing_text: Raw text from SEC filing (8-K, 10-Q, 10-K).
        context_window: Characters to include before/after each match.
    
    Returns:
        HedgingAnalysis with density, breakdown, and individual whispers.
    """
    normalized = _normalize_text(filing_text)
    total_words = _count_words(normalized)
    
    category_breakdown: Dict[str, int] = {
        cat: 0 for cat in HEDGING_PATTERNS.keys()
    }
    all_whispers: List[RegulatoryWhisper] = []
    pattern_frequency: Dict[str, int] = {}
    
    for category, patterns in HEDGING_PATTERNS.items():
        for pattern in patterns:
            matches = list(re.finditer(pattern, normalized))
            category_breakdown[category] += len(matches)
            pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + len(matches)
            
            for match in matches:
                context = _extract_context_window(
                    normalized, match.start(), context_window
                )
                whisper = RegulatoryWhisper(
                    pattern=pattern,
                    category=category,
                    match_text=match.group(),
                    position=match.start(),
                    context_window=context,
                )
                all_whispers.append(whisper)
    
    total_hedging = sum(category_breakdown.values())
    hedging_density = (total_hedging / total_words * 100) if total_words > 0 else 0.0
    
    top_patterns = sorted(
        pattern_frequency.items(), key=lambda x: x[1], reverse=True
    )[:10]
    
    # Risk score: normalized hedging density with non-linear scaling.
    # Assumes 0-5% is baseline; >10% is elevated risk signal.
    raw_score = min(hedging_density / 10.0, 1.0)  # Cap at 1.0
    risk_score = (raw_score ** 1.5) * 100.0  # Non-linear boost for high density
    
    return HedgingAnalysis(
        total_words=total_words,
        hedging_matches=total_hedging,
        hedging_density=hedging_density,
        category_breakdown=category_breakdown,
        top_patterns=top_patterns,
        risk_score=risk_score,
        whispers=all_whispers,
    )


def score_regulatory_risk(analysis: HedgingAnalysis) -> Dict[str, float]:
    """
    Convert HedgingAnalysis into multi-dimensional risk scores.
    
    Returns dict with keys: density, uncertainty, contingency, compliance,
    and overall_risk (0–100 scale).
    """
    cb = analysis.category_breakdown
    total = analysis.hedging_matches
    
    if total == 0:
        return {
            "density_score": 0.0,
            "uncertainty_score": 0.0,
            "contingency_score": 0.0,
            "compliance_score": 0.0,
            "overall_risk": 0.0,
        }
    
    # Dimension-specific scoring
    uncertainty_norm = (cb.get("uncertainty_markers", 0) / total) * 100
    contingency_norm = (cb.get
