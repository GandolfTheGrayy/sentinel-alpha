"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns ('may', 'subject to', 'could materially',
'risk', 'uncertain', etc.) and computes density scores. High hedging density may signal
management caution or forthcoming headwinds; low density may indicate confidence.

Used by Judge to weight predictions and by Linguist to flag tone shifts.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class HedgingAnalysis:
    """Result of regulatory whispers analysis on a filing."""
    total_words: int
    hedging_phrase_count: int
    density_score: float
    top_phrases: List[Tuple[str, int]]
    flagged_sections: List[Dict[str, str]]


# Comprehensive hedging language patterns used in SEC filings
HEDGING_PATTERNS = {
    "uncertainty": [
        r"\bmay\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bif\b",
        r"\bunable to\b",
        r"\bsubject to\b",
        r"\bdepends on\b",
    ],
    "risk_language": [
        r"\brisk\b",
        r"\brisks\b",
        r"\brisking\b",
        r"\brisk of\b",
        r"\brisks of\b",
        r"\badverse\b",
        r"\bnegative\b",
        r"\bthreaten",
        r"\bthreat\b",
    ],
    "material_impact": [
        r"\bmaterially\b",
        r"\bmaterial\b",
        r"\bmaterial adverse\b",
        r"\bsignificant\b",
        r"\bsubstantial\b",
        r"\bcritical\b",
        r"\bsevere\b",
    ],
    "uncertain_outcomes": [
        r"\buncertain\b",
        r"\buncertainty\b",
        r"\bunpredictable\b",
        r"\bunforeseen\b",
        r"\bvariable\b",
        r"\bfluctuate",
        r"\bvolatile\b",
        r"\bvolatility\b",
    ],
    "qualifications": [
        r"\bapproximately\b",
        r"\bestimated\b",
        r"\bexpect\b",
        r"\bexpected\b",
        r"\bbelieve\b",
        r"\bsubjective\b",
        r"\blikely\b",
        r"\bunlikely\b",
    ],
    "contingency": [
        r"\bunless\b",
        r"\bexcept\b",
        r"\bprovided that\b",
        r"\bwithout\b",
        r"\bcontingent\b",
        r"\bcontingency\b",
        r"\bconditional\b",
    ],
}


def detect_regulatory_whispers(filing_text: str) -> HedgingAnalysis:
    """
    Scan SEC filing text and compute hedging language density.
    
    Returns HedgingAnalysis with density score (0.0–1.0), phrase counts, and flagged sections.
    """
    if not filing_text or not isinstance(filing_text, str):
        return HedgingAnalysis(
            total_words=0,
            hedging_phrase_count=0,
            density_score=0.0,
            top_phrases=[],
            flagged_sections=[],
        )

    # Normalize text: lowercase, remove excessive whitespace
    text_normalized = re.sub(r"\s+", " ", filing_text.lower().strip())
    words = text_normalized.split()
    total_words = len(words)

    if total_words == 0:
        return HedgingAnalysis(
            total_words=0,
            hedging_phrase_count=0,
            density_score=0.0,
            top_phrases=[],
            flagged_sections=[],
        )

    # Count hedging phrases across all categories
    phrase_counts: Dict[str, int] = {}
    total_hedging_count = 0

    for category, patterns in HEDGING_PATTERNS.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, text_normalized, re.IGNORECASE))
            if matches > 0:
                phrase_counts[pattern] = matches
                total_hedging_count += matches

    # Compute density: hedging phrases per 1000 words
    density_score = (total_hedging_count / total_words * 1000) if total_words > 0 else 0.0
    
    # Cap at 1.0 for interpretability
    normalized_density = min(density_score / 100.0, 1.0)

    # Top phrases by frequency
    top_phrases = sorted(phrase_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    # Extract flagged sections: sentences containing multiple hedging terms
    sentences = re.split(r"[.!?]+", text_normalized)
    flagged_sections = []

    for sent in sentences:
        sent_strip = sent.strip()
        if not sent_strip or len(sent_strip.split()) < 3:
            continue

        hedging_in_sent = 0
        matched_phrases = []

        for pattern in [p for patterns in HEDGING_PATTERNS.values() for p in patterns]:
            if re.search(pattern, sent_strip, re.IGNORECASE):
                hedging_in_sent += 1
                matched_phrases.append(pattern)

        # Flag sentences with 3+ distinct hedging patterns
        if hedging_in_sent >= 3:
            flagged_sections.append(
                {
                    "sentence": sent_strip[:150],  # Truncate for readability
                    "hedging_count": hedging_in_sent,
                    "patterns": matched_phrases[:5],
                }
            )

    return HedgingAnalysis(
        total_words=total_words,
        hedging_phrase_count=total_hedging_count,
        density_score=normalized_density,
        top_phrases=top_phrases,
        flagged_sections=flagged_sections[:20],  # Limit output
    )


def score_hedging_intensity(analysis: HedgingAnalysis) -> Dict[str, float]:
    """
    Map hedging analysis to interpretable intensity scores (0.0–1.0 scale).
    
    Returns dict with 'caution_level', 'risk_emphasis', 'overall_tone_shift'.
    """
    density = analysis.density_score
    
    # Calibrate based on observed filing patterns
    caution_level = min(density * 1.2, 1.0)
    risk_emphasis = min(
        sum(count for phrase, count in analysis.top_phrases if "risk" in phrase) / max(1, analysis.hedging_phrase_count),
        1.0
    )
    overall_tone_shift = density  # Direct proxy for tone shift

    return {
        "caution_level": caution_level,
        "risk_emphasis": risk_emphasis,
        "overall_tone_shift": overall_tone_shift,
    }


def compare_hedging_drift(
    previous_analysis: HedgingAnalysis,
    current_analysis: HedgingAnalysis,
) -> Dict[str, float]:
    """
    Detect linguistic drift: compare hedging density & phrase composition over time.
    
    Returns dict with 'density_delta', 'direction_flag', 'significance'.
    """
    prev_density = previous_analysis.density_score
    curr_density = current_analysis.density_score
    density_delta = curr_density - prev_density

    # Direction flag: positive = more caution, negative = more confidence
    direction_flag = 1.0 if density_delta >
