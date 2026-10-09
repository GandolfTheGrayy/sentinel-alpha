"""
Regulatory Whispers detector for Sentinel Sentiment Engine.

Scans SEC filings for hedging language patterns ('may', 'subject to', 'could materially',
'risk', 'uncertain', etc.) and computes a regulatory caution score. High density of
hedging language may signal management uncertainty or risk disclosure emphasis.

Integrates with Linguist pillar to enrich sentiment analysis with regulatory tone signals.
Used by Judge for prediction confidence calibration.
"""

import re
from typing import Dict, List, Tuple


HEDGING_PATTERNS = {
    "may": r"\bmay\b",
    "might": r"\bmight\b",
    "could": r"\bcould\b",
    "subject_to": r"\bsubject to\b",
    "could_materially": r"\bcould\s+materially\b",
    "materially_adverse": r"\bmaterially\s+adverse\b",
    "uncertain": r"\buncertain",
    "risk": r"\brisk(?:s)?\b",
    "risks": r"\brisks\b",
    "contingent": r"\bcontingent\b",
    "if_and_when": r"\bif\s+and\s+when\b",
    "depends_on": r"\bdepends\s+on\b",
    "to_the_extent": r"\bto\s+the\s+extent\b",
    "unless": r"\bunless\b",
    "cannot_assure": r"\bcannot\s+assure\b",
    "no_assurance": r"\bno\s+assurance\b",
    "forward_looking": r"\bforward.looking\b",
    "assumption": r"\bassumption(?:s)?\b",
    "estimate": r"\bestimate(?:s|d)?\b",
}


def extract_hedging_signals(text: str) -> Dict[str, int]:
    """Extract counts of hedging language patterns from SEC filing text."""
    if not text:
        return {k: 0 for k in HEDGING_PATTERNS.keys()}
    
    text_lower = text.lower()
    signals = {}
    
    for signal_name, pattern in HEDGING_PATTERNS.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        signals[signal_name] = len(matches)
    
    return signals


def compute_regulatory_caution_score(text: str, word_count: int = None) -> Tuple[float, Dict[str, int]]:
    """
    Compute normalized regulatory caution score (0.0–1.0) from SEC filing text.
    
    Returns tuple of (caution_score, signal_breakdown).
    Score is normalized by filing length to account for document size.
    """
    if not text or not text.strip():
        return 0.0, {k: 0 for k in HEDGING_PATTERNS.keys()}
    
    if word_count is None:
        word_count = len(text.split())
    
    if word_count == 0:
        return 0.0, {k: 0 for k in HEDGING_PATTERNS.keys()}
    
    signals = extract_hedging_signals(text)
    total_hedges = sum(signals.values())
    
    # Normalize by word count (hedges per 1000 words)
    hedges_per_1k = (total_hedges / word_count) * 1000 if word_count > 0 else 0.0
    
    # Clamp to 0.0–1.0 assuming typical filing has 20–50 hedges per 1k words
    # at maximum caution; cap at 1.0 for extreme cases
    caution_score = min(1.0, hedges_per_1k / 50.0)
    
    return caution_score, signals


def score_regulatory_section(text: str, section_name: str = "Unknown") -> Dict[str, any]:
    """
    Score a single SEC filing section (e.g. MD&A, Risk Factors).
    
    Returns dict with caution_score, signals, section_name, and word_count.
    """
    word_count = len(text.split()) if text else 0
    caution_score, signals = compute_regulatory_caution_score(text, word_count)
    
    return {
        "section": section_name,
        "caution_score": caution_score,
        "signals": signals,
        "word_count": word_count,
        "total_hedges": sum(signals.values()),
    }


def aggregate_filing_caution(sections: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Aggregate caution scores across multiple sections of a single filing.
    
    Returns dict with weighted_caution_score, section_breakdown, and top_signal.
    Risk Factors and MD&A sections weighted more heavily (2x).
    """
    if not sections:
        return {
            "weighted_caution_score": 0.0,
            "section_count": 0,
            "total_hedges": 0,
            "sections": [],
            "top_signal": None,
        }
    
    heavy_weight_sections = {"Risk Factors", "MD&A", "Management's Discussion and Analysis"}
    weighted_sum = 0.0
    weight_total = 0.0
    all_hedges = {}
    
    for section in sections:
        weight = 2.0 if section.get("section") in heavy_weight_sections else 1.0
        weighted_sum += section["caution_score"] * weight
        weight_total += weight
        
        for signal, count in section.get("signals", {}).items():
            all_hedges[signal] = all_hedges.get(signal, 0) + count
    
    weighted_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    
    top_signal = max(all_hedges.items(), key=lambda x: x[1]) if all_hedges else (None, 0)
    
    return {
        "weighted_caution_score": min(1.0, weighted_score),
        "section_count": len(sections),
        "total_hedges": sum(all_hedges.values()),
        "sections": sections,
        "top_signal": top_signal[0],
        "top_signal_count": top_signal[1],
    }


def detect_tone_shift(current_filing: Dict[str, any], prior_filing: Dict[str, any]) -> Dict[str, float]:
    """
    Detect regulatory tone shift (delta in caution score) between two filings.
    
    Returns dict with shift (current - prior), direction ('increasing'/'decreasing'/'stable'),
    and magnitude.
    """
    current_score = current_filing.get("weighted_caution_score", 0.0)
    prior_score = prior_filing.get("weighted_caution_score", 0.0)
    
    shift = current_score - prior_score
    magnitude = abs(shift)
    
    if magnitude < 0.05:
        direction = "stable"
    elif shift > 0:
        direction = "increasing_caution"
    else:
        direction = "decreasing_caution"
    
    return {
        "shift": shift,
        "direction": direction,
        "magnitude": magnitude,
        "prior_score": prior_score,
        "current_score": current_score,
    }
