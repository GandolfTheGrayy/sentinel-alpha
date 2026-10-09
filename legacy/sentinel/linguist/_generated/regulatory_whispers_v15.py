"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns ('may', 'subject to', 'could materially',
'risk', 'uncertain', etc.) and computes a regulatory caution score (0.0–1.0) indicating
the density and severity of forward-looking uncertainty language. Higher scores suggest
management is emphasizing risk or hedging their claims—a potential bearish signal when
combined with positive earnings guidance.

Integrates with Linguist reasoning pipeline to contextualize filing tone shifts.
"""

import re
from typing import Dict, List, Tuple


# Hedging phrase patterns organized by severity tier.
HEDGING_PATTERNS = {
    "high_severity": [
        r"\bmay\s+(not\s+)?(?:be|result|cause|lead)",
        r"\bsubject\s+to\b",
        r"\bcould\s+materially",
        r"\bmaterial\s+(?:adverse|risk)",
        r"\bsignificant\s+(?:risk|uncertainty)",
        r"\bfail(?:ure|ed|ing)?\s+to",
        r"\bloss(?:es)?\s+of",
        r"\bif\s+(?:we|the|our)\s+(?:fail|cannot)",
    ],
    "medium_severity": [
        r"\bmay\b",
        r"\bcould\b",
        r"\bmight\b",
        r"\bsubject\b",
        r"\buncertain(?:ty)?",
        r"\brisk(?:s)?(?:\s+of)?",
        r"\bif\s+(?:and\s+)?when\b",
        r"\bpotential(?:ly)?\b",
    ],
    "low_severity": [
        r"\bexpect(?:ed|s|ing)?\b",
        r"\bestimate(?:d|s|ing)?\b",
        r"\bapproximate(?:ly)?\b",
        r"\bapprox\.\b",
        r"\b~\s*\d",
    ],
}


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences, normalized."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def score_hedging_density(text: str) -> Dict[str, float]:
    """
    Analyze hedging language density in text; return severity-weighted score.

    Returns dict with keys: 'high_severity_count', 'medium_severity_count',
    'low_severity_count', 'total_hedges', 'weighted_score' (0.0–1.0).
    """
    text_lower = text.lower()
    sentences = tokenize_sentences(text_lower)
    
    high_count = 0
    medium_count = 0
    low_count = 0
    
    for pattern in HEDGING_PATTERNS["high_severity"]:
        high_count += len(re.findall(pattern, text_lower))
    
    for pattern in HEDGING_PATTERNS["medium_severity"]:
        medium_count += len(re.findall(pattern, text_lower))
    
    for pattern in HEDGING_PATTERNS["low_severity"]:
        low_count += len(re.findall(pattern, text_lower))
    
    total_hedges = high_count + medium_count + low_count
    sentence_count = max(len(sentences), 1)
    
    # Weighted score: high=3x, medium=1x, low=0.3x per occurrence.
    weighted_sum = (high_count * 3.0) + (medium_count * 1.0) + (low_count * 0.3)
    max_possible = sentence_count * 3.0  # Assume ~3 hedges per sentence as ceiling.
    weighted_score = min(weighted_sum / max_possible, 1.0)
    
    return {
        "high_severity_count": high_count,
        "medium_severity_count": medium_count,
        "low_severity_count": low_count,
        "total_hedges": total_hedges,
        "sentence_count": sentence_count,
        "weighted_score": weighted_score,
    }


def detect_risk_section_intensity(text: str) -> Dict[str, float]:
    """
    Extract and score the Risk Factors or MD&A sections separately.

    Returns dict with 'risk_section_score', 'mda_section_score', 'overall_intensity'.
    """
    text_lower = text.lower()
    
    # Extract Risk Factors section (heuristic).
    risk_match = re.search(
        r"(?:risk\s+factors?|risks?).*?(?=(?:item|part|exhibits|signatures|$))",
        text_lower,
        re.IGNORECASE | re.DOTALL,
    )
    risk_section = risk_match.group(0) if risk_match else ""
    
    # Extract MD&A section (heuristic).
    mda_match = re.search(
        r"(?:management.{0,20}discussion|md\s*&\s*a).*?(?=(?:item|part|exhibits|signatures|$))",
        text_lower,
        re.IGNORECASE | re.DOTALL,
    )
    mda_section = mda_match.group(0) if mda_match else ""
    
    risk_score = score_hedging_density(risk_section)["weighted_score"]
    mda_score = score_hedging_density(mda_section)["weighted_score"]
    
    overall_intensity = (risk_score * 0.6) + (mda_score * 0.4)
    
    return {
        "risk_section_score": risk_score,
        "mda_section_score": mda_score,
        "overall_intensity": overall_intensity,
    }


def compute_regulatory_whispers_score(filing_text: str) -> Dict[str, float]:
    """
    Compute composite Regulatory Whispers score from a full SEC filing.

    Combines hedging density, section-specific intensity, and length normalization.
    Returns dict with 'regulatory_whispers_score' (0.0–1.0) and breakdown metrics.
    """
    if not filing_text or len(filing_text.strip()) == 0:
        return {
            "regulatory_whispers_score": 0.0,
            "hedging_density": 0.0,
            "section_intensity": 0.0,
            "text_length": 0,
        }
    
    # Overall hedging density.
    hedging_result = score_hedging_density(filing_text)
    hedging_score = hedging_result["weighted_score"]
    
    # Section-specific analysis.
    section_result = detect_risk_section_intensity(filing_text)
    section_score = section_result["overall_intensity"]
    
    # Composite: weight hedging density and section intensity.
    composite = (hedging_score * 0.5) + (section_score * 0.5)
    
    return {
        "regulatory_whispers_score": min(composite, 1.0),
        "hedging_density": hedging_score,
        "section_intensity": section_score,
        "text_length": len(filing_text),
        "high_severity_hedges": hedging_result["high_severity_count"],
        "medium_severity_hedges": hedging_result["medium_severity_count"],
        "low_severity_hedges": hedging_result["low_severity_count"],
    }


def compare_filing_tone_shift(
    prior_filing_text: str, current_filing_text: str
) -> Dict[str, float]:
    """
    Detect tone shift (Linguistic Drift) between two filings.

    Returns dict with 'prior_score', 'current_score', 'tone_shift' (delta),
    'shift_direction' ('more_cautious', 'more_optimistic', 'stable').
