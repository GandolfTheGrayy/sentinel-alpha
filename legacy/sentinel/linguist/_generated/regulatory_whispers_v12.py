"""
Regulatory Whispers Detector — Sentinel Linguist Module

Scans SEC filings for hedging language patterns that signal management
caution or legal risk-shifting. Quantifies "regulatory whisper density"
by detecting phrases like 'may', 'subject to', 'could materially', 'if',
'contingent', etc. and scoring their prevalence relative to filing length.

This module feeds into the Linguist certainty scorer: high whisper density
correlates with lower confidence in forward guidance, and can signal
latent downside risk not yet reflected in sentiment or price.

Used by: sentinel/linguist/sample_score.py (as a sub-signal)
         sentinel/judge/predictor.py (for regulatory risk weighting)
"""

import re
from typing import Dict, List, Tuple
from collections import defaultdict


# ============================================================================
# Core Hedging Patterns
# ============================================================================

HEDGING_PATTERNS = {
    # Explicit uncertainty
    "may": r"\bmay\b",
    "might": r"\bmight\b",
    "could": r"\bcould\b",
    "can": r"\bcan\b",
    "appears": r"\bappears\b",
    "seems": r"\bseems\b",
    "suggests": r"\bsuggests\b",
    
    # Contingency & conditionality
    "if": r"\bif\b",
    "subject_to": r"\bsubject\s+to\b",
    "contingent": r"\bcontingent\b",
    "depending": r"\bdepending\b",
    "conditional": r"\bconditional\b",
    "unless": r"\bunless\b",
    
    # Materiality & magnitude hedging
    "materially": r"\bmaterially\b",
    "significantly": r"\bsignificantly\b",
    "substantially": r"\bsubstantially\b",
    "adverse": r"\badverse\b",
    "negative": r"\bnegative\b",
    "challenge": r"\bchallenge\b",
    "risk": r"\brisk\b",
    "uncertainty": r"\buncertainty\b",
    
    # Limitation & scope hedging
    "approximately": r"\bapproximately\b",
    "estimate": r"\bestimate\b",
    "estimated": r"\bestimated\b",
    "expect": r"\bexpect\b",
    "anticipated": r"\banticipated\b",
    "projected": r"\bprojected\b",
    
    # Legal disclaimers & disclaimers
    "forward_looking": r"\bforward.{0,3}looking\b",
    "although": r"\balthough\b",
    "however": r"\bhowever\b",
    "but": r"\bbut\b",
    "except": r"\bexcept\b",
    "notwithstanding": r"\bnotwithstanding\b",
}


# ============================================================================
# INTENSE Hedging Phrases (multi-word, high risk signal)
# ============================================================================

INTENSE_PHRASES = [
    r"materially\s+adverse",
    r"could\s+materially",
    r"may\s+not",
    r"unable\s+to",
    r"fail(?:ure)?",
    r"liquidity\s+(?:constraints|risks)",
    r"going\s+concern",
    r"impairment",
    r"write.?down",
    r"restructur",
    r"covenant\s+(?:breach|violation)",
    r"default",
    r"bankruptcy",
    r"insolvency",
]


# ============================================================================
# Main Detection & Scoring Functions
# ============================================================================

def detect_hedging_tokens(text: str) -> Dict[str, int]:
    """
    Count occurrences of hedging tokens in text (case-insensitive).
    
    Returns a dict mapping token names to occurrence counts.
    """
    text_lower = text.lower()
    counts = {}
    
    for token_name, pattern in HEDGING_PATTERNS.items():
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        counts[token_name] = len(matches)
    
    return counts


def detect_intense_phrases(text: str) -> Dict[str, int]:
    """
    Count occurrences of high-risk hedging phrases (multi-word patterns).
    
    Returns a dict mapping phrase names to occurrence counts.
    """
    text_lower = text.lower()
    phrase_counts = {}
    
    for i, phrase_pattern in enumerate(INTENSE_PHRASES):
        matches = re.findall(phrase_pattern, text_lower, re.IGNORECASE)
        phrase_counts[f"intense_{i}"] = len(matches)
    
    return phrase_counts


def compute_whisper_density(text: str) -> float:
    """
    Compute regulatory whisper density: ratio of hedging tokens to word count.
    
    Density = (total_hedging_tokens) / (word_count) * 100
    Returns float in range [0, 100] representing percentage.
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    hedging_counts = detect_hedging_tokens(text)
    total_hedges = sum(hedging_counts.values())
    
    # Count words
    word_count = len(re.findall(r"\b\w+\b", text))
    
    if word_count == 0:
        return 0.0
    
    density = (total_hedges / word_count) * 100.0
    return round(density, 2)


def score_regulatory_whispers(text: str) -> Dict:
    """
    Comprehensive regulatory whispers analysis on SEC filing text.
    
    Returns dict with:
      - whisper_density: float [0–100]
      - hedging_tokens: dict of token counts
      - intense_phrases: dict of high-risk phrase counts
      - overall_score: float [0–100] (higher = more hedging/caution)
      - risk_level: str ("low", "moderate", "high", "critical")
    """
    if not text:
        return {
            "whisper_density": 0.0,
            "hedging_tokens": {},
            "intense_phrases": {},
            "overall_score": 0.0,
            "risk_level": "low",
        }
    
    hedging_tokens = detect_hedging_tokens(text)
    intense_phrases = detect_intense_phrases(text)
    
    density = compute_whisper_density(text)
    
    # Compute overall score: base density + intense phrase penalty
    intense_count = sum(intense_phrases.values())
    intense_penalty = intense_count * 2.0  # Each intense phrase adds weight
    
    overall_score = min(100.0, density + intense_penalty)
    
    # Classify risk level based on overall score
    if overall_score >= 70:
        risk_level = "critical"
    elif overall_score >= 50:
        risk_level = "high"
    elif overall_score >= 25:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    return {
        "whisper_density": density,
        "hedging_tokens": hedging_tokens,
        "intense_phrases": intense_phrases,
        "overall_score": round(overall_score, 2),
        "risk_level": risk_level,
    }


def extract_sentences_with_hedges(text: str, context_sentences: int = 1) -> List[str]:
    """
    Extract sentences containing hedging language, with context.
    
    Args:
        text: Filing text to analyze.
        context_sentences: Number of surrounding sentences to include.
    
    Returns list of strings (extracted sentence blocks with context).
    """
    # Split text into sentences (naive split on periods/newlines)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
