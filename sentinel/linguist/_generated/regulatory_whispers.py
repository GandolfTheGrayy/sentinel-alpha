"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns that signal management caution,
uncertainty, or legal risk mitigation. Computes a "whisper density" score
(0–1) reflecting the concentration of hedging terms like 'may', 'subject to',
'could materially', 'risk', 'contingent', etc. across the filing text.

Used by Judge to modulate confidence in predictions when legal exposure is high.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class WhisperAnalysis:
    """Container for regulatory whisper detection results."""
    ticker: str
    filing_type: str
    total_words: int
    hedging_terms_found: int
    whisper_density: float
    term_breakdown: Dict[str, int]
    sample_sentences: List[str]


# Hedging language patterns organized by severity tier.
HEDGING_LEXICON = {
    "critical": [
        r"\bmay\b",
        r"\bcould\b",
        r"\bmight\b",
        r"\bsubject to\b",
        r"\bif\b",
        r"\bcontingent\b",
        r"\bdependent on\b",
    ],
    "high": [
        r"\buncertainty\b",
        r"\buncertain\b",
        r"\brisk\b",
        r"\brisk[s]?\b",
        r"\bcould materially\b",
        r"\bmay materially\b",
        r"\badverse\b",
        r"\badversity\b",
    ],
    "medium": [
        r"\blikely\b",
        r"\bunlikely\b",
        r"\bpotential\b",
        r"\bpossible\b",
        r"\bchallenges\b",
        r"\nobstacle[s]?\b",
        r"\bmitigat\w+\b",
        r"\blimit\w+\b",
    ],
    "low": [
        r"\bnote\b",
        r"\bconsider\b",
        r"\bevaluate\b",
        r"\bexamine\b",
        r"\bassess\b",
    ],
}


def _tokenize_text(text: str) -> List[str]:
    """Tokenize filing text into words, lowercased."""
    return re.findall(r"\b\w+\b", text.lower())


def _extract_sentences_with_hedging(text: str, max_samples: int = 5) -> List[str]:
    """Extract up to max_samples sentences containing hedging language."""
    sentences = re.split(r"[.!?]+", text)
    hedging_sentences = []
    
    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean or len(sentence_clean) < 20:
            continue
        
        # Check if sentence contains any hedging term.
        sentence_lower = sentence_clean.lower()
        for tier_terms in HEDGING_LEXICON.values():
            for pattern in tier_terms:
                if re.search(pattern, sentence_lower):
                    hedging_sentences.append(sentence_clean[:150])
                    break
            if len(hedging_sentences) >= max_samples:
                break
        
        if len(hedging_sentences) >= max_samples:
            break
    
    return hedging_sentences


def analyze_regulatory_whispers(
    ticker: str,
    filing_type: str,
    filing_text: str,
) -> WhisperAnalysis:
    """
    Scan SEC filing text for hedging language; return whisper density (0–1).
    
    Args:
        ticker: Stock ticker symbol.
        filing_type: Type of filing (e.g. '8-K', '10-Q', '10-K').
        filing_text: Full text of the SEC filing.
    
    Returns:
        WhisperAnalysis with density score, term breakdown, and sample sentences.
    """
    text_lower = filing_text.lower()
    words = _tokenize_text(filing_text)
    total_words = len(words) if words else 1
    
    term_breakdown: Dict[str, int] = {}
    total_hedging_hits = 0
    
    # Count hedging term occurrences across all tiers.
    for tier, patterns in HEDGING_LEXICON.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                term_breakdown[pattern] = matches
                total_hedging_hits += matches
    
    # Whisper density: ratio of hedging hits to total words.
    whisper_density = min(1.0, total_hedging_hits / total_words) if total_words > 0 else 0.0
    
    sample_sentences = _extract_sentences_with_hedging(filing_text, max_samples=5)
    
    return WhisperAnalysis(
        ticker=ticker,
        filing_type=filing_type,
        total_words=total_words,
        hedging_terms_found=total_hedging_hits,
        whisper_density=whisper_density,
        term_breakdown=term_breakdown,
        sample_sentences=sample_sentences,
    )


def score_whisper_confidence_impact(whisper_density: float) -> float:
    """
    Map whisper density (0–1) to confidence reduction factor (0–1).
    
    Higher whisper density reduces confidence; returns multiplier to apply
    to base prediction confidence. E.g., 0.8 means reduce confidence by 20%.
    """
    if whisper_density < 0.01:
        return 1.0  # No impact.
    elif whisper_density < 0.05:
        return 0.95
    elif whisper_density < 0.10:
        return 0.85
    elif whisper_density < 0.20:
        return 0.70
    else:
        return 0.50  # Heavy hedging → low confidence.


def format_whisper_report(analysis: WhisperAnalysis) -> str:
    """
    Format WhisperAnalysis into a human-readable report string.
    """
    lines = [
        f"Regulatory Whispers Report — {analysis.ticker} ({analysis.filing_type})",
        f"  Total words: {analysis.total_words}",
        f"  Hedging terms found: {analysis.hedging_terms_found}",
        f"  Whisper density: {analysis.whisper_density:.4f}",
        f"  Confidence impact: ×{score_whisper_confidence_impact(analysis.whisper_density):.2f}",
        "",
        "Top hedging terms:",
    ]
    
    sorted_terms = sorted(
        analysis.term_breakdown.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    
    for term, count in sorted_terms:
        lines.append(f"  {term}: {count}")
    
    if analysis.sample_sentences:
        lines.extend(["", "Sample sentences with hedging:"])
        for i, sent in enumerate(analysis.sample_sentences, 1):
            lines.append(f"  {i}. {sent}…")
    
    return "\n".join(lines)
