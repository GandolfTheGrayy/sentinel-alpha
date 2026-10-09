"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns ('may', 'subject to', 'could materially',
'if', 'risk', 'uncertain', etc.) and scores their density as a proxy for management
caution or regulatory pressure. Higher scores indicate more defensive/cautious tone.

Used by Judge to weight predictions: high regulatory whisper density may indicate
latent risk that market has not yet priced in.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RegulatoryWhispersScore:
    """Result of regulatory whispers analysis on a filing."""
    total_words: int
    hedging_count: int
    risk_count: int
    uncertainty_count: int
    density_score: float  # 0–100, higher = more hedging/caution
    top_patterns: List[Tuple[str, int]]  # [(pattern, count), ...] sorted by freq
    excerpt: str  # sample sentence containing hedging language


# Curated hedging language patterns from SEC filing linguistics research.
HEDGING_PATTERNS = {
    "may": r"\bmay\b",
    "might": r"\bmight\b",
    "could": r"\bcould\b",
    "can": r"\bcan\b",
    "appears": r"\bappears\b",
    "suggests": r"\bsuggests\b",
    "tends": r"\btends\b",
    "likely": r"\blikely\b",
    "possibly": r"\bpossibly\b",
    "arguably": r"\bargubly\b",
    "subject to": r"\bsubject\s+to\b",
    "if": r"\bif\b",
    "unless": r"\bunless\b",
    "except": r"\bexcept\b",
    "provided": r"\bprovided\b",
}

RISK_PATTERNS = {
    "risk": r"\brisk\b",
    "risks": r"\brisks\b",
    "uncertain": r"\buncertain\b",
    "uncertainty": r"\buncertainty\b",
    "volatile": r"\bvolatile\b",
    "volatility": r"\bvolatility\b",
    "adverse": r"\badverse\b",
    "adverse effect": r"\badverse\s+effect\b",
    "material adverse": r"\bmaterial\s+adverse\b",
}

MATERIAL_IMPACT_PATTERNS = {
    "materially": r"\bmaterially\b",
    "material": r"\bmaterial\b",
    "material impact": r"\bmaterial\s+impact\b",
    "could materially": r"\bcould\s+materially\b",
    "may materially": r"\bmay\s+materially\b",
    "significant": r"\bsignificant\b",
    "substantial": r"\bsubstantial\b",
}


def tokenize_words(text: str) -> List[str]:
    """Split text into words, lowercased; strip punctuation."""
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    return words


def count_pattern_occurrences(text: str, pattern: str) -> int:
    """Count non-overlapping regex matches (case-insensitive)."""
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def extract_sample_excerpt(text: str, pattern_dict: Dict[str, str]) -> str:
    """Extract first sentence containing any pattern from pattern_dict."""
    sentences = re.split(r"[.!?]", text)
    for sentence in sentences:
        for pattern in pattern_dict.values():
            if re.search(pattern, sentence, re.IGNORECASE):
                clean = sentence.strip()
                if len(clean) > 20:
                    return clean[:150] + ("..." if len(clean) > 150 else "")
    return ""


def analyze_regulatory_whispers(filing_text: str) -> RegulatoryWhispersScore:
    """
    Scan SEC filing text for hedging & risk language; return density score + breakdown.
    
    Counts occurrences of hedging words (may, could, subject to, etc.),
    risk language (risk, uncertain, volatile), and material-impact modifiers.
    Density = (total hedging+risk+material counts / word count) * 100.
    
    Args:
        filing_text: Full text of SEC filing (8-K, 10-Q, 10-K, etc.)
    
    Returns:
        RegulatoryWhispersScore with counts, density (0–100), and sample excerpt.
    """
    words = tokenize_words(filing_text)
    total_words = len(words)
    
    if total_words == 0:
        return RegulatoryWhispersScore(
            total_words=0,
            hedging_count=0,
            risk_count=0,
            uncertainty_count=0,
            density_score=0.0,
            top_patterns=[],
            excerpt="",
        )
    
    # Count each category.
    hedging_count = sum(
        count_pattern_occurrences(filing_text, pattern)
        for pattern in HEDGING_PATTERNS.values()
    )
    risk_count = sum(
        count_pattern_occurrences(filing_text, pattern)
        for pattern in RISK_PATTERNS.values()
    )
    material_count = sum(
        count_pattern_occurrences(filing_text, pattern)
        for pattern in MATERIAL_IMPACT_PATTERNS.values()
    )
    
    total_regulatory_signals = hedging_count + risk_count + material_count
    density_score = (total_regulatory_signals / total_words) * 100 if total_words > 0 else 0.0
    
    # Cap at 100 for readability (pathological filings with extreme repetition).
    density_score = min(density_score, 100.0)
    
    # Build top patterns list: combine all categories, count, and sort.
    all_patterns = {**HEDGING_PATTERNS, **RISK_PATTERNS, **MATERIAL_IMPACT_PATTERNS}
    pattern_counts = [
        (name, count_pattern_occurrences(filing_text, pattern))
        for name, pattern in all_patterns.items()
    ]
    pattern_counts.sort(key=lambda x: x[1], reverse=True)
    top_patterns = [(name, count) for name, count in pattern_counts if count > 0][:10]
    
    # Extract sample excerpt.
    combined_patterns = {**HEDGING_PATTERNS, **RISK_PATTERNS, **MATERIAL_IMPACT_PATTERNS}
    excerpt = extract_sample_excerpt(filing_text, combined_patterns)
    
    return RegulatoryWhispersScore(
        total_words=total_words,
        hedging_count=hedging_count,
        risk_count=risk_count,
        uncertainty_count=material_count,
        density_score=density_score,
        top_patterns=top_patterns,
        excerpt=excerpt,
    )


def regulatory_whispers_signal(score: RegulatoryWhispersScore) -> Dict[str, float]:
    """
    Convert RegulatoryWhispersScore to a dict of Judge-consumable signals.
    
    Maps density score to a bearish bias factor (0–1 scale):
    - 0–1% density: neutral (0.0)
    - 1–3% density: mild caution (0.3)
    - 3–5% density: moderate caution (0.6)
    - 5%+ density: high caution (0.9)
    
    Returns:
        Dict with keys: 'whisper_density', 'bearish_bias', 'confidence'.
    """
    density = score.density_score
    
    if density < 1.0:
        bearish_bias = 0.0
    elif density < 3.0:
        bearish_bias = 0.
