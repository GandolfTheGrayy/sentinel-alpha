"""
Regulatory Whispers Detector — identifies hedging language patterns in SEC filings.

This module scans SEC filing text for cautionary and hedging phrases that
signal management uncertainty or risk acknowledgment. It computes a "whispers
density" score (0–1) reflecting the concentration of such language, used by
the Linguist to adjust confidence in market predictions.

Hedging patterns include: 'may', 'could', 'subject to', 'materially', 'risk',
'uncertain', 'contingent', 'potential', 'estimated', 'approximately', and
regulatory boilerplate indicators. Higher density → lower prediction confidence.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RegulatoryWhispersResult:
    """Output of regulatory whispers analysis."""
    filing_text: str
    total_words: int
    hedging_matches: List[Tuple[str, int]]
    density_score: float
    top_patterns: List[Tuple[str, int]]


# Curated hedging and cautionary language patterns from SEC filings.
HEDGING_PATTERNS = [
    # Core uncertainty markers
    r'\bmay\b',
    r'\bcould\b',
    r'\bmight\b',
    r'\bwould\b',
    r'\bsubject\s+to\b',
    r'\bat\s+risk\b',
    r'\brisk(?:s)?\b',
    r'\buncertain(?:ty|ties)?\b',
    r'\bcontingent\b',
    r'\bpotential(?:ly)?\b',
    r'\bestimat(?:ed|es|ing)?\b',
    r'\bapproximate(?:ly)?\b',
    r'\bsubstantial\b',
    r'\bmaterial(?:ly)?\b',
    r'\badverse\b',
    r'\bexpectation(?:s)?\b',
    r'\bbelieve(?:s)?\b',
    r'\bintent(?:ion|s)?\b',
    r'\bplan(?:ned|s)?\b',
    r'\bproject(?:ed|s)?\b',
    r'\bforecast(?:ed|s)?\b',
    r'\beliminate(?:d)?\b',
    r'\breduc(?:ed|tion|es)?\b',
    r'\bimpact(?:s|ed)?\b',
    r'\bvariability\b',
    r'\bunpredictable\b',
    r'\bcompliance\b',
    r'\bregulatory\b',
    r'\blegal\b',
    r'\bobligat(?:ed|ion|ions)?\b',
    r'\binherent\b',
    r'\bexposure\b',
    r'\bliabilit(?:y|ies)?\b',
    r'\bwarranty\b',
    r'\bunless\b',
    r'\bexcept\b',
    r'\bdepend(?:s|ent|ency)?\b',
    r'\bsubject\b',
]

# Phrases indicating forward-looking statements (often precede hedging).
FORWARD_LOOKING_MARKERS = [
    r'forward-looking\s+statements',
    r'safe\s+harbor',
    r'certain\s+statements',
    r'management\s+believes',
    r'we\s+expect',
    r'we\s+anticipate',
]

# Regulatory boilerplate keywords indicating risk sections.
RISK_SECTION_MARKERS = [
    r'risk\s+factors',
    r'risks\s+and\s+uncertainties',
    r'critical\s+accounting',
    r'liquidity\s+risk',
    r'market\s+risk',
]


def normalize_text(text: str) -> str:
    """Normalize SEC filing text: lowercase, strip extra whitespace."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def count_words(text: str) -> int:
    """Count total words in text."""
    return len(text.split())


def find_hedging_matches(text: str) -> List[Tuple[str, int]]:
    """Find all hedging pattern matches and their counts."""
    matches: Dict[str, int] = {}
    normalized = normalize_text(text)
    
    for pattern in HEDGING_PATTERNS:
        found = re.findall(pattern, normalized, re.IGNORECASE)
        if found:
            matches[pattern] = len(found)
    
    return sorted(matches.items(), key=lambda x: x[1], reverse=True)


def detect_forward_looking_section(text: str) -> bool:
    """Check if text contains forward-looking statement markers."""
    normalized = normalize_text(text)
    for marker in FORWARD_LOOKING_MARKERS:
        if re.search(marker, normalized, re.IGNORECASE):
            return True
    return False


def detect_risk_section(text: str) -> bool:
    """Check if text contains risk section indicators."""
    normalized = normalize_text(text)
    for marker in RISK_SECTION_MARKERS:
        if re.search(marker, normalized, re.IGNORECASE):
            return True
    return False


def compute_whispers_density(filing_text: str) -> RegulatoryWhispersResult:
    """
    Analyze SEC filing text and compute hedging language density score (0–1).
    
    Args:
        filing_text: Raw or extracted SEC filing text (10-K, 10-Q, 8-K, etc.).
    
    Returns:
        RegulatoryWhispersResult with density score, matches, and top patterns.
    """
    normalized = normalize_text(filing_text)
    total_words = count_words(normalized)
    
    if total_words == 0:
        return RegulatoryWhispersResult(
            filing_text=filing_text,
            total_words=0,
            hedging_matches=[],
            density_score=0.0,
            top_patterns=[],
        )
    
    # Find all hedging matches.
    matches = find_hedging_matches(normalized)
    total_hedging_hits = sum(count for _, count in matches)
    
    # Compute raw density.
    raw_density = total_hedging_hits / total_words
    
    # Boost score if text is explicitly a risk section or forward-looking statement.
    multiplier = 1.0
    if detect_risk_section(normalized):
        multiplier *= 1.2
    if detect_forward_looking_section(normalized):
        multiplier *= 1.15
    
    # Cap at 1.0.
    final_density = min(raw_density * multiplier, 1.0)
    
    # Top 5 patterns by frequency.
    top_patterns = matches[:5]
    
    return RegulatoryWhispersResult(
        filing_text=filing_text,
        total_words=total_words,
        hedging_matches=matches,
        density_score=final_density,
        top_patterns=top_patterns,
    )


def score_regulatory_whispers(
    filing_text: str,
) -> Dict[str, float]:
    """
    High-level API: analyze filing and return confidence adjustment factor.
    
    Args:
        filing_text: SEC filing text.
    
    Returns:
        Dict with 'whispers_density' (0–1) and 'confidence_multiplier' (1.0 − density).
    """
    result = compute_whispers_density(filing_text)
    confidence_multiplier = 1.0 - result.density_score
    
    return {
        'whispers_density': result.density_score,
        'confidence_multiplier': confidence_multiplier,
        'total_words': result.total_words,
        'hedging_hit_count': sum(count for _, count in result.hedging_matches),
        'top_patterns': [p[0] for p in result.top_patterns],
    }
