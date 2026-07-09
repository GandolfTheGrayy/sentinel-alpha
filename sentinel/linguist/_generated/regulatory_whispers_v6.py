"""
Regulatory Whispers Detector — part of Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns that signal management uncertainty
or legal caution. Analyzes density of terms like 'may', 'subject to', 'could materially',
'if and to the extent', 'notwithstanding', etc. across 8-K, 10-Q, 10-K filings.

Returns a whispers_score (0–1) representing the intensity of regulatory hedging,
which correlates with market volatility and downside risk. Used by Judge to
calibrate confidence in price predictions.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class RegulatoryWhisper:
    """Single detected hedging phrase with metadata."""
    phrase: str
    category: str
    count: int
    position: int


@dataclass
class RegulatoryWhispersResult:
    """Aggregated hedging analysis for a filing."""
    ticker: str
    filing_type: str
    whispers_score: float
    total_hedges_detected: int
    hedge_categories: Dict[str, int]
    risk_signal: str
    raw_text_sample: str


# Regulatory hedging lexicon organized by category and intensity
HEDGING_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "uncertainty": {
        "high_intensity": [
            r"\bmay\b",
            r"\bcould\b",
            r"\bmight\b",
            r"\bif\b",
            r"\bpossibly\b",
            r"\bapparently\b",
            r"\ballegedly\b",
        ],
        "medium_intensity": [
            r"\bbelieve\b",
            r"\bexpect\b",
            r"\banticipatе\b",
            r"\bsubject to\b",
            r"\bdepending on\b",
        ],
    },
    "materiality": {
        "high_intensity": [
            r"\bcould materially\b",
            r"\bmay materially\b",
            r"\bmaterial.*risk\b",
            r"\bsignificant.*uncertainty\b",
        ],
        "medium_intensity": [
            r"\badverse.*effect\b",
            r"\bnegative.*impact\b",
            r"\bnot.*guaranteed\b",
        ],
    },
    "limitation": {
        "high_intensity": [
            r"\bnotwithstanding\b",
            r"\bexcept\b",
            r"\bunless\b",
            r"\bif and to the extent\b",
            r"\bsubject to.*exception\b",
        ],
        "medium_intensity": [
            r"\blimited to\b",
            r"\bonly to the extent\b",
        ],
    },
    "contingency": {
        "high_intensity": [
            r"\bcontingent\b",
            r"\bcontingency\b",
            r"\bconditioned upon\b",
            r"\bconditional\b",
        ],
        "medium_intensity": [
            r"\bin the event\b",
            r"\bshould occur\b",
        ],
    },
    "disclaimer": {
        "high_intensity": [
            r"\brisk factor\b",
            r"\bno assurance\b",
            r"\bno guarantee\b",
            r"\bdisclaimer\b",
        ],
        "medium_intensity": [
            r"\bwarning\b",
            r"\bcaution\b",
            r"\baside from\b",
        ],
    },
}


def extract_hedging_phrases(
    text: str, lexicon: Dict[str, Dict[str, List[str]]] = HEDGING_LEXICON
) -> List[RegulatoryWhisper]:
    """
    Extract all hedging phrases from SEC filing text.

    Args:
        text: Raw SEC filing text (8-K, 10-Q, 10-K body).
        lexicon: Hedging phrase patterns organized by category.

    Returns:
        List of RegulatoryWhisper objects with positions and categories.
    """
    whispers: List[RegulatoryWhisper] = []
    text_lower = text.lower()

    for category, intensity_dict in lexicon.items():
        for intensity, patterns in intensity_dict.items():
            for pattern in patterns:
                # Case-insensitive finditer
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    phrase = match.group(0)
                    whispers.append(
                        RegulatoryWhisper(
                            phrase=phrase,
                            category=f"{category}/{intensity}",
                            count=1,
                            position=match.start(),
                        )
                    )

    return whispers


def compute_whispers_score(
    text: str,
    filing_type: str = "10-Q",
    word_count: int = None,
) -> Tuple[float, Dict[str, int], int]:
    """
    Compute normalized hedging intensity score (0–1) from SEC filing text.

    Args:
        text: Raw SEC filing text.
        filing_type: Type of filing (8-K, 10-Q, 10-K) for context weighting.
        word_count: Optional pre-computed word count; if None, computed from text.

    Returns:
        Tuple of (whispers_score, hedge_categories_dict, total_hedge_count).
        whispers_score: normalized 0–1, where 1 = maximum hedging density.
    """
    if not text or len(text.strip()) < 100:
        return 0.0, {}, 0

    whispers = extract_hedging_phrases(text)

    if not whispers:
        return 0.0, {}, 0

    # Count hedges by category
    hedge_categories: Dict[str, int] = {}
    for whisper in whispers:
        cat = whisper.category.split("/")[0]
        hedge_categories[cat] = hedge_categories.get(cat, 0) + 1

    total_hedges = len(whispers)

    # Compute word count if not provided
    if word_count is None:
        word_count = len(text.split())

    # Normalize: hedges per 1000 words
    hedge_density = (total_hedges / max(word_count, 1)) * 1000

    # Apply filing-type weighting: 8-K usually shorter, so density is more significant
    filing_weights = {"8-K": 1.3, "10-Q": 1.0, "10-K": 0.9}
    weight = filing_weights.get(filing_type, 1.0)
    weighted_density = hedge_density * weight

    # Scale to 0–1: assume max reasonable density is ~50 hedges per 1000 words
    whispers_score = min(weighted_density / 50.0, 1.0)

    return whispers_score, hedge_categories, total_hedges


def analyze_filing(
    ticker: str,
    filing_type: str,
    text: str,
) -> RegulatoryWhispersResult:
    """
    Analyze a single SEC filing for regulatory hedging intensity.

    Args:
        ticker: Stock ticker symbol.
        filing_type: Filing type (8-K, 10-Q, 10-K).
        text: Raw filing text body.

    Returns:
        RegulatoryWhispersResult with aggregated whispers metrics.
    """
    score, categories, total = compute_whispers_score(text, filing_type)

    # Determine risk signal
    if score >= 0.7:
        risk_signal = "CRITICAL"
    elif score >= 0.5:
        risk_signal = "HIGH"
    elif score >= 0.3:
        risk_signal = "MODERATE"
    else:
        risk_signal = "LOW"

    # Extract a sample passage with high hedge density
    sentences = re.split(r"[.!?]", text)
    hedge_
