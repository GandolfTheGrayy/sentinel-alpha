"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns that signal regulatory
caution or uncertainty. Identifies density of hedging terms like 'may',
'subject to', 'could materially', 'contingent', etc. and scores them
as a signal of underlying risk or compliance concern.

Used by Judge to weight predictions: high hedging density often precedes
negative surprises or regulatory action.
"""

import re
from typing import Dict, List, Tuple
import sqlite3


HEDGING_PATTERNS = {
    "may": r"\bmay\b",
    "might": r"\bmight\b",
    "could": r"\bcould\b",
    "subject_to": r"\bsubject\s+to\b",
    "contingent": r"\bcontingent\b",
    "materially": r"\bmaterially\b",
    "if_and_when": r"\bif\s+and\s+when\b",
    "depends_on": r"\bdepends\s+on\b",
    "uncertain": r"\buncertain(ty)?\b",
    "risks": r"\brisk(s)?\b",
    "pending": r"\bpending\b",
    "unless": r"\bunless\b",
    "except": r"\bexcept\b",
    "approximately": r"\bapproximately\b",
    "estimated": r"\bestimated\b",
}

INTENSIFIER_PATTERNS = {
    "significant": r"\bsignificant(ly)?\b",
    "material": r"\bmaterial(ly)?\b",
    "substantial": r"\bsubstantial(ly)?\b",
    "substantial_risk": r"\bsubstantial\s+risk\b",
}


def tokenize_filing(text: str) -> List[str]:
    """Split filing text into sentences for context-aware analysis."""
    sentences = re.split(r"[.!?]\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_hedging_instances(
    text: str, hedging_dict: Dict[str, str]
) -> Dict[str, List[str]]:
    """Extract sentences containing hedging patterns with context."""
    instances = {}
    sentences = tokenize_filing(text)
    
    for hedge_name, pattern in hedging_dict.items():
        matches = []
        for sentence in sentences:
            if re.search(pattern, sentence, re.IGNORECASE):
                matches.append(sentence)
        instances[hedge_name] = matches
    
    return instances


def count_hedging_density(text: str) -> Dict[str, int]:
    """Count raw occurrences of each hedging pattern in text."""
    text_lower = text.lower()
    densities = {}
    
    for hedge_name, pattern in HEDGING_PATTERNS.items():
        count = len(re.findall(pattern, text_lower))
        densities[hedge_name] = count
    
    return densities


def score_regulatory_whispers(
    text: str, word_count: int = None
) -> Tuple[float, Dict[str, float]]:
    """
    Score hedging language density in SEC filing.
    
    Returns:
        (overall_score, breakdown_dict) where overall_score is 0–1
        (0 = low hedging, 1 = high hedging) and breakdown_dict
        maps pattern names to normalized densities.
    """
    if not text or not isinstance(text, str):
        return 0.0, {}
    
    if word_count is None:
        word_count = len(text.split())
    
    if word_count == 0:
        return 0.0, {}
    
    densities = count_hedging_density(text)
    total_hedging_hits = sum(densities.values())
    
    # Normalize by word count (hedges per 1000 words)
    normalized_density = (total_hedging_hits / word_count) * 1000
    
    # Calibrate scale: typical 10-K has ~20–60 hedges per 1000 words
    # Map [0, 100] to [0, 1] with saturation
    raw_score = min(normalized_density / 100.0, 1.0)
    
    # Build breakdown (also normalized)
    breakdown = {
        name: min((count / word_count) * 1000 / 20.0, 1.0)
        for name, count in densities.items()
    }
    
    return raw_score, breakdown


def detect_intensified_hedging(text: str) -> Dict[str, int]:
    """
    Identify hedging phrases intensified by modifiers.
    
    Counts patterns like 'material risk', 'significant uncertainty',
    which signal higher concern than baseline hedging.
    """
    text_lower = text.lower()
    intensified = {}
    
    for pattern_name, pattern in INTENSIFIER_PATTERNS.items():
        count = len(re.findall(pattern, text_lower))
        intensified[pattern_name] = count
    
    return intensified


def analyze_filing_section(
    section_text: str, section_name: str = "Unknown"
) -> Dict[str, float]:
    """
    Analyze a single section of a filing (e.g., Risk Factors, MD&A).
    
    Returns dict with keys: score, density, intensifier_count, word_count.
    """
    word_count = len(section_text.split())
    score, breakdown = score_regulatory_whispers(section_text, word_count)
    intensified = detect_intensified_hedging(section_text)
    intensifier_count = sum(intensified.values())
    
    return {
        "section": section_name,
        "score": score,
        "density": score,
        "intensifier_count": intensifier_count,
        "word_count": word_count,
        "breakdown": breakdown,
    }


def compare_filing_versions(
    old_text: str, new_text: str
) -> Dict[str, float]:
    """
    Compare hedging language between two versions of the same filing.
    
    Returns dict with keys: old_score, new_score, drift (new - old).
    Positive drift = increased hedging language.
    """
    old_score, _ = score_regulatory_whispers(old_text)
    new_score, _ = score_regulatory_whispers(new_text)
    drift = new_score - old_score
    
    return {
        "old_score": old_score,
        "new_score": new_score,
        "drift": drift,
        "trend": "increased_hedging" if drift > 0.05 else
                 "decreased_hedging" if drift < -0.05 else
                 "stable",
    }


def store_whispers_score(
    db_path: str, ticker: str, filing_type: str, score: float, 
    section_scores: Dict = None
) -> None:
    """
    Persist regulatory whispers score to SQLite for historical tracking.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS regulatory_whispers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            filing_type TEXT NOT NULL,
            score REAL NOT NULL,
            section_scores TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
    )
    
    import json
    section_json = json.dumps(section_scores or {})
    
    cursor.execute(
        """
        INSERT INTO regulatory_whispers (ticker, filing_type, score, section_scores)
        VALUES (?, ?, ?, ?)
        """,
        (ticker, filing_type, score, section_json),
    )
    
    conn.commit()
    conn.close()


def retrieve_whispers_trend(
    db_path: str, ticker: str, filing_type: str, limit: int = 5
) -> List[Dict
