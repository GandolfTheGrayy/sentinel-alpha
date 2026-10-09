"""
Regulatory Whispers Detector — Sentinel Linguist pillar.

Scans SEC filings for hedging language patterns that signal management caution
or regulatory constraints. Identifies keywords like 'may', 'subject to', 'could
materially', 'contingent', 'risk', etc., and produces a density score indicating
the degree of linguistic hedging in the filing.

Used by Judge to weight predictions: high hedging often correlates with
downside risk or management uncertainty.
"""

import re
from typing import Dict, List, Tuple
import sqlite3
from pathlib import Path


# Hedging language patterns organized by intensity
HEDGING_PATTERNS = {
    "high_intensity": [
        r"\bmay\s+(?:be|have|result|impact|cause|affect)",
        r"\bcould\s+materially",
        r"\bsubject\s+to\s+(?:risks?|uncertainties?|contingencies?)",
        r"\bcontingent\s+(?:upon|on)",
        r"\bmaterial(?:ly)?\s+(?:risk|adverse|decline|loss)",
        r"\bif\s+conditions?\s+(?:change|occur|arise)",
        r"\bno\s+assurance",
        r"\buncertain\b",
        r"\bvolatil",
    ],
    "medium_intensity": [
        r"\bmay\b",
        r"\bcould\b",
        r"\bmight\b",
        r"\brisk\b",
        r"\bsubject\s+to\b",
        r"\badverse(?:ly)?\b",
        r"\bchallenges?\b",
        r"\buncertain(?:ty|ties)?\b",
        r"\bfluctuat",
    ],
    "low_intensity": [
        r"\blikely\b",
        r"\bpossible(?:ly)?\b",
        r"\bexpect(?:ed|ation)?\b",
        r"\bestimate(?:d)?\b",
        r"\bapproximate(?:ly)?\b",
    ],
}


def extract_hedging_signals(text: str) -> Dict[str, int]:
    """
    Count hedging language patterns in SEC filing text by intensity tier.
    
    Returns dict with keys 'high_intensity', 'medium_intensity', 'low_intensity',
    each mapping to integer match counts.
    """
    if not text:
        return {"high_intensity": 0, "medium_intensity": 0, "low_intensity": 0}
    
    # Normalize: lowercase, handle line breaks
    normalized = text.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    
    results = {}
    for intensity, patterns in HEDGING_PATTERNS.items():
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            count += len(matches)
        results[intensity] = count
    
    return results


def compute_hedging_density(text: str) -> float:
    """
    Compute normalized hedging density: 0.0 (no hedging) to 1.0 (extreme hedging).
    
    Weights high_intensity patterns 3x, medium 2x, low 1x.
    Normalizes by text length (word count).
    """
    if not text or len(text.strip()) == 0:
        return 0.0
    
    signals = extract_hedging_signals(text)
    
    weighted_score = (
        signals["high_intensity"] * 3.0
        + signals["medium_intensity"] * 2.0
        + signals["low_intensity"] * 1.0
    )
    
    # Normalize by word count (rough proxy for filing length)
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    
    # Scale: 100 hedges per 1000 words = density ~0.3
    # Clamp to [0, 1]
    density = min(1.0, (weighted_score / (word_count / 1000.0)) / 100.0)
    return density


def analyze_filing(
    ticker: str,
    filing_type: str,
    text: str,
) -> Dict[str, any]:
    """
    Analyze a single SEC filing for regulatory whispers.
    
    Returns dict with keys:
      - ticker: stock symbol
      - filing_type: e.g. '8-K', '10-Q', '10-K'
      - hedging_density: float [0, 1]
      - high_intensity_count: int
      - medium_intensity_count: int
      - low_intensity_count: int
      - interpretation: string label ('minimal', 'moderate', 'high', 'extreme')
    """
    signals = extract_hedging_signals(text)
    density = compute_hedging_density(text)
    
    # Map density to interpretation
    if density < 0.1:
        interpretation = "minimal"
    elif density < 0.25:
        interpretation = "moderate"
    elif density < 0.5:
        interpretation = "high"
    else:
        interpretation = "extreme"
    
    return {
        "ticker": ticker,
        "filing_type": filing_type,
        "hedging_density": round(density, 4),
        "high_intensity_count": signals["high_intensity"],
        "medium_intensity_count": signals["medium_intensity"],
        "low_intensity_count": signals["low_intensity"],
        "interpretation": interpretation,
    }


def store_hedging_analysis(
    db_path: str,
    ticker: str,
    filing_type: str,
    hedging_density: float,
    high_count: int,
    medium_count: int,
    low_count: int,
    interpretation: str,
) -> bool:
    """
    Store hedging analysis result in SQLite for historical tracking.
    
    Creates table if missing. Returns True on success.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS hedging_analysis (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                filing_type TEXT NOT NULL,
                hedging_density REAL NOT NULL,
                high_count INTEGER,
                medium_count INTEGER,
                low_count INTEGER,
                interpretation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        
        cursor.execute(
            """
            INSERT INTO hedging_analysis
            (ticker, filing_type, hedging_density, high_count, medium_count, low_count, interpretation)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ticker, filing_type, hedging_density, high_count, medium_count, low_count, interpretation),
        )
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing hedging analysis: {e}")
        return False


def compare_filings(
    analyses: List[Dict[str, any]],
) -> Dict[str, any]:
    """
    Compare hedging density across multiple filings (e.g. 10-Q over quarters).
    
    Returns summary: avg density, trend (increasing/stable/decreasing), outliers.
    """
    if not analyses:
        return {}
    
    densities = [a["hedging_density"] for a in analyses]
    
    avg_density = sum(densities) / len(densities)
    
    # Detect trend
    if len(densities) >= 2:
        recent = densities[-1]
        prior = densities[-2]
        if recent > prior * 1.1:
            trend = "increasing"
        elif recent < prior * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"
    
    return {
        "average_density": round(avg_density, 4),
        "trend": trend,
