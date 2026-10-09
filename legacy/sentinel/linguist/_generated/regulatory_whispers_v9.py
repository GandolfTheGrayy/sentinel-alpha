"""
Regulatory Whispers detector for Sentinel Sentiment Engine.

Scans SEC filings for hedging language patterns (e.g., 'may', 'subject to',
'could materially') and scores their density as a signal of management caution.
Integrates into the Linguist pillar's certainty and tone analysis workflow.

This module identifies regulatory/legal hedging as a proxy for management
uncertainty and risk acknowledgment. High hedging density may correlate with
lower conviction in forward guidance or undisclosed concerns.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class HedgingPattern:
    """A single hedging language pattern with weight and regex."""
    pattern: str
    weight: float
    compiled_regex: re.Pattern


class RegulatoryWhispersDetector:
    """Scanner for hedging language density in SEC filings and corporate disclosures."""

    # Curated hedging patterns with normalized weights (0.0–1.0).
    # Patterns are ordered by regulatory/legal strength and frequency.
    HEDGING_PATTERNS = [
        HedgingPattern(r"\bmay\b", 0.4, re.compile(r"\bmay\b", re.IGNORECASE)),
        HedgingPattern(r"\bmight\b", 0.3, re.compile(r"\bmight\b", re.IGNORECASE)),
        HedgingPattern(r"\bcould\b", 0.5, re.compile(r"\bcould\b", re.IGNORECASE)),
        HedgingPattern(r"\bsubject\s+to\b", 0.6, re.compile(r"\bsubject\s+to\b", re.IGNORECASE)),
        HedgingPattern(r"\bif\s+and\s+when\b", 0.5, re.compile(r"\bif\s+and\s+when\b", re.IGNORECASE)),
        HedgingPattern(r"\bpotential(?:ly)?\b", 0.4, re.compile(r"\bpotential(?:ly)?\b", re.IGNORECASE)),
        HedgingPattern(r"\bmaterially\b", 0.7, re.compile(r"\bmaterially\b", re.IGNORECASE)),
        HedgingPattern(r"\badverse(?:ly)?\b", 0.6, re.compile(r"\badverse(?:ly)?\b", re.IGNORECASE)),
        HedgingPattern(r"\brisk(?:s)?\b", 0.5, re.compile(r"\brisk(?:s)?\b", re.IGNORECASE)),
        HedgingPattern(r"\buncertain(?:ty|ties)?\b", 0.7, re.compile(r"\buncertain(?:ty|ties)?\b", re.IGNORECASE)),
        HedgingPattern(r"\bvariance\b", 0.4, re.compile(r"\bvariance\b", re.IGNORECASE)),
        HedgingPattern(r"\bfactors\b", 0.2, re.compile(r"\bfactors\b", re.IGNORECASE)),
        HedgingPattern(r"\bexpectation(?:s)?\b", 0.3, re.compile(r"\bexpectation(?:s)?\b", re.IGNORECASE)),
        HedgingPattern(r"\bassum(?:e|ption)(?:s)?\b", 0.4, re.compile(r"\bassum(?:e|ption)(?:s)?\b", re.IGNORECASE)),
        HedgingPattern(r"\bno\s+assurance\b", 0.8, re.compile(r"\bno\s+assurance\b", re.IGNORECASE)),
        HedgingPattern(r"\bcannot\s+(?:be\s+)?(?:assured|guaranteed)\b", 0.8, re.compile(r"\bcannot\s+(?:be\s+)?(?:assured|guaranteed)\b", re.IGNORECASE)),
    ]

    def __init__(self) -> None:
        """Initialize the Regulatory Whispers detector with hedge patterns."""
        self.patterns = self.HEDGING_PATTERNS

    def scan_text(self, text: str) -> Dict[str, object]:
        """
        Scan a text block for hedging language and return aggregated metrics.
        
        Args:
            text: Raw filing or disclosure text to analyze.
            
        Returns:
            Dictionary with keys:
              - 'hedging_score': float (0.0–1.0), normalized density of hedges.
              - 'hedge_count': int, total hedging instances found.
              - 'unique_patterns': int, distinct patterns matched.
              - 'pattern_breakdown': dict mapping pattern name to count.
              - 'weighted_density': float, hedge count × avg weight / word count.
        """
        if not text or not isinstance(text, str):
            return {
                "hedging_score": 0.0,
                "hedge_count": 0,
                "unique_patterns": 0,
                "pattern_breakdown": {},
                "weighted_density": 0.0,
            }

        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        
        pattern_counts: Dict[str, int] = {}
        total_weighted_score = 0.0
        
        for pattern_obj in self.patterns:
            matches = pattern_obj.compiled_regex.findall(text_lower)
            count = len(matches)
            if count > 0:
                pattern_counts[pattern_obj.pattern] = count
                total_weighted_score += count * pattern_obj.weight
        
        total_hedge_count = sum(pattern_counts.values())
        unique_patterns_count = len(pattern_counts)
        
        # Normalize score: capped at 1.0, scales with hedge density and average weight.
        avg_pattern_weight = (
            total_weighted_score / total_hedge_count
            if total_hedge_count > 0
            else 0.0
        )
        weighted_density = (total_hedge_count / word_count) * avg_pattern_weight
        hedging_score = min(1.0, weighted_density * 100)  # Scale to 0.0–1.0 range.
        
        return {
            "hedging_score": hedging_score,
            "hedge_count": total_hedge_count,
            "unique_patterns": unique_patterns_count,
            "pattern_breakdown": pattern_counts,
            "weighted_density": weighted_density,
        }

    def compare_texts(self, text_a: str, text_b: str) -> Dict[str, object]:
        """
        Compare hedging density across two text blocks (e.g., old vs. new filing).
        
        Args:
            text_a: First text (e.g., prior 10-Q).
            text_b: Second text (e.g., current 10-Q).
            
        Returns:
            Dictionary with 'text_a', 'text_b' scan results and 'delta_score' (b - a).
        """
        result_a = self.scan_text(text_a)
        result_b = self.scan_text(text_b)
        delta = result_b["hedging_score"] - result_a["hedging_score"]
        
        return {
            "text_a": result_a,
            "text_b": result_b,
            "delta_score": delta,
            "trend": "increasing_caution" if delta > 0.05 else (
                "decreasing_caution" if delta < -0.05 else "stable"
            ),
        }

    def extract_high_hedge_sentences(self, text: str, threshold: float = 0.5) -> List[str]:
        """
        Extract sentences with above-threshold hedging density.
        
        Args:
            text: Raw text to segment and analyze.
            threshold: Hedging score threshold (0.0–1.0) to flag sentences.
