"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to identify
tone shifts in company communications over time. It uses fixture text samples
representing different sentiment states and verifies that drift scoring
correctly quantifies changes in linguistic patterns, certainty levels, and
regulatory language intensity across temporal windows.

Part of Sentinel's Linguist pillar validation suite.
"""

import pytest
from typing import Dict, List, Tuple


class MockDriftDetector:
    """Mock implementation of a Linguistic Drift detector for testing."""
    
    def __init__(self) -> None:
        """Initialize the drift detector with baseline vocabulary weights."""
        self.certainty_markers = {
            "will": 0.9, "shall": 0.95, "must": 0.85,
            "may": 0.3, "could": 0.2, "might": 0.15,
            "expect": 0.6, "anticipate": 0.65, "believe": 0.4
        }
        self.caution_markers = {
            "risk": -0.3, "uncertain": -0.5, "challenge": -0.2,
            "headwind": -0.25, "volatile": -0.4, "pressure": -0.15
        }
        self.regulatory_markers = {
            "SEC": 0.1, "compliance": 0.15, "filing": 0.08,
            "disclosure": 0.12, "auditor": 0.1, "restatement": -0.3
        }
    
    def compute_certainty_score(self, text: str) -> float:
        """
        Compute normalized certainty score [0.0, 1.0] for input text.
        
        Higher scores indicate more assertive, confident language.
        """
        tokens = text.lower().split()
        score = 0.5  # neutral baseline
        weight = 0.0
        
        for token in tokens:
            token_clean = token.strip(".,;:!?").lower()
            if token_clean in self.certainty_markers:
                score += self.certainty_markers[token_clean] * 0.05
                weight += 0.05
            if token_clean in self.caution_markers:
                score += self.caution_markers[token_clean] * 0.05
                weight += 0.05
        
        if weight > 0:
            score = (score + (0.5 * weight)) / (1.0 + weight)
        
        return max(0.0, min(1.0, score))
    
    def compute_regulatory_intensity(self, text: str) -> float:
        """
        Compute regulatory language intensity [0.0, 1.0] for input text.
        
        Higher scores indicate denser regulatory/compliance vocabulary.
        """
        tokens = text.lower().split()
        intensity = 0.0
        
        for token in tokens:
            token_clean = token.strip(".,;:!?").lower()
            if token_clean in self.regulatory_markers:
                intensity += abs(self.regulatory_markers[token_clean])
        
        # Normalize by token count
        if len(tokens) > 0:
            intensity = intensity / len(tokens)
        
        return max(0.0, min(1.0, intensity))
    
    def compute_drift(
        self,
        baseline_text: str,
        current_text: str
    ) -> Dict[str, float]:
        """
        Compute drift vector between two text samples.
        
        Returns dict with keys: certainty_drift, regulatory_drift, combined_drift.
        Positive values = shift toward more cautious/regulatory language.
        """
        baseline_certainty = self.compute_certainty_score(baseline_text)
        current_certainty = self.compute_certainty_score(current_text)
        certainty_drift = baseline_certainty - current_certainty
        
        baseline_regulatory = self.compute_regulatory_intensity(baseline_text)
        current_regulatory = self.compute_regulatory_intensity(current_text)
        regulatory_drift = current_regulatory - baseline_regulatory
        
        combined_drift = (certainty_drift + regulatory_drift) / 2.0
        
        return {
            "certainty_drift": certainty_drift,
            "regulatory_drift": regulatory_drift,
            "combined_drift": combined_drift,
            "baseline_certainty": baseline_certainty,
            "current_certainty": current_certainty,
            "baseline_regulatory": baseline_regulatory,
            "current_regulatory": current_regulatory
        }
    
    def detect_anomaly(self, drift: float, threshold: float = 0.3) -> bool:
        """
        Detect if drift magnitude exceeds anomaly threshold.
        
        Returns True if absolute drift exceeds threshold.
        """
        return abs(drift) > threshold


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def detector() -> MockDriftDetector:
    """Provide a fresh drift detector instance for each test."""
    return MockDriftDetector()


@pytest.fixture
def baseline_optimistic() -> str:
    """Baseline text with confident, positive tone."""
    return (
        "We will achieve strong revenue growth this year. "
        "Our markets will expand significantly. "
        "We expect robust demand and believe our strategy will succeed."
    )


@pytest.fixture
def current_cautious() -> str:
    """Current text with cautious, defensive tone."""
    return (
        "We may see modest growth. Revenue could be affected by risks. "
        "Market challenges and uncertain conditions might impact performance. "
        "We face headwinds and volatile pressure in key segments."
    )


@pytest.fixture
def baseline_light_regulatory() -> str:
    """Baseline with minimal regulatory language."""
    return (
        "Our operations are running smoothly. Product demand is strong. "
        "Customer satisfaction drives our growth strategy. "
        "We are optimistic about future performance."
    )


@pytest.fixture
def current_heavy_regulatory() -> str:
    """Current text with increased regulatory/compliance focus."""
    return (
        "SEC compliance remains critical. Our latest filing discloses "
        "new risks. The auditor raised concerns in recent disclosure. "
        "We ensure regulatory adherence and full SEC compliance measures."
    )


@pytest.fixture
def neutral_text() -> str:
    """Text with neutral, balanced tone."""
    return (
        "We report quarterly results. Revenue was stable. "
        "Market conditions were mixed. We maintain our strategic direction."
    )


# ============================================================================
# TESTS: Certainty Scoring
# ============================================================================

def test_certainty_baseline_optimistic(
    detector: MockDriftDetector,
    baseline_optimistic: str
) -> None:
    """Optimistic text should score high on certainty [0.7, 1.0]."""
    score = detector.compute_certainty_score(baseline_optimistic)
    assert 0.7 <= score <= 1.0, f"Expected high certainty, got {score}"


def test_certainty_current_cautious(
    detector: MockDriftDetector,
    current_cautious: str
) -> None:
    """Cautious text should score low on certainty [0.0, 0.4]."""
    score = detector.compute_certainty_score(current_cautious)
    assert 0.0 <= score <= 0.4, f"Expected low certainty, got {score}"


def test_certainty_neutral_midrange(
    detector: MockDriftDetector,
    neutral_text: str
) -> None:
    """Neutral text should score in middle range [0.3, 0.7]."""
    score = detector.compute_certainty_score(neutral_text)
    assert 0.3 <= score <= 0.7, f"Expected neutral certainty, got {score}"


def test_certainty_empty_string(detector: MockDriftDetector) -> None:
    """Empty text should default to neutral baseline score."""
    score = detector.compute_certainty_score("")
    assert score == 0.5, f"Expected baseline 0.5, got {score}"


# ============================================================================
# TESTS: Regulatory Intensity Scoring
# ============================================================================

def test_
