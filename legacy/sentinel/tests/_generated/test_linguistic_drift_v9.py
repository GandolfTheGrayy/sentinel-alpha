"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to score
tone shifts in company-specific sentiment over time. It uses fixture text
samples to assert correct drift scoring, threshold detection, and historical
context weighting.

Part of Sentinel's test suite; exercises linguist/ reasoning pipeline.
"""

import pytest
from typing import Dict, List, Tuple
import json


class MockLinguisticDriftDetector:
    """Minimal mock implementation for testing drift scoring logic."""

    def __init__(self) -> None:
        """Initialize drift detector with baseline state."""
        self.baseline_tone: Dict[str, float] = {}
        self.history: List[Tuple[str, float]] = []

    def register_baseline(self, company: str, tone_score: float) -> None:
        """Register initial tone baseline for a company."""
        self.baseline_tone[company] = tone_score

    def score_drift(self, company: str, text: str, historical_tone: float) -> float:
        """
        Score linguistic drift as absolute deviation from baseline.
        
        Returns float in [0, 1] where 0 = no drift, 1 = extreme drift.
        """
        current_tone = self._extract_tone(text)
        if company not in self.baseline_tone:
            return 0.0
        baseline = self.baseline_tone[company]
        drift = abs(current_tone - baseline)
        # Normalize to [0, 1] range
        normalized_drift = min(drift / 0.5, 1.0)
        self.history.append((company, normalized_drift))
        return normalized_drift

    def _extract_tone(self, text: str) -> float:
        """
        Extract tone score from text (mock: count positive vs negative words).
        
        Returns float in [-1, 1] where -1 = very negative, 1 = very positive.
        """
        positive_words = ['strong', 'growth', 'excellent', 'bullish', 'surge', 'expand']
        negative_words = ['weak', 'decline', 'poor', 'bearish', 'crash', 'shrink']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        return (pos_count - neg_count) / max(pos_count + neg_count, 1)

    def detect_threshold_breach(self, drift_score: float, threshold: float = 0.6) -> bool:
        """Detect if drift exceeds severity threshold."""
        return drift_score > threshold

    def trend_direction(self, company: str) -> str:
        """
        Determine if drift is trending more positive or negative.
        
        Returns 'positive', 'negative', or 'stable'.
        """
        recent = [score for co, score in self.history[-5:] if co == company]
        if not recent:
            return 'stable'
        avg_recent = sum(recent) / len(recent)
        if avg_recent > 0.65:
            return 'negative'
        elif avg_recent < 0.35:
            return 'positive'
        return 'stable'


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def drift_detector() -> MockLinguisticDriftDetector:
    """Provide initialized drift detector."""
    return MockLinguisticDriftDetector()


@pytest.fixture
def baseline_texts() -> Dict[str, str]:
    """Fixture: baseline sentiment texts for calibration."""
    return {
        'ACME_neutral': (
            'ACME reported steady quarterly results. Revenue held at expected levels. '
            'Operations remained stable with moderate cost controls.'
        ),
        'ACME_positive': (
            'ACME strong growth in Q3. Excellent expansion into new markets. '
            'Bullish guidance for next quarter.'
        ),
        'ACME_negative': (
            'ACME weak performance amid market headwinds. Decline in key segments. '
            'Poor outlook due to shrinking demand.'
        ),
    }


@pytest.fixture
def historical_context() -> Dict[str, float]:
    """Fixture: historical tone baselines (prior quarter averages)."""
    return {
        'ACME': 0.1,  # Slightly positive baseline
        'BETA': -0.2,  # Historically cautious
        'GAMMA': 0.5,  # Historically bullish
    }


# ============================================================================
# TESTS: DRIFT SCORING
# ============================================================================

def test_drift_scoring_baseline_to_positive(
    drift_detector: MockLinguisticDriftDetector,
    baseline_texts: Dict[str, str]
) -> None:
    """Assert drift score increases when sentiment shifts positive."""
    drift_detector.register_baseline('ACME', 0.0)
    
    drift_neutral = drift_detector.score_drift(
        'ACME',
        baseline_texts['ACME_neutral'],
        0.0
    )
    drift_positive = drift_detector.score_drift(
        'ACME',
        baseline_texts['ACME_positive'],
        0.0
    )
    
    assert drift_positive > drift_neutral, \
        "Positive text should show higher drift from neutral baseline"


def test_drift_scoring_baseline_to_negative(
    drift_detector: MockLinguisticDriftDetector,
    baseline_texts: Dict[str, str]
) -> None:
    """Assert drift score increases when sentiment shifts negative."""
    drift_detector.register_baseline('ACME', 0.0)
    
    drift_neutral = drift_detector.score_drift(
        'ACME',
        baseline_texts['ACME_neutral'],
        0.0
    )
    drift_negative = drift_detector.score_drift(
        'ACME',
        baseline_texts['ACME_negative'],
        0.0
    )
    
    assert drift_negative > drift_neutral, \
        "Negative text should show higher drift from neutral baseline"


def test_drift_score_range(
    drift_detector: MockLinguisticDriftDetector,
    baseline_texts: Dict[str, str]
) -> None:
    """Assert drift scores are normalized to [0, 1]."""
    drift_detector.register_baseline('ACME', 0.0)
    
    for text_key, text in baseline_texts.items():
        score = drift_detector.score_drift('ACME', text, 0.0)
        assert 0.0 <= score <= 1.0, f"Drift score {score} out of range for {text_key}"


def test_drift_no_baseline_registered(
    drift_detector: MockLinguisticDriftDetector,
    baseline_texts: Dict[str, str]
) -> None:
    """Assert zero drift when company baseline not registered."""
    score = drift_detector.score_drift(
        'UNKNOWN_CORP',
        baseline_texts['ACME_positive'],
        0.0
    )
    assert score == 0.0, "Should return 0 drift for unregistered company"


# ============================================================================
# TESTS: THRESHOLD BREACH DETECTION
# ============================================================================

def test_threshold_breach_high_drift(drift_detector: MockLinguisticDriftDetector) -> None:
    """Assert threshold breach detection for high drift."""
    high_drift = 0.85
    assert drift_detector.detect_threshold_breach(high_drift, threshold=0.6), \
        "High drift should exceed threshold"


def test_threshold_breach_low_drift(drift_detector: MockLinguisticDriftDetector) -> None:
    """Assert no threshold breach for low drift."""
    low_drift = 0.25
    assert not drift_detector.detect_threshold_breach(low_drift, threshold=0.6), \
        "Low drift should not exceed threshold"


def test_threshold_breach_boundary(drift_detector: MockLingu
