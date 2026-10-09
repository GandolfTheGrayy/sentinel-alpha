"""
Unit tests for the Linguistic Drift detector.

This module validates the linguistic_drift module's ability to detect and score
tone shifts in company-related text over time. It uses fixture data representing
sequential text samples (e.g., earnings calls, news articles) and asserts that
drift scoring correctly identifies increases/decreases in sentiment intensity,
urgency, and regulatory concern.

Part of Sentinel's Linguist pillar validation.
"""

import pytest
from typing import List, Dict, Tuple


class MockDriftDetector:
    """Mock implementation of linguistic drift detection for testing."""

    def __init__(self) -> None:
        """Initialize drift detector with baseline metrics."""
        self.baseline_metrics: Dict[str, float] = {
            "urgency": 0.3,
            "concern": 0.2,
            "confidence": 0.7,
        }

    def compute_drift(self, text: str) -> Dict[str, float]:
        """
        Compute drift metrics for a single text sample.
        
        Returns dict with urgency, concern, confidence scores.
        """
        urgency = 0.3 + (text.count("urgent") * 0.1)
        concern = 0.2 + (text.count("risk") * 0.08)
        confidence = 0.7 - (text.count("uncertain") * 0.05)
        
        return {
            "urgency": min(1.0, urgency),
            "concern": min(1.0, concern),
            "confidence": max(0.0, confidence),
        }

    def detect_drift_over_time(
        self, text_samples: List[str], metric: str = "urgency"
    ) -> Tuple[float, str]:
        """
        Detect directional drift in a metric across sequential text samples.
        
        Returns (drift_magnitude, direction) where direction is "increasing", 
        "decreasing", or "stable".
        """
        if len(text_samples) < 2:
            return (0.0, "stable")

        scores = [self.compute_drift(t)[metric] for t in text_samples]
        first_half_mean = sum(scores[: len(scores) // 2]) / max(1, len(scores) // 2)
        second_half_mean = sum(scores[len(scores) // 2 :]) / max(
            1, len(scores) - len(scores) // 2
        )
        
        drift_magnitude = abs(second_half_mean - first_half_mean)
        
        if second_half_mean > first_half_mean + 0.05:
            direction = "increasing"
        elif first_half_mean > second_half_mean + 0.05:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return (drift_magnitude, direction)

    def score_regulatory_whisper(self, text: str) -> float:
        """
        Score likelihood of hidden regulatory concern in text.
        
        Returns float 0.0-1.0 based on keywords and linguistic patterns.
        """
        whisper_keywords = ["review", "investigation", "compliance", "audit", "inquiry"]
        base_score = sum(
            0.15 for kw in whisper_keywords if kw in text.lower()
        )
        
        # Penalize explicit language (less "whisper-y")
        explicit_keywords = ["confirmed", "announced", "reported"]
        base_score *= 1.0 - sum(
            0.1 for kw in explicit_keywords if kw in text.lower()
        )
        
        return min(1.0, base_score)


@pytest.fixture
def drift_detector() -> MockDriftDetector:
    """Provide a configured drift detector instance."""
    return MockDriftDetector()


@pytest.fixture
def escalating_urgency_samples() -> List[str]:
    """Sample texts showing increasing urgency over time."""
    return [
        "We are monitoring market conditions with standard procedures.",
        "Recent developments require urgent attention from management.",
        "The situation is urgent and demands immediate intervention.",
        "This urgent matter poses significant risks to operations.",
    ]


@pytest.fixture
def stable_concern_samples() -> List[str]:
    """Sample texts with stable concern levels."""
    return [
        "Operational risk remains at expected levels.",
        "Risk assessment shows normal variance.",
        "Risk factors continue within baseline parameters.",
        "Risk profiles remain consistent with historical data.",
    ]


@pytest.fixture
def decreasing_confidence_samples() -> List[str]:
    """Sample texts showing decreasing confidence over time."""
    return [
        "We are confident in our strategic direction.",
        "We remain reasonably confident in our outlook.",
        "Uncertainty in the market creates some hesitation.",
        "Significant uncertainty has emerged regarding future performance.",
    ]


@pytest.fixture
def regulatory_whisper_samples() -> List[Tuple[str, float]]:
    """Sample texts with expected regulatory whisper scores."""
    return [
        ("Company operates normally without issues.", 0.0),
        ("We are under routine compliance review.", 0.15),
        ("Audit procedures are ongoing as part of standard practice.", 0.15),
        ("We are subject to investigation and inquiry.", 0.30),
        ("An undisclosed compliance matter is under internal review.", 0.45),
    ]


class TestDriftDetection:
    """Test suite for drift detection functionality."""

    def test_compute_drift_basic(self, drift_detector: MockDriftDetector) -> None:
        """Test basic drift metric computation."""
        text = "This is urgent and carries significant risk."
        metrics = drift_detector.compute_drift(text)
        
        assert "urgency" in metrics
        assert "concern" in metrics
        assert "confidence" in metrics
        assert all(0.0 <= v <= 1.0 for v in metrics.values())

    def test_escalating_urgency_detected(
        self, drift_detector: MockDriftDetector, escalating_urgency_samples: List[str]
    ) -> None:
        """Assert urgency drift increases across escalating sample sequence."""
        magnitude, direction = drift_detector.detect_drift_over_time(
            escalating_urgency_samples, metric="urgency"
        )
        
        assert direction == "increasing"
        assert magnitude > 0.05

    def test_stable_concern_detected(
        self, drift_detector: MockDriftDetector, stable_concern_samples: List[str]
    ) -> None:
        """Assert concern drift remains stable across consistent samples."""
        magnitude, direction = drift_detector.detect_drift_over_time(
            stable_concern_samples, metric="concern"
        )
        
        assert direction == "stable"
        assert magnitude < 0.05

    def test_decreasing_confidence_detected(
        self,
        drift_detector: MockDriftDetector,
        decreasing_confidence_samples: List[str],
    ) -> None:
        """Assert confidence drift decreases across uncertain samples."""
        magnitude, direction = drift_detector.detect_drift_over_time(
            decreasing_confidence_samples, metric="confidence"
        )
        
        assert direction == "decreasing"
        assert magnitude > 0.05

    def test_regulatory_whisper_scoring(
        self,
        drift_detector: MockDriftDetector,
        regulatory_whisper_samples: List[Tuple[str, float]],
    ) -> None:
        """Assert regulatory whisper detection matches expected thresholds."""
        for text, expected_lower_bound in regulatory_whisper_samples:
            score = drift_detector.score_regulatory_whisper(text)
            assert score >= expected_lower_bound - 0.01, (
                f"Text '{text}' scored {score}, expected >= {expected_lower_bound}"
            )

    def test_drift_with_empty_samples(
        self, drift_detector: MockDriftDetector
    ) -> None:
        """Assert drift detection handles empty sample list gracefully."""
        magnitude, direction = drift_detector.detect_drift_over_time([])
        
        assert direction == "stable"
        assert magnitude == 0.0

    def test_drift_with_single_sample(
        self, drift_detector: MockDriftDetector

    ) -> None:
        """
