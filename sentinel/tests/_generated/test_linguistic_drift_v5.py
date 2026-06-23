"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to identify
tone shifts in company communications (SEC filings, news, social media) over
time. Drift scoring helps Sentinel detect emerging sentiment changes that may
precede stock price movements.

Tests use fixture text samples representing different sentiment polarities and
lexical shifts, asserting that drift scores correctly reflect magnitude and
direction of tone changes.
"""

import pytest
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch, MagicMock
import json


class LinguisticDriftDetector:
    """Mock detector for testing; mirrors production API."""

    def __init__(self) -> None:
        """Initialize detector with empty history."""
        self.history: Dict[str, List[Dict]] = {}

    def record_sample(self, ticker: str, text: str, timestamp: float) -> None:
        """Record a text sample for drift tracking."""
        if ticker not in self.history:
            self.history[ticker] = []
        self.history[ticker].append({
            "text": text,
            "timestamp": timestamp,
            "polarity": self._compute_polarity(text),
        })

    def _compute_polarity(self, text: str) -> float:
        """Compute simple polarity: -1.0 (negative) to 1.0 (positive)."""
        positive_words = ["growth", "strong", "excellent", "robust", "upside"]
        negative_words = ["decline", "weak", "risk", "challenge", "downside"]

        pos_count = sum(1 for w in positive_words if w in text.lower())
        neg_count = sum(1 for w in negative_words if w in text.lower())

        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def compute_drift(self, ticker: str, window: int = 2) -> Tuple[float, str]:
        """
        Compute linguistic drift as magnitude of polarity shift over last window samples.

        Returns:
            (drift_score, direction) where drift_score is [0, 1] and direction is "positive", "negative", or "stable".
        """
        if ticker not in self.history or len(self.history[ticker]) < window:
            return 0.0, "stable"

        samples = self.history[ticker][-window:]
        polarities = [s["polarity"] for s in samples]

        if len(polarities) < 2:
            return 0.0, "stable"

        drift_magnitude = abs(polarities[-1] - polarities[-2])

        if polarities[-1] > polarities[-2]:
            direction = "positive"
        elif polarities[-1] < polarities[-2]:
            direction = "negative"
        else:
            direction = "stable"

        return drift_magnitude, direction

    def batch_compute_drift(
        self, ticker: str, window: int = 3
    ) -> Dict[str, float]:
        """Compute drift metrics over multiple time windows."""
        if ticker not in self.history or len(self.history[ticker]) < window:
            return {"drift_score": 0.0, "confidence": 0.0}

        samples = self.history[ticker][-window:]
        polarities = [s["polarity"] for s in samples]

        # Compute variance across window
        mean_polarity = sum(polarities) / len(polarities)
        variance = (
            sum((p - mean_polarity) ** 2 for p in polarities) / len(polarities)
        )

        # Confidence: how stable is the drift signal?
        # High variance = low confidence (noisy), low variance = high confidence
        confidence = 1.0 - min(variance, 1.0)

        return {
            "drift_score": variance,
            "confidence": confidence,
            "polarity_trajectory": polarities,
        }


@pytest.fixture
def detector() -> LinguisticDriftDetector:
    """Provide a fresh detector instance for each test."""
    return LinguisticDriftDetector()


@pytest.fixture
def positive_samples() -> List[str]:
    """Text samples with positive sentiment."""
    return [
        "Strong growth in Q3 with robust revenue expansion.",
        "Excellent market position and upside potential ahead.",
        "Growth drivers remain intact with strong momentum.",
    ]


@pytest.fixture
def negative_samples() -> List[str]:
    """Text samples with negative sentiment."""
    return [
        "Significant decline in revenue and downside risks emerging.",
        "Weak performance due to market challenges and competition.",
        "Risk factors pose downside to near-term outlook.",
    ]


@pytest.fixture
def neutral_samples() -> List[str]:
    """Text samples with neutral sentiment."""
    return [
        "The company operates in multiple markets and segments.",
        "Operations are conducted across several geographic regions.",
        "Products are available to consumers and businesses.",
    ]


class TestLinguisticDriftBasics:
    """Test basic polarity computation and single-sample recording."""

    def test_polarity_positive(self, detector: LinguisticDriftDetector) -> None:
        """Assert positive text yields positive polarity."""
        text = "Strong growth and excellent results."
        polarity = detector._compute_polarity(text)
        assert polarity > 0.0, "Positive text should yield positive polarity"

    def test_polarity_negative(self, detector: LinguisticDriftDetector) -> None:
        """Assert negative text yields negative polarity."""
        text = "Decline and weak performance with risks."
        polarity = detector._compute_polarity(text)
        assert polarity < 0.0, "Negative text should yield negative polarity"

    def test_polarity_neutral(self, detector: LinguisticDriftDetector) -> None:
        """Assert neutral text yields near-zero polarity."""
        text = "The company operates in many regions."
        polarity = detector._compute_polarity(text)
        assert abs(polarity) <= 0.1, "Neutral text should yield near-zero polarity"

    def test_record_single_sample(
        self, detector: LinguisticDriftDetector
    ) -> None:
        """Assert sample recording increments history."""
        detector.record_sample("AAPL", "Strong growth", 1000.0)
        assert "AAPL" in detector.history
        assert len(detector.history["AAPL"]) == 1
        assert detector.history["AAPL"][0]["text"] == "Strong growth"


class TestLinguisticDriftDetection:
    """Test drift scoring across multiple samples."""

    def test_no_drift_identical_polarity(
        self, detector: LinguisticDriftDetector
    ) -> None:
        """Assert identical polarity yields zero drift (stable)."""
        detector.record_sample("TSLA", "Strong growth ahead", 1000.0)
        detector.record_sample("TSLA", "Strong performance continues", 2000.0)
        drift, direction = detector.compute_drift("TSLA", window=2)
        assert drift < 0.1, "Identical polarity should yield near-zero drift"
        assert direction == "stable", "Direction should be stable"

    def test_positive_drift(self, detector: LinguisticDriftDetector) -> None:
        """Assert shift from negative to positive yields positive drift."""
        detector.record_sample("MSFT", "Decline and weak results", 1000.0)
        detector.record_sample("MSFT", "Strong growth and upside", 2000.0)
        drift, direction = detector.compute_drift("MSFT", window=2)
        assert drift > 0.5, "Shift to positive should yield high drift"
        assert direction == "positive", "Direction should be positive"

    def test_negative_drift(self, detector: LinguisticDriftDetector) -> None:
        """Assert shift from positive to negative yields negative drift."""
        detector.record_sample("GOOGL", "Excellent growth and momentum", 1000.0)
        detector.record_sample("GOOGL", "Decline and downside risks", 2000.0)
