"""
Unit tests for the Linguistic Drift detector.

This module validates the drift scoring logic that measures tone/sentiment shifts
in company communications over time. Tests use fixture text spanning multiple periods
to assert correct detection of shifts in certainty language, regulatory language,
and risk acknowledgment patterns.

Part of the Sentinel Sentiment Engine's test harness.
"""

import pytest
from typing import Dict, List, Tuple


# Mock drift detector implementation for testing
class LinguisticDriftDetector:
    """Detects shifts in linguistic patterns across time periods."""

    def __init__(self) -> None:
        """Initialize the drift detector with baseline patterns."""
        self.certainty_markers = {
            "high": ["will", "expect", "confident", "strong", "leading"],
            "medium": ["should", "likely", "probable", "trend"],
            "low": ["may", "could", "might", "uncertain", "challenging"],
        }
        self.risk_markers = [
            "risk",
            "uncertain",
            "volatility",
            "headwind",
            "challenge",
            "difficult",
        ]
        self.regulatory_markers = [
            "sec",
            "compliance",
            "regulation",
            "investigation",
            "violation",
            "filing",
        ]

    def score_text(self, text: str) -> Dict[str, float]:
        """
        Score a single text passage for certainty, risk, and regulatory language.

        Returns dict with keys: certainty (0-1), risk_density (0-1), regulatory_density (0-1).
        """
        text_lower = text.lower()
        words = text_lower.split()

        high_count = sum(1 for w in words if w in self.certainty_markers["high"])
        med_count = sum(1 for w in words if w in self.certainty_markers["medium"])
        low_count = sum(1 for w in words if w in self.certainty_markers["low"])

        total_certainty = high_count + med_count + low_count
        if total_certainty == 0:
            certainty_score = 0.5
        else:
            certainty_score = (
                high_count * 1.0 + med_count * 0.5 + low_count * 0.0
            ) / total_certainty

        risk_count = sum(1 for word in self.risk_markers if word in text_lower)
        risk_density = min(1.0, risk_count / max(1, len(words) / 100))

        reg_count = sum(1 for word in self.regulatory_markers if word in text_lower)
        regulatory_density = min(1.0, reg_count / max(1, len(words) / 100))

        return {
            "certainty": certainty_score,
            "risk_density": risk_density,
            "regulatory_density": regulatory_density,
        }

    def compute_drift(
        self, period_texts: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """
        Compute drift scores across time periods.

        Args:
            period_texts: List of (period_label, text) tuples in chronological order.

        Returns dict with keys: certainty_drift, risk_drift, regulatory_drift (all 0-1).
        """
        scores = [self.score_text(text) for _, text in period_texts]

        if len(scores) < 2:
            return {
                "certainty_drift": 0.0,
                "risk_drift": 0.0,
                "regulatory_drift": 0.0,
            }

        certainty_drift = abs(scores[-1]["certainty"] - scores[0]["certainty"])
        risk_drift = abs(scores[-1]["risk_density"] - scores[0]["risk_density"])
        regulatory_drift = abs(
            scores[-1]["regulatory_density"] - scores[0]["regulatory_density"]
        )

        return {
            "certainty_drift": certainty_drift,
            "risk_drift": risk_drift,
            "regulatory_drift": regulatory_drift,
        }


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def detector() -> LinguisticDriftDetector:
    """Provide a fresh LinguisticDriftDetector instance."""
    return LinguisticDriftDetector()


@pytest.fixture
def stable_periods() -> List[Tuple[str, str]]:
    """Text from periods with minimal tone shift."""
    return [
        (
            "Q1 2023",
            "We expect strong growth this year. Leading market position will drive revenue.",
        ),
        (
            "Q2 2023",
            "We should continue to perform well. Trend remains positive for our business.",
        ),
        (
            "Q3 2023",
            "We will maintain leadership. Strong fundamentals will sustain growth.",
        ),
    ]


@pytest.fixture
def deteriorating_periods() -> List[Tuple[str, str]]:
    """Text from periods showing deterioration in tone."""
    return [
        (
            "Q1 2023",
            "We will expand significantly. Confident in our strong market position.",
        ),
        (
            "Q2 2023",
            "Growth may slow due to headwinds. Uncertain macroeconomic conditions.",
        ),
        (
            "Q3 2023",
            "Risk of declining revenue. Volatility and challenges ahead.",
        ),
    ]


@pytest.fixture
def regulatory_shift_periods() -> List[Tuple[str, str]]:
    """Text from periods with rising regulatory mentions."""
    return [
        ("Q1 2023", "Operating normally. No significant compliance issues."),
        (
            "Q2 2023",
            "SEC inquiry filed. We are reviewing compliance with new regulations.",
        ),
        (
            "Q3 2023",
            "Ongoing SEC investigation. Multiple regulatory filings required. Violation penalties possible.",
        ),
    ]


@pytest.fixture
def high_certainty_text() -> str:
    """Single passage with strong certainty language."""
    return "We will lead the market. Confident expectations for strong growth."


@pytest.fixture
def low_certainty_text() -> str:
    """Single passage with weak certainty language."""
    return "We may face challenges. Uncertain about future trends and risks."


# ============================================================================
# UNIT TESTS
# ============================================================================


class TestLinguisticDriftDetector:
    """Test suite for LinguisticDriftDetector."""

    def test_detector_initialization(self, detector: LinguisticDriftDetector) -> None:
        """Detector initializes with expected marker categories."""
        assert "high" in detector.certainty_markers
        assert "medium" in detector.certainty_markers
        assert "low" in detector.certainty_markers
        assert len(detector.risk_markers) > 0
        assert len(detector.regulatory_markers) > 0

    def test_score_text_high_certainty(
        self, detector: LinguisticDriftDetector, high_certainty_text: str
    ) -> None:
        """High-certainty text scores > 0.7 on certainty metric."""
        score = detector.score_text(high_certainty_text)
        assert score["certainty"] > 0.7
        assert 0.0 <= score["certainty"] <= 1.0

    def test_score_text_low_certainty(
        self, detector: LinguisticDriftDetector, low_certainty_text: str
    ) -> None:
        """Low-certainty text scores < 0.5 on certainty metric."""
        score = detector.score_text(low_certainty_text)
        assert score["certainty"] < 0.5
        assert 0.0 <= score["certainty"] <= 1.0

    def test_score_text_has_all_keys(
        self, detector: LinguisticDriftDetector, high_certainty_text: str
    ) -> None:
        """Score result includes all required keys."""
        score = detector.score_text(high_certainty_text)
        assert "certainty" in score
        assert "risk_density" in score
        assert "regulatory_density" in score
