"""
Unit tests for the Linguistic Drift detector module.

This test suite validates the core drift-scoring logic that detects
tone and sentiment shifts in company communications over time.
Linguistic Drift is a key signal in the Sentinel Sentiment Engine:
it flags when a company's language patterns (confidence, hedging,
regulatory tone) diverge from historical baselines, often preceding
material price moves.

Tests use fixture text samples spanning multiple time periods to
assert correct drift magnitude and direction scoring.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


class MockLinguisticDriftDetector:
    """Minimal implementation for testing drift detection logic."""

    def __init__(self):
        """Initialize detector with empty baseline registry."""
        self.baselines: Dict[str, Dict[str, float]] = {}

    def compute_drift_score(
        self, company_ticker: str, current_text: str, reference_period_days: int = 90
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute linguistic drift as (magnitude, component_scores).

        Args:
            company_ticker: Stock ticker symbol.
            current_text: Current filing or communication text.
            reference_period_days: Lookback window for baseline.

        Returns:
            Tuple of (drift_magnitude: 0.0–1.0, component_breakdown: dict).
        """
        baseline = self.baselines.get(company_ticker, self._default_baseline())
        current = self._extract_features(current_text)

        drift_components = {
            "confidence_shift": abs(current["confidence"] - baseline["confidence"]),
            "hedging_shift": abs(current["hedging"] - baseline["hedging"]),
            "regulatory_shift": abs(
                current["regulatory_tone"] - baseline["regulatory_tone"]
            ),
            "uncertainty_shift": abs(
                current["uncertainty"] - baseline["uncertainty"]
            ),
        }

        magnitude = sum(drift_components.values()) / len(drift_components)
        return min(magnitude, 1.0), drift_components

    def register_baseline(self, company_ticker: str, text: str) -> None:
        """Register a baseline text snapshot for a company."""
        self.baselines[company_ticker] = self._extract_features(text)

    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic feature scores from text (0.0–1.0 scale)."""
        text_lower = text.lower()

        confidence_words = [
            "will",
            "expect",
            "confident",
            "strong",
            "growing",
            "accelerating",
        ]
        confidence = sum(
            1 for w in confidence_words if w in text_lower
        ) / max(1, len(confidence_words))

        hedging_words = [
            "may",
            "might",
            "could",
            "approximately",
            "estimated",
            "subject to",
        ]
        hedging = sum(1 for w in hedging_words if w in text_lower) / max(
            1, len(hedging_words)
        )

        regulatory_words = [
            "sec",
            "regulation",
            "compliance",
            "disclosure",
            "filing",
            "audit",
        ]
        regulatory_tone = sum(1 for w in regulatory_words if w in text_lower) / max(
            1, len(regulatory_words)
        )

        uncertainty_words = ["uncertain", "risk", "decline", "challenge", "difficult"]
        uncertainty = sum(1 for w in uncertainty_words if w in text_lower) / max(
            1, len(uncertainty_words)
        )

        return {
            "confidence": confidence,
            "hedging": hedging,
            "regulatory_tone": regulatory_tone,
            "uncertainty": uncertainty,
        }

    def _default_baseline(self) -> Dict[str, float]:
        """Return neutral baseline feature scores."""
        return {
            "confidence": 0.5,
            "hedging": 0.5,
            "regulatory_tone": 0.3,
            "uncertainty": 0.3,
        }


@pytest.fixture
def detector() -> MockLinguisticDriftDetector:
    """Provide a fresh detector instance for each test."""
    return MockLinguisticDriftDetector()


@pytest.fixture
def fixture_texts() -> Dict[str, str]:
    """Provide realistic company filing text samples."""
    return {
        "baseline_optimistic": (
            "We are confident in our strong market position and expect accelerating "
            "growth in the coming quarters. Our pipeline is robust and we will deliver "
            "exceptional returns to shareholders."
        ),
        "shifted_cautious": (
            "We may face headwinds from market conditions. Our outlook is uncertain "
            "and could be subject to regulatory challenges. Risks remain elevated and "
            "we estimate potential difficulties ahead."
        ),
        "shifted_pessimistic": (
            "Declining revenues and difficult market conditions present uncertain "
            "prospects. We risk further challenges. Compliance issues may arise from "
            "recent SEC disclosures in our audit filings."
        ),
        "neutral_regulatory": (
            "The company filed this 10-Q with the SEC in accordance with regulation. "
            "Disclosure of material facts is subject to audit procedures and compliance "
            "requirements outlined in our regulatory framework."
        ),
    }


class TestLinguisticDriftBasic:
    """Test basic drift detection mechanics."""

    def test_drift_magnitude_range(self, detector: MockLinguisticDriftDetector) -> None:
        """Assert drift magnitude is always in [0.0, 1.0]."""
        test_text = "We expect strong growth and confident expansion into new markets."
        magnitude, _ = detector.compute_drift_score("AAPL", test_text)
        assert 0.0 <= magnitude <= 1.0, f"Drift magnitude out of range: {magnitude}"

    def test_drift_components_complete(
        self, detector: MockLinguisticDriftDetector
    ) -> None:
        """Assert all required components are returned."""
        test_text = "Uncertain times ahead, may face regulatory challenges."
        _, components = detector.compute_drift_score("MSFT", test_text)
        required_keys = {"confidence_shift", "hedging_shift", "regulatory_shift", "uncertainty_shift"}
        assert required_keys.issubset(
            components.keys()
        ), f"Missing components: {required_keys - components.keys()}"

    def test_zero_drift_when_unchanged(
        self, detector: MockLinguisticDriftDetector, fixture_texts: Dict[str, str]
    ) -> None:
        """Assert zero drift when comparing text to itself."""
        baseline = fixture_texts["baseline_optimistic"]
        detector.register_baseline("TEST", baseline)
        magnitude, _ = detector.compute_drift_score("TEST", baseline)
        assert magnitude == 0.0, f"Expected zero drift, got {magnitude}"


class TestLinguisticDriftShifts:
    """Test detection of meaningful tone shifts."""

    def test_optimistic_to_cautious_shift(
        self,
        detector: MockLinguisticDriftDetector,
        fixture_texts: Dict[str, str],
    ) -> None:
        """Assert high drift when tone shifts from optimistic to cautious."""
        detector.register_baseline("GOOG", fixture_texts["baseline_optimistic"])
        magnitude, components = detector.compute_drift_score(
            "GOOG", fixture_texts["shifted_cautious"]
        )
        assert magnitude > 0.3, (
            f"Expected significant drift (>0.3), got {magnitude}. "
            f"Components: {components}"
        )
        assert components["confidence_shift"] > 0.2, "Confidence should drop"
        assert components["hedging_shift"] > 0.2, "Hedging should increase"

    def test_optimistic_to_pessimistic_shift(
        self,
        detector: MockLinguisticDriftDetector,
        fixture_texts: Dict[str, str],
    ) -> None:
        """Assert maximum drift for optimistic→pessimistic swing."""
        detector.register_
