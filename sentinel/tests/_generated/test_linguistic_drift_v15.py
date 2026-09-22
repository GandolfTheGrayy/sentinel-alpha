"""
Unit tests for the Linguistic Drift detector module.

This test suite validates the Linguistic Drift detector's ability to identify
tone and sentiment shifts in company communications over time. It uses fixture
text samples representing historical SEC filings and news to assert correct
drift scoring and anomaly flagging.

Part of Sentinel's Linguist pillar — ensures linguistic shift detection is
accurate before integration into daily prediction pipeline.
"""

import pytest
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DriftScore:
    """Container for linguistic drift analysis results."""
    ticker: str
    metric_name: str
    baseline_score: float
    current_score: float
    drift_magnitude: float
    is_anomaly: bool
    confidence: float


class LinguisticDriftDetector:
    """Detects tone and sentiment shifts in company communications."""

    def __init__(self, anomaly_threshold: float = 0.35):
        """
        Initialize drift detector with anomaly threshold.

        Args:
            anomaly_threshold: Score delta above which shift is flagged anomalous.
        """
        self.anomaly_threshold = anomaly_threshold

    def analyze_drift(
        self,
        ticker: str,
        baseline_text: str,
        current_text: str,
        metric_name: str = "tone_shift"
    ) -> DriftScore:
        """
        Compute linguistic drift between baseline and current text samples.

        Args:
            ticker: Stock ticker symbol.
            baseline_text: Historical reference text (e.g., prior quarter filing).
            current_text: Current communication text to compare.
            metric_name: Name of metric being tracked (e.g., "tone_shift", "caution_index").

        Returns:
            DriftScore object with magnitude, anomaly flag, and confidence.
        """
        baseline_score = self._score_text(baseline_text)
        current_score = self._score_text(current_text)
        drift_magnitude = abs(current_score - baseline_score)
        is_anomaly = drift_magnitude > self.anomaly_threshold
        confidence = min(1.0, 0.5 + (drift_magnitude * 1.5))

        return DriftScore(
            ticker=ticker,
            metric_name=metric_name,
            baseline_score=baseline_score,
            current_score=current_score,
            drift_magnitude=drift_magnitude,
            is_anomaly=is_anomaly,
            confidence=confidence
        )

    def _score_text(self, text: str) -> float:
        """
        Score text for tone/sentiment on scale [0, 1].

        Args:
            text: Text to analyze.

        Returns:
            Tone score (0=negative, 0.5=neutral, 1=positive).
        """
        text_lower = text.lower()
        positive_words = [
            "growth", "strong", "expansion", "innovative", "leader", "exceed",
            "improve", "opportunity", "success", "profitable", "confidence"
        ]
        negative_words = [
            "decline", "risk", "challenge", "uncertain", "volatile", "loss",
            "weak", "pressure", "concern", "headwind", "disruption"
        ]

        pos_count = sum(text_lower.count(word) for word in positive_words)
        neg_count = sum(text_lower.count(word) for word in negative_words)
        total = pos_count + neg_count

        if total == 0:
            return 0.5

        return pos_count / total

    def batch_analyze(
        self,
        samples: List[Tuple[str, str, str, str]]
    ) -> List[DriftScore]:
        """
        Analyze drift for multiple ticker/text pairs.

        Args:
            samples: List of (ticker, baseline_text, current_text, metric_name) tuples.

        Returns:
            List of DriftScore objects.
        """
        return [
            self.analyze_drift(ticker, baseline, current, metric)
            for ticker, baseline, current, metric in samples
        ]

    def flag_anomalies(self, scores: List[DriftScore]) -> List[DriftScore]:
        """
        Filter and return only anomalous drift events.

        Args:
            scores: List of DriftScore objects.

        Returns:
            Subset of scores where is_anomaly == True.
        """
        return [s for s in scores if s.is_anomaly]


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def detector() -> LinguisticDriftDetector:
    """Instantiate detector with default threshold."""
    return LinguisticDriftDetector(anomaly_threshold=0.35)


@pytest.fixture
def sample_texts() -> Dict[str, Tuple[str, str]]:
    """
    Fixture: baseline and current SEC filing excerpts for test companies.

    Returns:
        Dict mapping ticker to (baseline_text, current_text) tuples.
    """
    return {
        "AAPL": (
            # Baseline: Q3 2023 optimistic filing
            "Apple delivered strong revenue growth and innovative product expansion. "
            "Market leadership continues with record profits and customer confidence. "
            "We see significant opportunities ahead for technological advancement.",
            # Current: Q3 2024 cautious tone
            "Revenue growth has decelerated due to market pressures and uncertain "
            "macroeconomic conditions. We face significant headwinds in key markets. "
            "Challenges persist despite our innovative efforts."
        ),
        "TSLA": (
            # Baseline: bullish 2023
            "Tesla achieved record production and profitability. Expansion into new "
            "markets demonstrates our leadership. Innovation and success drive growth.",
            # Current: 2024 downgrade
            "Disruption and volatile market conditions create uncertainty. Pressure "
            "on margins and loss of market share in key segments pose risks."
        ),
        "GOOGL": (
            # Baseline: steady 2023
            "Strong revenue growth with continued innovation. Opportunities abound "
            "in cloud and AI. Our leader position strengthens quarterly.",
            # Current: 2024 stable (minimal drift)
            "We maintain strong revenue growth with innovative AI initiatives. "
            "Cloud expansion creates opportunity. Leadership position continues."
        ),
    }


@pytest.fixture
def batch_samples(sample_texts) -> List[Tuple[str, str, str, str]]:
    """
    Fixture: batch of (ticker, baseline, current, metric) tuples.

    Args:
        sample_texts: Sample fixture.

    Returns:
        Formatted list for batch_analyze() input.
    """
    samples = []
    for ticker, (baseline, current) in sample_texts.items():
        samples.append((ticker, baseline, current, "tone_shift"))
    return samples


# ============================================================================
# TEST CASES
# ============================================================================

class TestLinguisticDriftDetector:
    """Test suite for LinguisticDriftDetector."""

    def test_detector_instantiation(self) -> None:
        """Verify detector initializes with configurable threshold."""
        detector = LinguisticDriftDetector(anomaly_threshold=0.25)
        assert detector.anomaly_threshold == 0.25

    def test_score_text_positive(self, detector: LinguisticDriftDetector) -> None:
        """Assert positive text yields score > 0.5."""
        text = "Strong growth and innovative expansion with excellent opportunity."
        score = detector._score_text(text)
        assert score > 0.5, f"Expected positive score, got {score}"

    def test_score_text_negative(self, detector: LinguisticDriftDetector) -> None:
        """Assert negative text yields score < 0.5."""
        text = "Decline, risk, and uncertain challenges ahead."
        score = detector._score_text(text)
        assert score < 0.5, f"Expected negative score, got {score}"

    def test_score_text_neutral(self, detector: LinguisticDriftDetector) -> None:
        """Assert neutral text (no sentiment words) yields ~0.5."""
        text = "The company reported quarterly results and held a meeting."
        score = detector._score_text(text)
        assert 0
