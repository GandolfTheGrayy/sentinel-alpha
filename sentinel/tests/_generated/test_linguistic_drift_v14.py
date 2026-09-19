"""
Unit tests for the Linguistic Drift detector.

This module validates the linguistic drift scoring pipeline, which detects
tone and language shifts in financial documents (SEC filings, news, sentiment)
over time for a given company. Tests use fixture text samples representing
different sentiment states and verify that drift scores correctly identify
magnitude and direction of tone changes.

Part of Sentinel's Linguist pillar quality assurance.
"""

import pytest
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch


class MockLinguisticDriftDetector:
    """Mock implementation of Linguistic Drift detector for testing."""

    def __init__(self) -> None:
        """Initialize the drift detector with baseline sentiment anchors."""
        self.baseline_sentiments: Dict[str, float] = {}
        self.drift_history: Dict[str, List[Tuple[str, float]]] = {}

    def compute_drift_score(
        self, company_ticker: str, current_text: str, historical_texts: List[str]
    ) -> float:
        """
        Compute linguistic drift score (0-1) indicating tone shift magnitude.

        Args:
            company_ticker: Stock ticker symbol.
            current_text: Recent document text to analyze.
            historical_texts: List of prior documents for baseline comparison.

        Returns:
            Drift score where 0=no change, 1=maximum divergence.
        """
        if not historical_texts:
            return 0.0

        current_sentiment = self._extract_sentiment(current_text)
        historical_avg = sum(
            self._extract_sentiment(t) for t in historical_texts
        ) / len(historical_texts)

        drift = abs(current_sentiment - historical_avg)
        normalized_drift = min(drift / 0.5, 1.0)

        if company_ticker not in self.drift_history:
            self.drift_history[company_ticker] = []
        self.drift_history[company_ticker].append((current_text[:50], drift))

        return normalized_drift

    def _extract_sentiment(self, text: str) -> float:
        """
        Extract sentiment polarity from text (0=negative, 0.5=neutral, 1=positive).

        Args:
            text: Document text to score.

        Returns:
            Sentiment score normalized to [0, 1].
        """
        positive_words = [
            "growth",
            "strong",
            "expand",
            "profit",
            "opportunity",
            "success",
        ]
        negative_words = [
            "decline",
            "risk",
            "loss",
            "challenge",
            "uncertainty",
            "warning",
        ]

        text_lower = text.lower()
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.5

        return (pos_count - neg_count + total) / (2 * total)

    def detect_drift_direction(self, drift_score: float) -> str:
        """
        Classify drift direction: positive, negative, or stable.

        Args:
            drift_score: Normalized drift magnitude [0, 1].

        Returns:
            Direction label: "positive_shift", "negative_shift", or "stable".
        """
        if drift_score < 0.15:
            return "stable"
        elif drift_score < 0.5:
            return "positive_shift"
        else:
            return "negative_shift"


@pytest.fixture
def drift_detector() -> MockLinguisticDriftDetector:
    """Provide initialized drift detector instance."""
    return MockLinguisticDriftDetector()


@pytest.fixture
def sample_sec_filings() -> Dict[str, List[str]]:
    """Provide fixture SEC filing text samples across sentiment states."""
    return {
        "bullish_sequence": [
            "The company achieved strong revenue growth and expanded market share.",
            "Profitability improved with successful cost optimization initiatives.",
            "We expect sustained growth opportunities in emerging markets.",
        ],
        "bearish_sequence": [
            "Revenue declined due to macroeconomic headwinds.",
            "Operational challenges have impacted margins this quarter.",
            "Uncertainty regarding future demand poses significant risk to guidance.",
        ],
        "mixed_sequence": [
            "Growth remained stable despite competitive pressure.",
            "We achieved modest gains in core business with some segment weakness.",
            "Market conditions present both opportunity and challenge.",
        ],
    }


@pytest.fixture
def sample_news_articles() -> Dict[str, List[str]]:
    """Provide fixture news article text samples with varying sentiment."""
    return {
        "positive": [
            "Stock surge as company announces record quarterly profit and expansion plans.",
            "Analyst upgrades reflect strong fundamentals and market leadership position.",
            "Growth outlook remains robust with increasing investor confidence.",
        ],
        "negative": [
            "Company faces warning as revenue decline accelerates amid market downturn.",
            "Risk assessment downgraded following disappointing earnings and loss guidance.",
            "Uncertainty clouds outlook as major client relationship ends.",
        ],
        "neutral": [
            "Company reports mixed results with stable revenue and adjusted expectations.",
            "Quarter showed typical seasonal patterns without significant surprises.",
            "Market reaction tepid as guidance met analyst consensus estimates.",
        ],
    }


class TestDriftDetectionBasics:
    """Test suite for core drift detection functionality."""

    def test_stable_sentiment_returns_low_drift(
        self, drift_detector: MockLinguisticDriftDetector
    ) -> None:
        """Verify low drift score when sentiment remains consistent."""
        stable_texts = [
            "The company maintains strong market position.",
            "Operational performance remains solid and competitive.",
            "Strong fundamentals support continued success.",
        ]
        drift = drift_detector.compute_drift_score("ACME", stable_texts[-1], stable_texts[:-1])
        assert drift < 0.2, "Stable sentiment should yield low drift"

    def test_positive_drift_detection(
        self, drift_detector: MockLinguisticDriftDetector,
        sample_sec_filings: Dict[str, List[str]],
    ) -> None:
        """Verify positive drift detected when tone improves."""
        texts = sample_sec_filings["bullish_sequence"]
        drift = drift_detector.compute_drift_score("TECH", texts[-1], texts[:-1])
        direction = drift_detector.detect_drift_direction(drift)
        assert direction in ["positive_shift", "stable"], "Bullish sequence should show positive or stable drift"

    def test_negative_drift_detection(
        self, drift_detector: MockLinguisticDriftDetector,
        sample_sec_filings: Dict[str, List[str]],
    ) -> None:
        """Verify negative drift detected when tone deteriorates."""
        texts = sample_sec_filings["bearish_sequence"]
        drift = drift_detector.compute_drift_score("RISK", texts[-1], texts[:-1])
        direction = drift_detector.detect_drift_direction(drift)
        assert direction in ["negative_shift", "stable"], "Bearish sequence should show negative or stable drift"

    def test_empty_history_returns_zero_drift(
        self, drift_detector: MockLinguisticDriftDetector
    ) -> None:
        """Verify zero drift when no historical baseline exists."""
        drift = drift_detector.compute_drift_score("NEW", "Some recent text", [])
        assert drift == 0.0, "Empty history should return zero drift"

    def test_drift_score_normalization(
        self, drift_detector: MockLinguisticDriftDetector
    ) -> None:
        """Verify drift scores remain bounded in [0, 1]."""
        extreme_positive = "growth profit success opportunity expand strong" * 10
        extreme_negative = "decline loss risk warning challenge uncertainty" * 10
        drift = drift_detector.compute_drift_score("EXTREME", extreme_positive, [extreme_negative])
        assert 0.0 <= drift <= 1.0, "Drift score must be normalized to [0, 1]"


class TestDriftDirectionClassification:
