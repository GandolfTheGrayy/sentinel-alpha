"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to identify
tone shifts in company communications over time. It uses fixture text samples
representing different sentiment polarities and temporal contexts, asserting
that drift scoring correctly flags meaningful divergence from historical baselines.

Part of Sentinel's Linguist pillar test suite.
"""

import pytest
from typing import Dict, List, Tuple


# Fixture text samples representing different company communication tones
FIXTURE_TEXTS = {
    "optimistic_q1": """
    We are thrilled to report record-breaking Q1 results. Our innovative product
    launches have exceeded expectations, and customer acquisition is accelerating.
    We remain bullish on our market position and anticipate sustained growth momentum.
    """,
    "cautious_q2": """
    Q2 results reflect softer demand patterns in key verticals. We are taking
    proactive steps to optimize cost structure. While we maintain conviction in
    our long-term strategy, near-term macro headwinds warrant measured guidance.
    """,
    "pessimistic_q3": """
    Q3performance deteriorated significantly. Customer churn increased 15% YoY,
    and pipeline visibility has contracted. Management is concerned about sustained
    competitive pressure and recessionary risks. We are implementing contingency plans.
    """,
    "recovery_q4": """
    Q4 showed early signs of stabilization. New customer wins resumed, and retention
    improved sequentially. We are cautiously optimistic about emerging opportunities
    and expect normalized growth trajectory in 2025.
    """,
    "neutral_earnings": """
    Revenue for the period was $500M, representing 2% YoY growth. Operating expenses
    decreased 3% due to operational efficiencies. Earnings per share came in at $1.25.
    We continue to invest in core competencies.
    """,
}


class MockDriftDetector:
    """Mock Linguistic Drift detector for testing."""

    def __init__(self) -> None:
        """Initialize the drift detector with baseline sentiment anchors."""
        self.baseline_sentiment: Dict[str, float] = {}
        self.sentiment_history: List[Tuple[str, float]] = []

    def analyze_text(self, text: str) -> float:
        """
        Compute sentiment score for input text (-1.0 to 1.0).
        
        Uses keyword frequency heuristics: positive words boost score,
        negative words reduce it, neutral text centers near 0.
        """
        positive_keywords = [
            "thrilled", "record-breaking", "exceeded", "bullish", "accelerating",
            "innovative", "excited", "growth", "momentum", "opportunity",
            "optimistic", "stabilization", "improved", "wins"
        ]
        negative_keywords = [
            "softer", "concerned", "headwinds", "deteriorated", "churn",
            "contingency", "pressure", "recessionary", "concern", "contracted",
            "risks", "weakness", "challenged"
        ]

        text_lower = text.lower()
        pos_count = sum(text_lower.count(kw) for kw in positive_keywords)
        neg_count = sum(text_lower.count(kw) for kw in negative_keywords)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        score = (pos_count - neg_count) / total
        return max(-1.0, min(1.0, score))

    def set_baseline(self, ticker: str, text: str) -> None:
        """
        Establish baseline sentiment for a ticker from reference text.
        
        Args:
            ticker: Stock symbol.
            text: Reference communication text.
        """
        self.baseline_sentiment[ticker] = self.analyze_text(text)

    def compute_drift(self, ticker: str, text: str) -> float:
        """
        Compute drift magnitude relative to baseline (-1.0 to 1.0).
        
        Args:
            ticker: Stock symbol.
            text: Current communication text.

        Returns:
            Drift score: positive = more optimistic than baseline,
            negative = more pessimistic than baseline.
        """
        if ticker not in self.baseline_sentiment:
            raise ValueError(f"No baseline set for {ticker}")

        current_sentiment = self.analyze_text(text)
        baseline = self.baseline_sentiment[ticker]
        drift = current_sentiment - baseline

        return max(-1.0, min(1.0, drift))

    def flag_anomaly(self, drift: float, threshold: float = 0.4) -> bool:
        """
        Flag drift as anomalous if magnitude exceeds threshold.
        
        Args:
            drift: Computed drift score.
            threshold: Drift magnitude threshold for anomaly flagging.

        Returns:
            True if |drift| >= threshold, False otherwise.
        """
        return abs(drift) >= threshold


# ============================================================================
# Test Suite
# ============================================================================


@pytest.fixture
def detector() -> MockDriftDetector:
    """Provide a freshly initialized MockDriftDetector instance."""
    return MockDriftDetector()


def test_sentiment_analysis_optimistic(detector: MockDriftDetector) -> None:
    """Assert optimistic text receives positive sentiment score."""
    score = detector.analyze_text(FIXTURE_TEXTS["optimistic_q1"])
    assert score > 0.3, f"Expected optimistic score > 0.3, got {score}"


def test_sentiment_analysis_pessimistic(detector: MockDriftDetector) -> None:
    """Assert pessimistic text receives negative sentiment score."""
    score = detector.analyze_text(FIXTURE_TEXTS["pessimistic_q3"])
    assert score < -0.2, f"Expected pessimistic score < -0.2, got {score}"


def test_sentiment_analysis_neutral(detector: MockDriftDetector) -> None:
    """Assert neutral text receives sentiment score near zero."""
    score = detector.analyze_text(FIXTURE_TEXTS["neutral_earnings"])
    assert -0.2 <= score <= 0.2, f"Expected neutral score near 0, got {score}"


def test_baseline_establishment(detector: MockDriftDetector) -> None:
    """Assert baseline can be set and retrieved for a ticker."""
    detector.set_baseline("ACME", FIXTURE_TEXTS["optimistic_q1"])
    baseline = detector.baseline_sentiment.get("ACME")
    assert baseline is not None, "Baseline not stored"
    assert baseline > 0.2, f"Expected positive baseline, got {baseline}"


def test_drift_positive_shift(detector: MockDriftDetector) -> None:
    """Assert drift correctly identifies shift from cautious to optimistic."""
    detector.set_baseline("TECH", FIXTURE_TEXTS["cautious_q2"])
    drift = detector.compute_drift("TECH", FIXTURE_TEXTS["optimistic_q1"])
    assert drift > 0.15, f"Expected positive drift > 0.15, got {drift}"


def test_drift_negative_shift(detector: MockDriftDetector) -> None:
    """Assert drift correctly identifies shift from optimistic to pessimistic."""
    detector.set_baseline("TECH", FIXTURE_TEXTS["optimistic_q1"])
    drift = detector.compute_drift("TECH", FIXTURE_TEXTS["pessimistic_q3"])
    assert drift < -0.3, f"Expected negative drift < -0.3, got {drift}"


def test_drift_recovery_signal(detector: MockDriftDetector) -> None:
    """Assert drift detects recovery tone after pessimistic baseline."""
    detector.set_baseline("CRASH", FIXTURE_TEXTS["pessimistic_q3"])
    drift = detector.compute_drift("CRASH", FIXTURE_TEXTS["recovery_q4"])
    assert drift > 0.25, f"Expected positive recovery drift > 0.25, got {drift}"


def test_drift_minimal_change(detector: MockDriftDetector) -> None:
    """Assert drift is near-zero when baseline and current are similar."""
    detector.set_baseline("STABLE", FIXTURE_TEXTS["neutral_earnings"])
    drift = detector.compute_drift("STABLE", FIXTURE_TEXTS["neutral_earnings"])
    assert abs(drift) < 0.1, f"
