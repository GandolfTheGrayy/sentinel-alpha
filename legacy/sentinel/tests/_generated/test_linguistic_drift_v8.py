"""
Unit test suite for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to score
tone and sentiment shifts in company communications over time. It uses
fixture text samples representing different time periods and ensures
that drift scoring correctly identifies meaningful shifts in language
patterns, certainty markers, and regulatory tone.

Part of Sentinel's linguist pillar validation suite.
"""

import pytest
from typing import Dict, List, Tuple


class LinguisticDriftFixtures:
    """Fixture data for linguistic drift testing."""

    TECH_COMPANY_2023_Q1 = """
    We are excited to announce record-breaking growth in our cloud division.
    Our innovative products have captured unprecedented market share. We are
    confident in our ability to scale rapidly and maintain our competitive
    advantage. Customer satisfaction metrics are at all-time highs.
    """

    TECH_COMPANY_2024_Q1 = """
    We have experienced modest growth in core segments. Market conditions remain
    challenging and we are monitoring competitive pressures carefully. While we
    see opportunities, we must balance expansion with prudent cost management.
    Customer retention has been stable but growth rates have moderated.
    """

    PHARMA_COMPANY_2023_Q2 = """
    Our clinical trial results demonstrate clear efficacy. We expect FDA approval
    by Q4 2023. The blockbuster potential of this drug is substantial and will
    transform our revenue profile significantly. We are highly optimistic about
    market penetration.
    """

    PHARMA_COMPANY_2024_Q2 = """
    Clinical trial results were mixed. We are working with regulatory agencies
    on outstanding questions. Approval timelines may extend beyond initial
    projections. We will provide updates as the review process advances.
    Market adoption scenarios range broadly depending on final labeling.
    """

    FINANCE_COMPANY_2023_Q3 = """
    Credit quality remains strong with minimal delinquencies. Our risk management
    framework is robust. We maintain confidence in loan portfolio performance.
    Interest rate environment is favorable for margins.
    """

    FINANCE_COMPANY_2024_Q3 = """
    We are observing early signs of stress in certain loan segments. Delinquency
    rates have ticked up modestly. We have increased loan loss provisions as a
    precaution. The interest rate outlook introduces uncertainty into margin
    guidance.
    """


@pytest.fixture
def drift_fixtures() -> Dict[str, Tuple[str, str]]:
    """Provide paired before/after text samples for drift detection testing."""
    return {
        "tech_optimism_to_caution": (
            LinguisticDriftFixtures.TECH_COMPANY_2023_Q1,
            LinguisticDriftFixtures.TECH_COMPANY_2024_Q1,
        ),
        "pharma_confidence_to_uncertainty": (
            LinguisticDriftFixtures.PHARMA_COMPANY_2023_Q2,
            LinguisticDriftFixtures.PHARMA_COMPANY_2024_Q2,
        ),
        "finance_stability_to_concern": (
            LinguisticDriftFixtures.FINANCE_COMPANY_2023_Q3,
            LinguisticDriftFixtures.FINANCE_COMPANY_2024_Q3,
        ),
    }


def extract_certainty_markers(text: str) -> Dict[str, int]:
    """Extract counts of certainty and hesitation markers from text."""
    certainty_words = [
        "confident",
        "confident",
        "certain",
        "clear",
        "strong",
        "robust",
        "expect",
        "will",
        "excellent",
        "outstanding",
    ]
    hesitation_words = [
        "may",
        "might",
        "could",
        "uncertain",
        "challenge",
        "risk",
        "cautious",
        "modest",
        "monitor",
        "observing",
        "monitoring",
        "extended",
        "moderated",
    ]

    text_lower = text.lower()
    certainty_count = sum(text_lower.count(word) for word in certainty_words)
    hesitation_count = sum(text_lower.count(word) for word in hesitation_words)

    return {
        "certainty": certainty_count,
        "hesitation": hesitation_count,
    }


def calculate_drift_score(before_text: str, after_text: str) -> float:
    """
    Calculate linguistic drift score between two text samples.

    Returns a float between -1.0 (shift to extreme caution) and 1.0 (shift to
    extreme confidence). 0.0 indicates no meaningful drift.
    """
    before_markers = extract_certainty_markers(before_text)
    after_markers = extract_certainty_markers(after_text)

    before_ratio = (
        (before_markers["certainty"] - before_markers["hesitation"])
        / (before_markers["certainty"] + before_markers["hesitation"] + 1)
    )
    after_ratio = (
        (after_markers["certainty"] - after_markers["hesitation"])
        / (after_markers["certainty"] + after_markers["hesitation"] + 1)
    )

    drift = after_ratio - before_ratio
    return max(-1.0, min(1.0, drift))


def detect_tone_shift_direction(drift_score: float) -> str:
    """Classify tone shift direction from drift score."""
    if drift_score > 0.15:
        return "optimism_increase"
    elif drift_score < -0.15:
        return "pessimism_increase"
    else:
        return "neutral"


class TestLinguisticDriftDetection:
    """Test suite for linguistic drift scoring and tone shift detection."""

    def test_drift_score_identifies_optimism_decrease(
        self, drift_fixtures: Dict[str, Tuple[str, str]]
    ) -> None:
        """Verify drift detector identifies shift from optimism to caution."""
        before, after = drift_fixtures["tech_optimism_to_caution"]
        drift_score = calculate_drift_score(before, after)

        assert drift_score < -0.1, "Should detect negative drift (caution increase)"
        assert isinstance(drift_score, float)

    def test_drift_score_identifies_confidence_decrease(
        self, drift_fixtures: Dict[str, Tuple[str, str]]
    ) -> None:
        """Verify drift detector identifies shift from confidence to uncertainty."""
        before, after = drift_fixtures["pharma_confidence_to_uncertainty"]
        drift_score = calculate_drift_score(before, after)

        assert drift_score < -0.1, "Should detect negative drift in pharma case"

    def test_drift_score_identifies_risk_concern_increase(
        self, drift_fixtures: Dict[str, Tuple[str, str]]
    ) -> None:
        """Verify drift detector identifies shift from stability to concern."""
        before, after = drift_fixtures["finance_stability_to_concern"]
        drift_score = calculate_drift_score(before, after)

        assert drift_score < -0.1, "Should detect negative drift in finance case"

    def test_tone_shift_direction_pessimism_flag(
        self, drift_fixtures: Dict[str, Tuple[str, str]]
    ) -> None:
        """Verify tone shift classifier correctly flags pessimism increase."""
        before, after = drift_fixtures["tech_optimism_to_caution"]
        drift_score = calculate_drift_score(before, after)
        direction = detect_tone_shift_direction(drift_score)

        assert direction == "pessimism_increase"

    def test_tone_shift_direction_neutral_near_zero(self) -> None:
        """Verify neutral classification for near-zero drift."""
        similar_text_a = "We are doing well and expect continued growth."
        similar_text_b = "We are doing well and expect continued growth."
        drift_score = calculate_drift_score(similar_text_a, similar_text_b)
        direction = detect_tone_shift_direction(drift_score)

        assert direction == "neutral"

    def test_certainty_marker_extraction_counts_correctly(self) -> None:
        """Verify certainty marker
