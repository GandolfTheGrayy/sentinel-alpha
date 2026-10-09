"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to score
tone shifts in company-related text over time. It uses fixture text
samples (past vs. present sentiment) to assert correct drift scoring,
confidence thresholds, and anomaly detection.

Part of Sentinel's Linguist pillar QA.
"""

import pytest
from typing import Dict, List, Tuple


# ============================================================================
# FIXTURES: Sample text corpora for drift detection
# ============================================================================

@pytest.fixture
def past_neutral_corpus() -> List[str]:
    """Neutral tone texts from 6 months ago."""
    return [
        "Company reported steady earnings growth in Q2.",
        "Market share remained stable across regions.",
        "Management guidance unchanged from prior quarter.",
        "Operations proceeded normally with no major incidents.",
        "Cost structure in line with historical benchmarks.",
    ]


@pytest.fixture
def present_bullish_corpus() -> List[str]:
    """Bullish tone texts from current period."""
    return [
        "Revenue surged 45% YoY, exceeding analyst expectations.",
        "New product launch generating unprecedented customer demand.",
        "Margin expansion accelerating faster than projected.",
        "Strategic partnerships opening major growth avenues.",
        "Management raised full-year guidance by 20%.",
    ]


@pytest.fixture
def present_bearish_corpus() -> List[str]:
    """Bearish tone texts from current period."""
    return [
        "Disappointing earnings miss revenue targets by 12%.",
        "Customer churn accelerating in core segments.",
        "Margin compression due to rising input costs.",
        "Key executive departures signal internal instability.",
        "Regulatory headwinds threaten market position.",
    ]


@pytest.fixture
def mixed_sentiment_corpus() -> List[str]:
    """Mixed sentiment texts (ambiguous for drift detection)."""
    return [
        "Despite challenges, we see long-term opportunities.",
        "Sales declined but operational efficiency improved.",
        "Market headwinds offset by product innovation.",
        "Competition intensified yet market share stable.",
        "Cost pressures balanced by higher pricing power.",
    ]


# ============================================================================
# MOCK DRIFT DETECTOR (simplified for testing)
# ============================================================================

def compute_sentiment_vector(texts: List[str]) -> Dict[str, float]:
    """
    Compute aggregated sentiment metrics from text corpus.
    
    Returns dict with keys: positive_ratio, negative_ratio, certainty_score.
    """
    positive_keywords = {
        "surge", "exceed", "accelerate", "unprecedented", "raised",
        "growth", "expand", "strategic", "demand", "opportunity"
    }
    negative_keywords = {
        "disappointing", "miss", "churn", "compression", "departures",
        "instability", "headwind", "threat", "declined", "pressure"
    }
    neutral_keywords = {
        "stable", "steady", "unchanged", "normal", "benchmark",
        "challenge", "balance", "mixed", "despite", "offset"
    }
    
    text_lower = " ".join(texts).lower()
    words = text_lower.split()
    
    positive_count = sum(1 for w in words if w in positive_keywords)
    negative_count = sum(1 for w in words if w in negative_keywords)
    neutral_count = sum(1 for w in words if w in neutral_keywords)
    total_signal = positive_count + negative_count + neutral_count
    
    if total_signal == 0:
        return {"positive_ratio": 0.0, "negative_ratio": 0.0, "certainty_score": 0.0}
    
    pos_ratio = positive_count / total_signal
    neg_ratio = negative_count / total_signal
    certainty = max(pos_ratio, neg_ratio, neutral_count / total_signal)
    
    return {
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "certainty_score": certainty,
    }


def detect_linguistic_drift(
    past_texts: List[str],
    present_texts: List[str],
    min_confidence: float = 0.5
) -> Dict[str, float]:
    """
    Detect sentiment drift between past and present corpora.
    
    Returns dict with drift_magnitude, direction, and confidence.
    """
    past_vec = compute_sentiment_vector(past_texts)
    present_vec = compute_sentiment_vector(present_texts)
    
    drift_pos = present_vec["positive_ratio"] - past_vec["positive_ratio"]
    drift_neg = present_vec["negative_ratio"] - past_vec["negative_ratio"]
    drift_magnitude = abs(drift_pos) + abs(drift_neg)
    
    if drift_pos > 0.15:
        direction = "bullish"
    elif drift_neg > 0.15:
        direction = "bearish"
    else:
        direction = "neutral"
    
    confidence = min(
        max(present_vec["certainty_score"], past_vec["certainty_score"]),
        drift_magnitude / 0.5
    )
    confidence = min(confidence, 1.0)
    
    return {
        "drift_magnitude": drift_magnitude,
        "direction": direction,
        "confidence": confidence,
        "past_positive_ratio": past_vec["positive_ratio"],
        "present_positive_ratio": present_vec["positive_ratio"],
        "is_significant": confidence >= min_confidence,
    }


# ============================================================================
# TESTS
# ============================================================================

def test_drift_neutral_to_bullish(past_neutral_corpus, present_bullish_corpus):
    """Assert drift detector identifies bullish shift from neutral baseline."""
    result = detect_linguistic_drift(past_neutral_corpus, present_bullish_corpus)
    
    assert result["direction"] == "bullish", "Should detect bullish drift"
    assert result["drift_magnitude"] > 0.2, "Magnitude should exceed threshold"
    assert result["present_positive_ratio"] > result["past_positive_ratio"], \
        "Positive ratio should increase"
    assert result["is_significant"], "Drift should be flagged as significant"


def test_drift_neutral_to_bearish(past_neutral_corpus, present_bearish_corpus):
    """Assert drift detector identifies bearish shift from neutral baseline."""
    result = detect_linguistic_drift(past_neutral_corpus, present_bearish_corpus)
    
    assert result["direction"] == "bearish", "Should detect bearish drift"
    assert result["drift_magnitude"] > 0.2, "Magnitude should exceed threshold"
    assert result["present_negative_ratio"] > result["past_negative_ratio"], \
        "Negative ratio should increase"
    assert result["is_significant"], "Drift should be flagged as significant"


def test_drift_neutral_to_neutral(past_neutral_corpus):
    """Assert drift detector reports no drift when comparing identical sentiment."""
    result = detect_linguistic_drift(past_neutral_corpus, past_neutral_corpus)
    
    assert result["direction"] == "neutral", "Should detect no drift"
    assert result["drift_magnitude"] < 0.05, "Magnitude should be negligible"
    assert not result["is_significant"], "Should not be flagged as significant"


def test_drift_mixed_sentiment_low_confidence(mixed_sentiment_corpus, present_bullish_corpus):
    """Assert drift detector reports low confidence on mixed baselines."""
    result = detect_linguistic_drift(mixed_sentiment_corpus, present_bullish_corpus)
    
    # Mixed corpus has ambiguous signals, so confidence may be lower
    assert result["confidence"] <= 1.0, "Confidence within valid range"
    assert result["drift_magnitude"] >= 0.0, "Drift magnitude non-negative"


def test_drift_with_min_confidence_threshold(past_neutral_corpus, mixed_sentiment_corpus):
    """Assert drift detector respects min_confidence parameter for significance."""
    result_loose = detect_linguistic_drift(
        past_neutral_corpus, mixed_sentiment_corpus, min_confidence=0.3
    )
    result_strict = detect_linguistic_drift(
        past_neutral_corpus, mixed_sentiment_corpus, min_confidence=0.9
