"""
Unit tests for the Linguistic Drift detector.

This module validates that the Linguistic Drift detector correctly identifies
tone shifts in company-specific sentiment signals (news, SEC filings, social media)
over time. It uses fixture text samples representing different sentiment states
and asserts that drift scores are computed accurately.

Part of Sentinel's Linguist pillar validation suite.
"""

import pytest
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DriftSample:
    """Represents a single text sample with metadata for drift testing."""
    ticker: str
    timestamp: str
    text: str
    source: str  # "news", "sec", "social"


class LinguisticDriftDetector:
    """
    Detects tone and sentiment drift in company-specific corpora.
    
    Computes a drift score (0.0 to 1.0) representing the magnitude of sentiment
    shift from a reference baseline to a recent window.
    """
    
    def __init__(self) -> None:
        """Initialize the drift detector with empty state."""
        self.baseline_scores: Dict[str, float] = {}
        self.recent_scores: Dict[str, float] = {}
    
    def compute_drift(self, ticker: str, baseline_texts: List[str], recent_texts: List[str]) -> float:
        """
        Compute drift score for a ticker given baseline and recent sentiment.
        
        Returns a float in [0.0, 1.0] where 0.0 = no drift, 1.0 = maximum drift.
        """
        baseline_avg = self._avg_sentiment_score(baseline_texts)
        recent_avg = self._avg_sentiment_score(recent_texts)
        drift = abs(recent_avg - baseline_avg)
        self.baseline_scores[ticker] = baseline_avg
        self.recent_scores[ticker] = recent_avg
        return min(drift, 1.0)
    
    def _avg_sentiment_score(self, texts: List[str]) -> float:
        """
        Compute average sentiment score across a list of texts.
        
        Simple heuristic: count positive/negative keywords, return normalized score.
        """
        if not texts:
            return 0.5
        
        positive_words = {"growth", "strong", "profit", "gain", "bullish", "surge", "beat", "excellent"}
        negative_words = {"loss", "decline", "weak", "miss", "bearish", "plunge", "risk", "concern"}
        
        scores = []
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            total = pos_count + neg_count
            if total == 0:
                scores.append(0.5)
            else:
                scores.append(pos_count / total)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def flag_anomaly(self, drift_score: float, threshold: float = 0.4) -> bool:
        """
        Check if drift score exceeds anomaly threshold.
        
        Returns True if drift_score >= threshold, indicating significant tone shift.
        """
        return drift_score >= threshold


@pytest.fixture
def detector() -> LinguisticDriftDetector:
    """Provides a fresh LinguisticDriftDetector instance for each test."""
    return LinguisticDriftDetector()


@pytest.fixture
def baseline_samples() -> List[str]:
    """Fixture: neutral-to-positive baseline sentiment texts."""
    return [
        "Apple reports strong quarterly earnings with steady revenue growth.",
        "AAPL stock shows solid fundamentals and excellent market positioning.",
        "Investors remain bullish on Apple's long-term profit trajectory.",
    ]


@pytest.fixture
def recent_positive_samples() -> List[str]:
    """Fixture: recent texts with amplified positive sentiment."""
    return [
        "Apple beats expectations with record-breaking profit surge.",
        "AAPL soars on exceptional growth and bullish analyst upgrades.",
        "Market sentiment shifts massively positive for Apple stock.",
    ]


@pytest.fixture
def recent_negative_samples() -> List[str]:
    """Fixture: recent texts with negative sentiment shift."""
    return [
        "Apple faces significant supply chain losses and revenue decline.",
        "Concerns mount over weak sales and upcoming regulatory risks.",
        "AAPL plunges as bearish forecasts dominate analyst coverage.",
    ]


@pytest.fixture
def recent_neutral_samples() -> List[str]:
    """Fixture: recent texts with neutral sentiment (minimal drift)."""
    return [
        "Apple releases new product line with mixed market reception.",
        "AAPL trades flat as earnings meet modest expectations.",
        "Investors await guidance on Apple's strategic direction.",
    ]


class TestLinguisticDriftBasic:
    """Tests for core drift detection functionality."""
    
    def test_detector_initialization(self, detector: LinguisticDriftDetector) -> None:
        """Detector initializes with empty baseline and recent score dicts."""
        assert detector.baseline_scores == {}
        assert detector.recent_scores == {}
    
    def test_compute_drift_no_drift(
        self, detector: LinguisticDriftDetector, baseline_samples: List[str]
    ) -> None:
        """Drift is minimal when baseline and recent texts have similar sentiment."""
        drift = detector.compute_drift("AAPL", baseline_samples, baseline_samples)
        assert drift < 0.15, f"Expected low drift, got {drift}"
    
    def test_compute_drift_positive_shift(
        self, detector: LinguisticDriftDetector,
        baseline_samples: List[str], recent_positive_samples: List[str]
    ) -> None:
        """Drift is significant when sentiment shifts from baseline to positive."""
        drift = detector.compute_drift("AAPL", baseline_samples, recent_positive_samples)
        assert drift > 0.2, f"Expected moderate-to-high positive drift, got {drift}"
    
    def test_compute_drift_negative_shift(
        self, detector: LinguisticDriftDetector,
        baseline_samples: List[str], recent_negative_samples: List[str]
    ) -> None:
        """Drift is significant when sentiment shifts from baseline to negative."""
        drift = detector.compute_drift("AAPL", baseline_samples, recent_negative_samples)
        assert drift > 0.2, f"Expected moderate-to-high negative drift, got {drift}"
    
    def test_compute_drift_neutral_shift(
        self, detector: LinguisticDriftDetector,
        baseline_samples: List[str], recent_neutral_samples: List[str]
    ) -> None:
        """Drift is low when sentiment shifts to neutral (small departure from baseline)."""
        drift = detector.compute_drift("AAPL", baseline_samples, recent_neutral_samples)
        assert drift < 0.25, f"Expected small drift for neutral shift, got {drift}"


class TestAnomalyFlagging:
    """Tests for drift-based anomaly detection."""
    
    def test_flag_anomaly_below_threshold(self, detector: LinguisticDriftDetector) -> None:
        """Drift below 0.4 threshold is not flagged as anomalous."""
        assert not detector.flag_anomaly(0.2)
        assert not detector.flag_anomaly(0.39)
    
    def test_flag_anomaly_at_threshold(self, detector: LinguisticDriftDetector) -> None:
        """Drift at or above 0.4 threshold is flagged as anomalous."""
        assert detector.flag_anomaly(0.4)
        assert detector.flag_anomaly(0.5)
        assert detector.flag_anomaly(1.0)
    
    def test_flag_anomaly_custom_threshold(self, detector: LinguisticDriftDetector) -> None:
        """Custom threshold parameter is respected."""
        assert not detector.flag_anomaly(0.5, threshold=0.6)
        assert detector.
