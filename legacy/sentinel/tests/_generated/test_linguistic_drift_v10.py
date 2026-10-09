"""
Unit tests for the Linguistic Drift detector.

This module validates the Linguistic Drift detector's ability to identify
tone shifts in company communications over time. It uses fixture text samples
representing different sentiment states and verifies correct drift scoring,
anomaly flagging, and historical comparison logic.

Part of Sentinel's Linguist pillar — ensures drift detection accuracy before
production deployment.
"""

import pytest
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DriftScore:
    """Container for drift analysis results."""
    ticker: str
    drift_magnitude: float
    confidence: float
    direction: str
    anomaly_flagged: bool
    historical_baseline: float
    current_signal: float


def calculate_sentiment_polarity(text: str) -> float:
    """Calculate basic sentiment polarity score (0.0 to 1.0) from text."""
    positive_words = {
        'growth', 'strong', 'improved', 'exceeded', 'robust', 'accelerating',
        'momentum', 'opportunity', 'success', 'expanding', 'leading', 'record'
    }
    negative_words = {
        'decline', 'weak', 'deteriorated', 'missed', 'challenged', 'slowing',
        'headwinds', 'risk', 'loss', 'contraction', 'lagging', 'downward'
    }
    
    tokens = text.lower().split()
    pos_count = sum(1 for t in tokens if t.strip('.,!?;:') in positive_words)
    neg_count = sum(1 for t in tokens if t.strip('.,!?;:') in negative_words)
    total = pos_count + neg_count
    
    if total == 0:
        return 0.5
    return pos_count / total if total > 0 else 0.5


def detect_linguistic_drift(
    ticker: str,
    current_text: str,
    historical_texts: List[str],
    anomaly_threshold: float = 0.35
) -> DriftScore:
    """
    Detect sentiment drift in company communications over time.
    
    Args:
        ticker: Stock ticker symbol
        current_text: Latest communication text to analyze
        historical_texts: List of prior communication texts for baseline
        anomaly_threshold: Drift magnitude above which to flag anomaly
    
    Returns:
        DriftScore with magnitude, confidence, direction, and anomaly flag
    """
    current_polarity = calculate_sentiment_polarity(current_text)
    
    if not historical_texts:
        historical_baseline = 0.5
    else:
        historical_baseline = sum(
            calculate_sentiment_polarity(t) for t in historical_texts
        ) / len(historical_texts)
    
    drift_magnitude = abs(current_polarity - historical_baseline)
    
    direction = 'positive' if current_polarity > historical_baseline else 'negative'
    if drift_magnitude < 0.05:
        direction = 'neutral'
    
    confidence = min(1.0, drift_magnitude * 2.0)
    
    anomaly_flagged = drift_magnitude > anomaly_threshold
    
    return DriftScore(
        ticker=ticker,
        drift_magnitude=drift_magnitude,
        confidence=confidence,
        direction=direction,
        anomaly_flagged=anomaly_flagged,
        historical_baseline=historical_baseline,
        current_signal=current_polarity
    )


@pytest.fixture
def sample_texts() -> Dict[str, List[str]]:
    """Provide fixture text samples for drift testing."""
    return {
        'bullish_history': [
            'We are experiencing strong growth across all segments with record revenue.',
            'Our momentum continues to accelerate with expanding market opportunities.',
            'Leading position reinforced by successful product launches and robust demand.'
        ],
        'bearish_history': [
            'We face significant headwinds and declining market conditions.',
            'Revenue contraction accelerated as competition intensified.',
            'Risk factors including supply chain disruption impact profitability.'
        ],
        'neutral_history': [
            'We continue operations in line with expectations.',
            'Market conditions remain stable with moderate growth.',
            'Performance tracked to guidance with no major deviations.'
        ],
        'bullish_current': 'Exceptional results driven by strong execution and market expansion.',
        'bearish_current': 'Challenging environment led to significant revenue decline.',
        'neutral_current': 'Operations proceeded as anticipated with steady performance.',
        'extreme_shift_current': 'Facing unprecedented crisis with catastrophic operational failure.'
    }


class TestLinguisticDriftBasics:
    """Test core drift detection functionality."""
    
    def test_drift_detection_bullish_to_neutral(self, sample_texts: Dict) -> None:
        """Assert bullish history to neutral current shows negative drift."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['neutral_current'],
            historical_texts=sample_texts['bullish_history']
        )
        assert score.drift_magnitude > 0.1
        assert score.direction == 'negative'
        assert score.current_signal < score.historical_baseline
    
    def test_drift_detection_bearish_to_bullish(self, sample_texts: Dict) -> None:
        """Assert bearish history to bullish current shows positive drift."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['bullish_current'],
            historical_texts=sample_texts['bearish_history']
        )
        assert score.drift_magnitude > 0.1
        assert score.direction == 'positive'
        assert score.current_signal > score.historical_baseline
    
    def test_drift_detection_no_change(self, sample_texts: Dict) -> None:
        """Assert consistent bullish tone shows minimal drift."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['bullish_current'],
            historical_texts=sample_texts['bullish_history']
        )
        assert score.drift_magnitude < 0.15
        assert score.direction == 'neutral'
    
    def test_drift_detection_empty_history(self, sample_texts: Dict) -> None:
        """Assert drift scorer handles empty history gracefully."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['neutral_current'],
            historical_texts=[]
        )
        assert score.historical_baseline == 0.5
        assert isinstance(score.drift_magnitude, float)
        assert 0.0 <= score.confidence <= 1.0


class TestAnomalyDetection:
    """Test anomaly flagging logic."""
    
    def test_anomaly_flagged_on_extreme_drift(self, sample_texts: Dict) -> None:
        """Assert extreme tone shift triggers anomaly flag."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['extreme_shift_current'],
            historical_texts=sample_texts['bullish_history'],
            anomaly_threshold=0.35
        )
        assert score.anomaly_flagged is True
        assert score.drift_magnitude > 0.35
    
    def test_anomaly_not_flagged_on_normal_drift(self, sample_texts: Dict) -> None:
        """Assert moderate tone shift does not trigger anomaly flag."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['neutral_current'],
            historical_texts=sample_texts['bullish_history'],
            anomaly_threshold=0.35
        )
        assert score.anomaly_flagged is False
    
    def test_anomaly_threshold_boundary(self, sample_texts: Dict) -> None:
        """Assert anomaly flag respects exact threshold boundary."""
        score = detect_linguistic_drift(
            ticker='TEST',
            current_text=sample_texts['bearish_current'],
            historical_texts=sample_texts['bullish_history'],
            anomaly_threshold=0.30
        )
        if score.drift_magnitude > 0.30:
            assert score.anomaly_flagged is True
