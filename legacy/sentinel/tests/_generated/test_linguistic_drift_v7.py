"""
Unit tests for the Linguistic Drift detector.

This module validates the drift scoring logic that measures tone/sentiment
shifts in company communications (SEC filings, news, social media) over time.
The drift detector is part of the Linguist pillar and feeds anomaly signals
to the Judge for prediction refinement.

Tests use fixture text spanning multiple time periods to assert that drift
scores correctly identify:
  - Increasing negativity (rising drift score)
  - Stable sentiment (low drift)
  - Sudden tone reversals (spike detection)
  - Edge cases (empty corpus, single document)
"""

import pytest
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TextSample:
    """Single timestamped text sample for drift analysis."""
    timestamp: datetime
    text: str
    source: str


class LinguisticDriftDetector:
    """
    Detects sentiment/tone drift in company communications over time.
    
    Compares linguistic patterns across time windows to flag anomalies
    that may precede market moves.
    """

    def __init__(self, window_days: int = 30):
        """Initialize drift detector with rolling window size."""
        self.window_days = window_days
        # Simple word lists for demo; real system uses embedding-based comparison
        self.negative_words = {
            "decline", "loss", "risk", "challenge", "uncertainty", "weak",
            "difficult", "pressure", "concern", "warning", "restructure",
            "impairment", "liability", "lawsuit", "default", "downside"
        }
        self.positive_words = {
            "growth", "opportunity", "strength", "success", "expand",
            "profit", "revenue", "robust", "confidence", "momentum",
            "innovation", "breakthrough", "outperform", "upside"
        }

    def _sentiment_score(self, text: str) -> float:
        """
        Compute raw sentiment score for a single document.
        
        Returns float in [-1.0, 1.0] where -1 is maximally negative,
        0 is neutral, +1 is maximally positive.
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        neg_count = sum(1 for w in words if w in self.negative_words)
        pos_count = sum(1 for w in words if w in self.positive_words)
        
        total_sentiment = pos_count - neg_count
        score = total_sentiment / len(words)
        return max(-1.0, min(1.0, score))

    def compute_drift(self, samples: List[TextSample]) -> Dict[str, float]:
        """
        Compute drift metrics across time-ordered samples.
        
        Args:
            samples: List of TextSample objects, ideally sorted by timestamp.
        
        Returns:
            Dict with keys:
              - "drift_score": 0–1, magnitude of change over window.
              - "drift_direction": -1 (getting negative), 0 (stable), +1 (positive).
              - "volatility": std dev of sentiment across window.
              - "recent_trend": slope of last N documents (positive/negative).
        """
        if not samples:
            return {
                "drift_score": 0.0,
                "drift_direction": 0,
                "volatility": 0.0,
                "recent_trend": 0.0
            }
        
        # Sort by timestamp to ensure chronological order
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)
        
        # Compute sentiment scores for each sample
        sentiments = [self._sentiment_score(s.text) for s in sorted_samples]
        
        if len(sentiments) == 1:
            return {
                "drift_score": 0.0,
                "drift_direction": 0,
                "volatility": 0.0,
                "recent_trend": 0.0
            }
        
        # Drift score: absolute change from first to last
        drift_magnitude = abs(sentiments[-1] - sentiments[0])
        
        # Direction: sign of change
        drift_direction = 0
        if sentiments[-1] < sentiments[0] - 0.05:
            drift_direction = -1  # Getting more negative
        elif sentiments[-1] > sentiments[0] + 0.05:
            drift_direction = 1   # Getting more positive
        
        # Volatility: std dev (approximation without numpy)
        mean_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiments) / len(sentiments)
        volatility = variance ** 0.5
        
        # Recent trend: slope of last 3 docs (or fewer if unavailable)
        recent_count = min(3, len(sentiments))
        if recent_count >= 2:
            recent_slope = (sentiments[-1] - sentiments[-recent_count]) / (recent_count - 1)
        else:
            recent_slope = 0.0
        
        return {
            "drift_score": drift_magnitude,
            "drift_direction": drift_direction,
            "volatility": volatility,
            "recent_trend": recent_slope
        }

    def detect_spike(self, samples: List[TextSample], threshold: float = 0.3) -> bool:
        """
        Detect sudden sentiment reversal (spike) in the corpus.
        
        Returns True if any consecutive pair of samples shows
        sentiment change > threshold.
        """
        if len(samples) < 2:
            return False
        
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)
        sentiments = [self._sentiment_score(s.text) for s in sorted_samples]
        
        for i in range(1, len(sentiments)):
            if abs(sentiments[i] - sentiments[i - 1]) > threshold:
                return True
        
        return False


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stable_samples() -> List[TextSample]:
    """Fixture: consistent positive sentiment over time."""
    base_date = datetime(2024, 1, 1)
    return [
        TextSample(
            timestamp=base_date + timedelta(days=0),
            text="We are pleased to report strong revenue growth and robust profitability.",
            source="earnings_call"
        ),
        TextSample(
            timestamp=base_date + timedelta(days=10),
            text="Our business momentum continues with excellent performance across all segments.",
            source="news"
        ),
        TextSample(
            timestamp=base_date + timedelta(days=20),
            text="Innovation and expansion opportunities remain strong as we move forward.",
            source="sec_8k"
        ),
    ]


@pytest.fixture
def degrading_samples() -> List[TextSample]:
    """Fixture: sentiment deteriorates over time."""
    base_date = datetime(2024, 1, 1)
    return [
        TextSample(
            timestamp=base_date + timedelta(days=0),
            text="Excellent growth prospects and strong market position.",
            source="earnings_call"
        ),
        TextSample(
            timestamp=base_date + timedelta(days=10),
            text="We face some challenges in the current environment.",
            source="news"
        ),
        TextSample(
            timestamp=base_date + timedelta(days=20),
            text="Significant risks and uncertainties continue to pressure our operations.",
            source="sec_10q"
        ),
        TextSample(
            timestamp=base_date + timedelta(days=30),
            text="Material decline in revenue and concerning liability developments.",
            source="sec_8k"
        ),
    ]


@pytest.fixture
def spiking_samples() -> List[TextSample]:
    """Fixture: sudden reversal from positive to negative."""
    base_date = datetime(2024, 1, 1)
    return [
        TextSample(
            timestamp=base_date + timedelta(days=0),
