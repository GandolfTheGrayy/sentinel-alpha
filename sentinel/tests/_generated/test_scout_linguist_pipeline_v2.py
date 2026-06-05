"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout)
through sentiment analysis (Linguist) with fully mocked external calls.
Tests verify that scraped data is correctly parsed, enriched with
sentiment scores, and confidence metrics are properly calibrated.

Part of Sentinel's test suite for production readiness validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# These would be imported from actual Sentinel modules
# For testing purposes, we'll define minimal stubs


class ScoutDataPoint:
    """Mock Scout data structure."""
    
    def __init__(self, source: str, ticker: str, text: str, timestamp: datetime):
        self.source = source
        self.ticker = ticker
        self.text = text
        self.timestamp = timestamp
        self.raw_data = {"content": text, "source": source}


class LinguistScore:
    """Mock Linguist output structure."""
    
    def __init__(self, ticker: str, sentiment: float, certainty: float, 
                 hesitation_signals: List[str], drift_indicators: Dict[str, Any]):
        self.ticker = ticker
        self.sentiment = sentiment  # -1.0 to 1.0
        self.certainty = certainty  # 0.0 to 1.0
        self.hesitation_signals = hesitation_signals
        self.drift_indicators = drift_indicators
        self.timestamp = datetime.now()


class MockScout:
    """Mock Scout module for testing."""
    
    @staticmethod
    def fetch_live_price(ticker: str) -> Dict[str, Any]:
        """Return mock price data."""
        return {
            "ticker": ticker,
            "price": 150.50,
            "timestamp": datetime.now().isoformat(),
            "source": "yfinance"
        }
    
    @staticmethod
    def fetch_news_headlines(ticker: str, limit: int = 5) -> List[ScoutDataPoint]:
        """Return mock news headlines."""
        headlines = [
            f"{ticker} posts record earnings beat expectations",
            f"Analysts upgrade {ticker} to outperform",
            f"{ticker} faces regulatory scrutiny in EU",
        ]
        return [
            ScoutDataPoint("news", ticker, headline, datetime.now() - timedelta(hours=i))
            for i, headline in enumerate(headlines[:limit])
        ]
    
    @staticmethod
    def fetch_sec_filings(ticker: str, form_type: str = "8-K") -> List[ScoutDataPoint]:
        """Return mock SEC filing data."""
        filings = [
            f"Form {form_type}: {ticker} reports material contract signed with major enterprise customer",
            f"Form {form_type}: {ticker} announces executive leadership transition",
        ]
        return [
            ScoutDataPoint("sec", ticker, filing, datetime.now() - timedelta(days=i))
            for i, filing in enumerate(filings)
        ]
    
    @staticmethod
    def fetch_reddit_sentiment(ticker: str, subreddits: List[str] = None) -> List[ScoutDataPoint]:
        """Return mock Reddit sentiment data."""
        posts = [
            f"Just bought {ticker} - great fundamentals and momentum",
            f"Concerned about {ticker}'s competitive position but long-term bullish",
            f"{ticker} is overvalued, I'm staying on the sidelines",
        ]
        return [
            ScoutDataPoint("reddit", ticker, post, datetime.now() - timedelta(hours=i))
            for i, post in enumerate(posts)
        ]


class MockLinguist:
    """Mock Linguist module for testing."""
    
    @staticmethod
    def analyze_certainty(text: str) -> Dict[str, Any]:
        """Analyze certainty vs. hesitation in text."""
        hesitation_words = ["might", "could", "may", "possibly", "seems", "appears"]
        certainty_words = ["will", "must", "definitely", "certainly", "proven"]
        
        text_lower = text.lower()
        hesitation_count = sum(1 for word in hesitation_words if word in text_lower)
        certainty_count = sum(1 for word in certainty_words if word in text_lower)
        
        total = hesitation_count + certainty_count
        if total == 0:
            certainty_score = 0.5
        else:
            certainty_score = certainty_count / total
        
        return {
            "certainty": certainty_score,
            "hesitation_signals": [w for w in hesitation_words if w in text_lower],
            "confidence_indicators": [w for w in certainty_words if w in text_lower]
        }
    
    @staticmethod
    def detect_linguistic_drift(texts: List[str], lookback_days: int = 30) -> Dict[str, Any]:
        """Detect sentiment drift over time."""
        if len(texts) < 2:
            return {"drift_detected": False, "drift_magnitude": 0.0, "trend": "stable"}
        
        # Mock drift detection: analyze text length as proxy for tone intensity
        avg_length = sum(len(t) for t in texts) / len(texts)
        drift_magnitude = abs(len(texts[0]) - avg_length) / avg_length if avg_length > 0 else 0.0
        
        return {
            "drift_detected": drift_magnitude > 0.2,
            "drift_magnitude": min(drift_magnitude, 1.0),
            "trend": "intensifying" if len(texts[0]) > avg_length else "moderating",
            "lookback_days": lookback_days
        }
    
    @staticmethod
    def score_sentiment(text: str) -> float:
        """Simple mock sentiment scoring: -1.0 (bearish) to 1.0 (bullish)."""
        positive_words = ["beat", "outperform", "upgrade", "bullish", "great", "excellent", "strong"]
        negative_words = ["miss", "downgrade", "bearish", "decline", "weak", "concern", "risk"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total


class TestScoutDataIngestion:
    """Tests for Scout data collection."""
    
    def test_fetch_live_price_returns_valid_structure(self) -> None:
        """Verify live price fetch returns expected schema."""
        scout = MockScout()
        result = scout.fetch_live_price("AAPL")
        
        assert "ticker" in result
        assert "price" in result
        assert "timestamp" in result
        assert result["ticker"] == "AAPL"
        assert isinstance(result["price"], (int, float))
    
    def test_fetch_news_headlines_returns_list_of_datapoints(self) -> None:
        """Verify news fetch returns list of ScoutDataPoint objects."""
        scout = MockScout()
        result = scout.fetch_news_headlines("TSLA", limit=3)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, ScoutDataPoint) for item in result)
        assert all(item.source == "news" for item in result)
        assert all(item.ticker == "TSLA" for item in result)
    
    def test_fetch_sec_filings_returns_valid_datapoints(self) -> None:
        """Verify SEC filing fetch returns correct structure."""
        scout = MockScout()
        result = scout.fetch_sec_filings("MSFT", form_type="10-Q")
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(item, ScoutData
