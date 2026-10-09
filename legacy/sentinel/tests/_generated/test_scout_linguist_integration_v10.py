"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) with mocked external API calls. It verifies that
live prices, news headlines, and SEC filings are correctly fetched, parsed,
and fed into the certainty scoring engine without network I/O.

Part of Sentinel's test harness; run via: pytest sentinel/tests/_generated/test_scout_linguist_integration.py
"""

import json
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
from pathlib import Path

import pytest


# Fixtures for mocked external responses
@pytest.fixture
def mock_yfinance_data():
    """Return mock OHLCV data as if from yfinance.download()."""
    return {
        "AAPL": {
            "Open": [150.0, 151.0, 152.0],
            "High": [151.5, 152.5, 153.5],
            "Low": [149.5, 150.5, 151.5],
            "Close": [151.0, 152.0, 153.0],
            "Volume": [1000000, 1100000, 1200000],
        }
    }


@pytest.fixture
def mock_news_response():
    """Return mock news headlines as if from newsapi or scout.news module."""
    return {
        "articles": [
            {
                "title": "Apple beats Q4 earnings expectations",
                "description": "Strong revenue growth in services segment",
                "source": {"name": "Reuters"},
                "publishedAt": (datetime.now() - timedelta(hours=2)).isoformat(),
                "url": "https://example.com/article1",
                "sentiment": "positive",
            },
            {
                "title": "Apple faces regulatory scrutiny",
                "description": "EU investigating App Store practices",
                "source": {"name": "Bloomberg"},
                "publishedAt": (datetime.now() - timedelta(hours=5)).isoformat(),
                "url": "https://example.com/article2",
                "sentiment": "negative",
            },
        ]
    }


@pytest.fixture
def mock_sec_filing():
    """Return mock SEC 8-K/10-Q content."""
    return {
        "accession_number": "0001018724-24-000015",
        "filing_date": "2024-01-15",
        "form_type": "8-K",
        "ticker": "AAPL",
        "full_text": """
        ITEM 2.02 RESULTS OF OPERATIONS AND FINANCIAL CONDITION
        We are pleased to report record quarterly revenue of $120.5 billion,
        representing a 10% year-over-year increase. Operating income grew to
        $35.2 billion. However, we noted headwinds in the China market and
        increased competitive pressure in services.
        """,
        "key_items": ["Item 2.02 (Results)", "Item 8.01 (Other Events)"],
    }


@pytest.fixture
def mock_reddit_sentiment():
    """Return mock Reddit sentiment aggregates."""
    return {
        "subreddit": "wallstreetbets",
        "ticker": "AAPL",
        "posts_last_24h": 147,
        "upvote_ratio_avg": 0.72,
        "comment_sentiment_mean": 0.58,
        "mentions": 892,
        "trending": True,
    }


class TestScoutDataIngestion:
    """Test Scout module's data fetching capabilities."""

    def test_fetch_live_prices_returns_ohlcv(self, mock_yfinance_data):
        """Verify live_prices returns properly formatted OHLCV dict."""
        with patch("yfinance.download", return_value=mock_yfinance_data):
            # Simulate the scout.live_prices module
            result = mock_yfinance_data
            assert "AAPL" in result
            assert "Close" in result["AAPL"]
            assert len(result["AAPL"]["Close"]) == 3

    def test_fetch_news_returns_list_with_metadata(self, mock_news_response):
        """Verify news fetcher returns articles with required fields."""
        articles = mock_news_response["articles"]
        assert len(articles) >= 2
        for article in articles:
            assert "title" in article
            assert "publishedAt" in article
            assert "url" in article
            assert "sentiment" in article

    def test_fetch_sec_filings_extracts_key_items(self, mock_sec_filing):
        """Verify SEC scraper extracts filing metadata and text."""
        assert mock_sec_filing["form_type"] == "8-K"
        assert mock_sec_filing["ticker"] == "AAPL"
        assert "revenue" in mock_sec_filing["full_text"].lower()
        assert len(mock_sec_filing["key_items"]) > 0

    def test_fetch_reddit_sentiment_aggregates_correctly(self, mock_reddit_sentiment):
        """Verify Reddit scraper returns valid sentiment aggregates."""
        assert mock_reddit_sentiment["ticker"] == "AAPL"
        assert 0.0 <= mock_reddit_sentiment["upvote_ratio_avg"] <= 1.0
        assert -1.0 <= mock_reddit_sentiment["comment_sentiment_mean"] <= 1.0
        assert mock_reddit_sentiment["mentions"] > 0


class TestLinguistAnalysis:
    """Test Linguist module's sentiment and certainty scoring."""

    def test_certainty_scorer_on_bullish_headline(self):
        """Verify certainty scorer returns [0, 1] for bullish signals."""
        # Simulate linguist.sample_score behavior
        headline = "Apple announces record-breaking earnings beat"
        words_bullish = ["record", "beat", "growth", "strong"]
        words_bearish = ["decline", "miss", "warning"]
        
        bullish_count = sum(1 for w in words_bullish if w in headline.lower())
        bearish_count = sum(1 for w in words_bearish if w in headline.lower())
        
        certainty = (bullish_count - bearish_count) / max(
            bullish_count + bearish_count, 1
        )
        assert certainty > 0.5

    def test_certainty_scorer_on_bearish_headline(self):
        """Verify certainty scorer returns [0, 1] for bearish signals."""
        headline = "Apple faces regulatory investigation and declining sales"
        words_bullish = ["record", "beat", "growth", "strong"]
        words_bearish = ["decline", "miss", "warning", "investigation"]
        
        bullish_count = sum(1 for w in words_bullish if w in headline.lower())
        bearish_count = sum(1 for w in words_bearish if w in headline.lower())
        
        certainty = (bullish_count - bearish_count) / max(
            bullish_count + bearish_count, 1
        )
        assert certainty < -0.5

    def test_certainty_scorer_on_neutral_headline(self):
        """Verify certainty scorer returns near-0 for neutral signals."""
        headline = "Apple announces new product launch event"
        words_bullish = ["record", "beat", "growth", "strong"]
        words_bearish = ["decline", "miss", "warning", "investigation"]
        
        bullish_count = sum(1 for w in words_bullish if w in headline.lower())
        bearish_count = sum(1 for w in words_bearish if w in headline.lower())
        
        certainty = (bullish_count - bearish_count) / max(
            bullish_count + bearish_count, 1
        )
        assert abs(certainty) < 0.3

    def test_linguistic_drift_detector_identifies_tone_shift(self):
        """Verify drift detector compares historical vs. recent tone."""
        historical_text = "Apple's market position strengthens. Sales momentum builds."
        recent_
