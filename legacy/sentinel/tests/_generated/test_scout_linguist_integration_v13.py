"""
Integration test for Scout → Linguist pipeline.

This test validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) with mocked external API calls. It ensures that
price fetches, news scrapes, SEC filing retrieval, and certainty scoring work
together correctly without hitting live endpoints.

Part of Sentinel's test harness; runs via pytest.
"""

import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List
import json

import pytest
import numpy as np


# Add sentinel root to path for imports
SENTINEL_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SENTINEL_ROOT))

from sentinel.scout import live_prices, news, sec_filings
from sentinel.linguist import sample_score


class TestScoutLinguistIntegration:
    """Integration test suite for Scout ingest → Linguist scoring pipeline."""

    @pytest.fixture
    def mock_yfinance_data(self) -> Dict[str, Any]:
        """Generate mock OHLCV data for a test ticker."""
        return {
            "AAPL": {
                "price": 150.25,
                "timestamp": datetime.now(),
                "open": 149.50,
                "high": 151.80,
                "low": 149.00,
                "close": 150.25,
                "volume": 52_345_678,
            }
        }

    @pytest.fixture
    def mock_news_articles(self) -> List[Dict[str, str]]:
        """Generate mock news articles for sentiment analysis."""
        return [
            {
                "title": "Apple Beats Q4 Earnings Expectations",
                "summary": "Strong revenue growth driven by iPhone sales and services expansion.",
                "url": "https://example.com/apple-earnings-1",
                "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "sentiment": "positive",
            },
            {
                "title": "Apple Faces Regulatory Scrutiny Over App Store Practices",
                "summary": "EU regulators launch investigation into competitive concerns.",
                "url": "https://example.com/apple-regulatory",
                "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "sentiment": "negative",
            },
            {
                "title": "Apple Announces New M4 Chip for MacBook Pro",
                "summary": "Performance improvements expected to drive hardware upgrade cycle.",
                "url": "https://example.com/apple-m4",
                "published_at": (datetime.now() - timedelta(hours=12)).isoformat(),
                "sentiment": "positive",
            },
        ]

    @pytest.fixture
    def mock_sec_filings(self) -> List[Dict[str, str]]:
        """Generate mock SEC 8-K and 10-Q filing excerpts."""
        return [
            {
                "filing_type": "8-K",
                "date": (datetime.now() - timedelta(days=1)).date().isoformat(),
                "headline": "Material Agreement Entered Into",
                "excerpt": "Company has entered into a strategic partnership expected to drive revenue.",
                "url": "https://example.com/sec-8k-1",
            },
            {
                "filing_type": "10-Q",
                "date": (datetime.now() - timedelta(days=5)).date().isoformat(),
                "headline": "Q3 Financial Results",
                "excerpt": "Revenue increased 15% YoY. Operating margins expanded due to operational efficiency.",
                "url": "https://example.com/sec-10q-1",
            },
        ]

    def test_live_prices_fetch(self, mock_yfinance_data: Dict[str, Any]) -> None:
        """Test live price fetching with mocked yfinance."""
        with patch("sentinel.scout.live_prices.yf.download") as mock_download:
            mock_df = MagicMock()
            mock_df["Close"].iloc[-1] = 150.25
            mock_df["Volume"].iloc[-1] = 52_345_678
            mock_download.return_value = mock_df

            result = live_prices.fetch_live_price("AAPL")
            assert result is not None
            assert isinstance(result, dict)
            assert "price" in result or result.get("price") is not None

    def test_news_fetch_and_parse(self, mock_news_articles: List[Dict[str, str]]) -> None:
        """Test news article fetching and basic parsing."""
        with patch("sentinel.scout.news.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "articles": mock_news_articles
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            articles = news.fetch_news_for_ticker("AAPL", limit=3)
            assert articles is not None
            assert isinstance(articles, list)
            # Mock returns 3 articles; verify structure
            if len(articles) > 0:
                assert "title" in articles[0] or "headline" in articles[0]

    def test_sec_filings_fetch(self, mock_sec_filings: List[Dict[str, str]]) -> None:
        """Test SEC EDGAR filing retrieval."""
        with patch("sentinel.scout.sec_filings.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "filings": mock_sec_filings
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            filings = sec_filings.fetch_sec_filings("0000320193", filing_types=["8-K", "10-Q"])
            assert filings is not None
            assert isinstance(filings, list)

    def test_linguist_certainty_scorer_positive(self) -> None:
        """Test Linguist certainty scoring on positive sentiment text."""
        text = (
            "Strong earnings beat. Revenue exceeded guidance by 12%. "
            "Management expressed confidence in Q1 outlook. "
            "Multiple analysts upgraded the stock."
        )

        with patch("sentinel.linguist.sample_score.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text='{"certainty": 0.82, "sentiment": "bullish"}')]
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            score = sample_score.score_certainty(text, ticker="AAPL")
            assert score is not None
            assert isinstance(score, dict)
            assert "certainty" in score or "sentiment" in score

    def test_linguist_certainty_scorer_negative(self) -> None:
        """Test Linguist certainty scoring on negative/uncertain sentiment text."""
        text = (
            "Regulatory investigation may impact operations. "
            "Management cautious about near-term guidance. "
            "Some analysts express concerns about competitive pressure."
        )

        with patch("sentinel.linguist.sample_score.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_message = MagicMock()
            mock_message.content = [MagicMock(text='{"certainty": 0.45, "sentiment": "bearish"}')]
            mock_client.messages.create.return_value = mock_message
            mock_anthropic.return_value = mock_client

            score = sample_score.score_certainty(text, ticker="AAPL")
            assert score is not None
            assert isinstance(score, dict)

    def test_scout_linguist_pipeline_integration(
        self,
        mock_yfinance_data: Dict[str, Any],
