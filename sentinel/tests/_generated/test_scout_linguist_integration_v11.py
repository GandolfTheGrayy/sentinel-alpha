"""
Integration test for Scout → Linguist pipeline.

This module validates end-to-end data flow from Scout (price, news, SEC filing ingestion)
through Linguist (sentiment analysis, certainty scoring) using mocked external API calls.
It verifies that raw signals are correctly transformed into actionable sentiment scores
without hitting live endpoints. Part of Sentinel's spine validation suite.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import Scout and Linguist modules
import sys
from pathlib import Path

# Add sentinel root to path for imports
sentinel_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sentinel_root))

from scout.live_prices import fetch_live_price
from scout.news import fetch_news_headlines
from scout.sec_filings import fetch_recent_filings
from linguist.sample_score import score_certainty


class TestScoutLinguistIntegration:
    """Integration test suite for Scout data ingestion → Linguist analysis pipeline."""

    @pytest.fixture
    def mock_yfinance_price(self) -> Dict[str, Any]:
        """Mock yfinance price response."""
        return {
            "AAPL": {
                "currentPrice": 185.42,
                "currency": "USD",
                "regularMarketChange": 2.34,
                "regularMarketChangePercent": 1.28,
                "fiftyTwoWeekHigh": 199.62,
                "fiftyTwoWeekLow": 164.89,
            }
        }

    @pytest.fixture
    def mock_news_response(self) -> List[Dict[str, str]]:
        """Mock news API response."""
        return [
            {
                "title": "Apple Q4 earnings beat expectations with strong iPhone sales",
                "url": "https://example.com/news/1",
                "source": "Reuters",
                "published": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "sentiment_label": "positive",
            },
            {
                "title": "Analyst downgrades Apple on supply chain concerns",
                "url": "https://example.com/news/2",
                "source": "MarketWatch",
                "published": (datetime.utcnow() - timedelta(hours=5)).isoformat(),
                "sentiment_label": "negative",
            },
            {
                "title": "Apple announces new product line for enterprise",
                "url": "https://example.com/news/3",
                "source": "TechCrunch",
                "published": (datetime.utcnow() - timedelta(hours=8)).isoformat(),
                "sentiment_label": "neutral",
            },
        ]

    @pytest.fixture
    def mock_sec_response(self) -> List[Dict[str, str]]:
        """Mock SEC EDGAR filing response."""
        return [
            {
                "accession": "0000320193-24-000006",
                "filing_type": "8-K",
                "filed_date": "2024-01-15",
                "url": "https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-24-000006",
                "snippet": "Material agreement signed with major enterprise customer for cloud services.",
            },
            {
                "accession": "0000320193-23-000105",
                "filing_type": "10-Q",
                "filed_date": "2024-01-10",
                "url": "https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-23-000105",
                "snippet": "Revenue grew 8% YoY. Gross margin compressed due to competitive pricing.",
            },
        ]

    @pytest.fixture
    def mock_sentiment_scores(self) -> Dict[str, float]:
        """Mock Linguist sentiment scoring output."""
        return {
            "overall_sentiment": 0.68,
            "certainty_score": 0.75,
            "hesitation_detected": False,
            "linguistic_drift": "stable",
            "regulatory_whisper_risk": 0.12,
        }

    def test_scout_price_fetch_mocked(self, mock_yfinance_price: Dict[str, Any]) -> None:
        """Test Scout price fetcher with mocked yfinance call."""
        with patch("scout.live_prices.yf.Ticker") as mock_ticker:
            mock_instance = Mock()
            mock_instance.info = mock_yfinance_price["AAPL"]
            mock_ticker.return_value = mock_instance

            result = fetch_live_price("AAPL")

            assert result is not None
            assert result["currentPrice"] == 185.42
            assert result["regularMarketChangePercent"] == 1.28

    def test_scout_news_fetch_mocked(self, mock_news_response: List[Dict[str, str]]) -> None:
        """Test Scout news fetcher with mocked HTTP call."""
        with patch("scout.news.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"articles": mock_news_response}
            mock_get.return_value = mock_response

            result = fetch_news_headlines("AAPL", limit=3)

            assert len(result) == 3
            assert result[0]["title"].startswith("Apple Q4 earnings")
            assert result[1]["sentiment_label"] == "negative"

    def test_scout_sec_filings_mocked(self, mock_sec_response: List[Dict[str, str]]) -> None:
        """Test Scout SEC filing scraper with mocked HTTP call."""
        with patch("scout.sec_filings.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"filings": mock_sec_response}
            mock_get.return_value = mock_response

            result = fetch_recent_filings("0000320193", days=30, filing_types=["8-K", "10-Q"])

            assert len(result) == 2
            assert result[0]["filing_type"] == "8-K"
            assert "Material agreement" in result[0]["snippet"]

    def test_linguist_sentiment_scoring_mocked(
        self, mock_sentiment_scores: Dict[str, float]
    ) -> None:
        """Test Linguist certainty scorer with mocked Claude call."""
        sample_text = (
            "Strong earnings beat. Supply chain normalized. "
            "Analyst confidence elevated despite macro headwinds."
        )

        with patch("linguist.sample_score.anthropic.Anthropic") as mock_anthropic:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(text=json.dumps(mock_sentiment_scores))]
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            result = score_certainty(sample_text)

            assert result["overall_sentiment"] == 0.68
            assert result["certainty_score"] == 0.75
            assert result["hesitation_detected"] is False
            assert result["linguistic_drift"] == "stable"

    def test_scout_linguist_pipeline_end_to_end(
        self,
        mock_yfinance_price: Dict[str, Any],
        mock_news_response: List[Dict[str, str]],
        mock_sec_response: List[Dict[str, str]],
        mock_sentiment_scores: Dict[str, float],
    ) -> None:
        """Test full Scout → Linguist pipeline with all mocked calls."""
        with patch("scout.live_prices.yf.Ticker") as mock_ticker, \
             patch("scout.news.requests.get") as mock_news_get, \
             patch("scout.
