"""
Sentinel Scout → Linguist integration test.

This pytest module validates the end-to-end data flow from Scout (live prices,
news, SEC filings) through Linguist (certainty scoring, linguistic drift detection)
with all external calls mocked. It ensures that sentiment signals and raw market
data flow correctly through the reasoning pipeline before Judge makes predictions.

Tests cover:
  - Live price fetcher with fallback logic
  - News headline ingestion and parsing
  - SEC EDGAR 8-K/10-Q retrieval and text extraction
  - Certainty scoring on real-world sentiment samples
  - Linguistic drift detection across time windows
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from typing import Any

import pytest


class TestScoutLivePrice:
    """Unit tests for Scout live price fetcher."""

    @patch("yfinance.Ticker")
    def test_fetch_live_price_success(self, mock_ticker: Mock) -> None:
        """Verify live price fetch returns price and timestamp."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = {
            "Close": [150.25],
            "Volume": [1_000_000],
        }
        mock_ticker.return_value = mock_ticker_instance

        from sentinel.scout.live_prices import fetch_live_price

        price, timestamp = fetch_live_price("AAPL")
        assert price == 150.25
        assert isinstance(timestamp, datetime)

    @patch("yfinance.Ticker")
    def test_fetch_live_price_fallback_to_stooq(self, mock_ticker: Mock) -> None:
        """Verify fallback to stooq when yfinance fails."""
        mock_ticker.return_value.history.side_effect = Exception("yfinance down")

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.text = "AAPL,150.25,2024-01-15"
            mock_get.return_value = mock_response

            from sentinel.scout.live_prices import fetch_live_price

            price, timestamp = fetch_live_price("AAPL")
            assert price == 150.25


class TestScoutNews:
    """Unit tests for Scout news headline fetcher."""

    @patch("requests.get")
    def test_fetch_news_headlines_success(self, mock_get: Mock) -> None:
        """Verify news headline fetch returns structured data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Apple beats Q4 earnings expectations",
                    "url": "https://example.com/news/1",
                    "publishedAt": "2024-01-15T10:30:00Z",
                    "source": {"name": "Reuters"},
                }
            ]
        }
        mock_get.return_value = mock_response

        from sentinel.scout.news import fetch_news_headlines

        headlines = fetch_news_headlines("AAPL", limit=5)
        assert len(headlines) == 1
        assert headlines[0]["title"] == "Apple beats Q4 earnings expectations"

    @patch("requests.get")
    def test_fetch_news_empty_response(self, mock_get: Mock) -> None:
        """Verify graceful handling of empty news response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"articles": []}
        mock_get.return_value = mock_response

        from sentinel.scout.news import fetch_news_headlines

        headlines = fetch_news_headlines("UNKNOWN", limit=5)
        assert headlines == []


class TestScoutSecFilings:
    """Unit tests for Scout SEC EDGAR scraper."""

    @patch("requests.get")
    def test_fetch_sec_filings_8k_success(self, mock_get: Mock) -> None:
        """Verify 8-K filing retrieval and text extraction."""
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <body>
                <div id="filing-documents">
                    <a href="/Archives/8k.htm">8-K</a>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        from sentinel.scout.sec_filings import fetch_sec_filings

        filings = fetch_sec_filings("AAPL", form_type="8-K")
        assert isinstance(filings, list)

    @patch("requests.get")
    def test_fetch_sec_filings_10q_success(self, mock_get: Mock) -> None:
        """Verify 10-Q filing retrieval."""
        mock_response = MagicMock()
        mock_response.text = """
        <html>
            <body>
                <div id="filing-documents">
                    <a href="/Archives/10q.htm">10-Q</a>
                </div>
            </body>
        </html>
        """
        mock_get.return_value = mock_response

        from sentinel.scout.sec_filings import fetch_sec_filings

        filings = fetch_sec_filings("AAPL", form_type="10-Q")
        assert isinstance(filings, list)

    @patch("requests.get")
    def test_fetch_sec_filings_connection_error(self, mock_get: Mock) -> None:
        """Verify graceful handling of SEC connection errors."""
        mock_get.side_effect = Exception("Connection timeout")

        from sentinel.scout.sec_filings import fetch_sec_filings

        filings = fetch_sec_filings("AAPL", form_type="8-K")
        assert filings == []


class TestLinguistCertaintyScorer:
    """Unit tests for Linguist certainty and hesitation analysis."""

    @patch("anthropic.Anthropic")
    def test_score_certainty_high_confidence(self, mock_anthropic: Mock) -> None:
        """Verify certainty scorer identifies high-confidence signals."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps({
            "certainty_score": 0.87,
            "hesitation_markers": [],
            "reasoning": "Clear earnings beat with strong guidance.",
        })
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            from sentinel.linguist.sample_score import score_certainty

            text = "We achieved record earnings and significantly raised full-year guidance."
            result = score_certainty(text)
            assert result["certainty_score"] >= 0.8

    @patch("anthropic.Anthropic")
    def test_score_certainty_low_confidence(self, mock_anthropic: Mock) -> None:
        """Verify certainty scorer identifies low-confidence signals."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps({
            "certainty_score": 0.32,
            "hesitation_markers": ["may", "could", "potentially"],
            "reasoning": "Multiple conditional phrases and regulatory uncertainty.",
        })
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            from sentinel.linguist.sample_score import score_certainty

            text = "We may face headwinds, though we could see growth if conditions improve."
            result = score_certainty(text)
            assert result["certainty_score"] <= 0.5

    @patch("anthropic.Anthropic")
    def test_score_certainty_mixed_sentiment(self, mock_anthropic: Mock) -> None:
        """Verify certainty scorer handles mixed sentiment correctly."""
        mock_client = MagicMock()
        mock_response = Magic
