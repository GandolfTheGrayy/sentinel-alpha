"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) using mocked external API calls. It ensures that
live prices, news headlines, and SEC filings are correctly collected, parsed,
and fed into the LLM-based certainty scorer without requiring live internet access.

Part of Sentinel's test suite; runs via pytest.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from sentinel.scout.live_prices import fetch_live_price
from sentinel.scout.news import fetch_news_headlines
from sentinel.scout.sec_filings import fetch_sec_filings
from sentinel.linguist.sample_score import score_sentiment_certainty


class TestScoutLiveprices:
    """Test Suite for Scout live price fetcher."""

    @patch("yfinance.Ticker")
    def test_fetch_live_price_success(self, mock_ticker: MagicMock) -> None:
        """Test successful fetch of live stock price."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"currentPrice": 150.25}
        mock_ticker.return_value = mock_ticker_instance

        price = fetch_live_price("AAPL")
        assert price == 150.25

    @patch("yfinance.Ticker")
    def test_fetch_live_price_fallback(self, mock_ticker: MagicMock) -> None:
        """Test fallback behavior when yfinance fails."""
        mock_ticker.side_effect = Exception("API error")

        with patch("sentinel.scout.live_prices.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {"close": 149.75}
            mock_get.return_value = mock_response

            price = fetch_live_price("AAPL")
            assert price == 149.75

    @patch("yfinance.Ticker")
    def test_fetch_live_price_invalid_ticker(self, mock_ticker: MagicMock) -> None:
        """Test handling of invalid ticker symbol."""
        mock_ticker.return_value.info = {}

        price = fetch_live_price("INVALID")
        assert price is None


class TestScoutNews:
    """Test Suite for Scout news headline fetcher."""

    @patch("requests.get")
    def test_fetch_news_headlines_success(self, mock_get: MagicMock) -> None:
        """Test successful fetch of news headlines."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Apple beats Q3 earnings expectations",
                    "description": "Strong iPhone sales drive revenue",
                    "url": "https://example.com/apple-earnings",
                    "publishedAt": "2024-01-15T10:00:00Z",
                }
            ]
        }
        mock_get.return_value = mock_response

        headlines = fetch_news_headlines("AAPL", limit=5)
        assert len(headlines) == 1
        assert headlines[0]["title"] == "Apple beats Q3 earnings expectations"

    @patch("requests.get")
    def test_fetch_news_headlines_empty(self, mock_get: MagicMock) -> None:
        """Test handling of empty news response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"articles": []}
        mock_get.return_value = mock_response

        headlines = fetch_news_headlines("UNKNOWN", limit=5)
        assert len(headlines) == 0

    @patch("requests.get")
    def test_fetch_news_headlines_http_error(self, mock_get: MagicMock) -> None:
        """Test handling of HTTP errors during news fetch."""
        mock_get.side_effect = Exception("Connection timeout")

        headlines = fetch_news_headlines("AAPL")
        assert headlines == []


class TestScoutSecFilings:
    """Test Suite for Scout SEC filings scraper."""

    @patch("requests.get")
    def test_fetch_sec_filings_success(self, mock_get: MagicMock) -> None:
        """Test successful fetch of SEC 8-K filings."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <tr>
                <td>8-K</td>
                <td>2024-01-10</td>
                <td>Material agreement signed</td>
            </tr>
        </html>
        """
        mock_get.return_value = mock_response

        filings = fetch_sec_filings("0000789019", form_type="8-K", limit=5)
        assert isinstance(filings, list)

    @patch("requests.get")
    def test_fetch_sec_filings_empty(self, mock_get: MagicMock) -> None:
        """Test handling of no SEC filings found."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>No filings</body></html>"
        mock_get.return_value = mock_response

        filings = fetch_sec_filings("9999999999", form_type="8-K")
        assert filings == []

    @patch("requests.get")
    def test_fetch_sec_filings_network_error(self, mock_get: MagicMock) -> None:
        """Test handling of network errors during SEC fetch."""
        mock_get.side_effect = Exception("SEC server unavailable")

        filings = fetch_sec_filings("0000789019")
        assert filings == []


class TestLinguistSentimentScoring:
    """Test Suite for Linguist sentiment certainty scorer."""

    @patch("anthropic.Anthropic")
    def test_score_sentiment_certainty_bullish(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Test scoring of bullish sentiment."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps(
            {
                "certainty": 0.82,
                "sentiment": "bullish",
                "reasoning": "Strong earnings beat and forward guidance",
            }
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        score = score_sentiment_certainty(
            ticker="AAPL",
            text="Apple reported record Q3 revenue and raised FY guidance",
        )

        assert score["certainty"] == 0.82
        assert score["sentiment"] == "bullish"

    @patch("anthropic.Anthropic")
    def test_score_sentiment_certainty_bearish(
        self, mock_anthropic: MagicMock
    ) -> None:
        """Test scoring of bearish sentiment."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content[0].text = json.dumps(
            {
                "certainty": 0.65,
                "sentiment": "bearish",
                "reasoning": "Supply chain concerns and margin compression",
            }
        )
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        score = score_sentiment_certainty(
            ticker="AAPL",
            text="Apple warns of supply chain headwinds impacting Q4",
        )

        assert score["certainty"] == 0.65
        assert score["sentiment"] == "bearish"

    @patch("anthropic.Anthropic")
    def test_score_sentiment_certainty_neutral(self, mock_anthropic: MagicMock) ->
