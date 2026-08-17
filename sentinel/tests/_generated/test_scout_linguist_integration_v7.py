"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) with mocked external calls. It ensures that live
prices, news headlines, and SEC filings are correctly fetched, parsed, and
scored for linguistic certainty and regulatory signals without hitting real APIs.

Part of the Sentinel daily test suite; run via pytest.
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import sys
import os

# Ensure sentinel modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sentinel.scout.live_prices import fetch_live_price
from sentinel.scout.news import fetch_news_headlines
from sentinel.scout.sec_filings import fetch_sec_filings
from sentinel.linguist.sample_score import score_certainty


class TestScoutLinguistIntegration(unittest.TestCase):
    """Integration tests for Scout data ingestion + Linguist analysis."""

    def setUp(self) -> None:
        """Initialize test fixtures."""
        self.test_ticker = "AAPL"
        self.test_company_name = "Apple Inc."

    @patch("yfinance.Ticker")
    def test_live_price_fetch_with_mock(self, mock_yf_ticker: Mock) -> None:
        """Verify live price fetcher returns structured data on success."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {"currentPrice": 175.50}
        mock_yf_ticker.return_value = mock_ticker_instance

        price = fetch_live_price(self.test_ticker)
        self.assertIsNotNone(price)
        self.assertGreater(price, 0)

    @patch("requests.get")
    def test_news_headlines_fetch_with_mock(self, mock_requests_get: Mock) -> None:
        """Verify news headline fetcher parses mock HTML response."""
        mock_html = """
        <html>
            <div class="newsitem">
                <a href="/news/123">Apple stock surges on Q4 earnings beat</a>
                <span class="date">2024-01-15</span>
            </div>
            <div class="newsitem">
                <a href="/news/124">Tim Cook signals strong iPhone demand</a>
                <span class="date">2024-01-14</span>
            </div>
        </html>
        """
        mock_response = Mock()
        mock_response.text = mock_html
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        headlines = fetch_news_headlines(self.test_ticker)
        self.assertIsInstance(headlines, list)
        self.assertGreater(len(headlines), 0)
        if headlines:
            self.assertIn("text", headlines[0])
            self.assertIn("source", headlines[0])

    @patch("requests.get")
    def test_sec_filings_fetch_with_mock(self, mock_requests_get: Mock) -> None:
        """Verify SEC EDGAR fetcher retrieves and parses 8-K filings."""
        mock_json_response = {
            "filings": {
                "recent": {
                    "accessionNumber": ["0000320193-24-000001"],
                    "filingDate": ["2024-01-10"],
                    "form": ["8-K"],
                    "documentAndEntityInformation": [
                        {
                            "entityName": "APPLE INC",
                            "cik": "0000320193"
                        }
                    ]
                }
            }
        }
        mock_response = Mock()
        mock_response.json.return_value = mock_json_response
        mock_response.status_code = 200
        mock_requests_get.return_value = mock_response

        filings = fetch_sec_filings(self.test_ticker, filing_types=["8-K"])
        self.assertIsInstance(filings, list)

    def test_linguist_certainty_score_with_sample_text(self) -> None:
        """Verify Linguist analyzes sentiment text and returns structured scores."""
        sample_texts = [
            "We are confident the market will respond positively to our Q4 results.",
            "We may see some headwinds in the coming quarter, but we'll assess.",
            "This is definitely a transformative acquisition that will drive growth."
        ]

        for text in sample_texts:
            score_dict = score_certainty(text)
            self.assertIsInstance(score_dict, dict)
            self.assertIn("certainty_score", score_dict)
            self.assertIn("sentiment", score_dict)
            self.assertIn("hedging_language", score_dict)
            # Verify score is normalized to [0, 1]
            self.assertGreaterEqual(score_dict["certainty_score"], 0.0)
            self.assertLessEqual(score_dict["certainty_score"], 1.0)

    @patch("sentinel.scout.news.fetch_news_headlines")
    @patch("sentinel.scout.live_prices.fetch_live_price")
    def test_scout_pipeline_chain(
        self, mock_price: Mock, mock_news: Mock
    ) -> None:
        """Verify Scout modules can be chained: price → news → sentiment."""
        # Mock price fetch
        mock_price.return_value = 175.50

        # Mock news fetch
        mock_news.return_value = [
            {
                "text": "Apple stock surges on record iPhone sales",
                "source": "Reuters",
                "date": "2024-01-15"
            },
            {
                "text": "Analysts cautiously optimistic on Services revenue",
                "source": "Bloomberg",
                "date": "2024-01-14"
            }
        ]

        # Fetch data
        price = fetch_live_price(self.test_ticker)
        news = fetch_news_headlines(self.test_ticker)

        # Validate chain output
        self.assertIsNotNone(price)
        self.assertEqual(price, 175.50)
        self.assertEqual(len(news), 2)

        # Pass news through Linguist
        scores = [score_certainty(item["text"]) for item in news]
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertIn("certainty_score", score)

    @patch("sentinel.scout.sec_filings.fetch_sec_filings")
    def test_sec_filing_linguistic_analysis(
        self, mock_sec_fetch: Mock
    ) -> None:
        """Verify SEC filing text can be analyzed for regulatory tone."""
        mock_sec_fetch.return_value = [
            {
                "accession": "0000320193-24-000001",
                "filing_date": "2024-01-10",
                "form_type": "8-K",
                "text": (
                    "We have determined that a material definitive agreement "
                    "has been executed. Management is confident in our ability "
                    "to realize synergies."
                )
            }
        ]

        filings = fetch_sec_filings(self.test_ticker, filing_types=["8-K"])
        self.assertGreater(len(filings), 0)

        # Analyze filing text for certainty
        for filing in filings:
            score = score_certainty(filing["text"])
            self.assertIsInstance(score, dict)
            self.assertGreaterEqual(score["certainty_score"], 0.0)

    def test_linguist_detects_hedging_language(self) -> None:
        """Verify Linguist correctly identifies hedging qualifiers."""
        hedged_text = "We may potentially see some possible upside, assuming conditions permit."
        confident_text = "We will drive significant revenue growth this quarter."

        hedged_score = score_certainty(hedged_text)
        confident_score = score_certainty(confident_text)

        # Hedged text
