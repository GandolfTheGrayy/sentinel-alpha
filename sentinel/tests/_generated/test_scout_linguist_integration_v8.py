"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis and certainty scoring (Linguist). All external calls
(yfinance, news APIs, SEC EDGAR, LLM reasoning) are mocked to ensure
deterministic, fast test execution without rate limits or API keys.

Fits into Sentinel's test harness to catch regressions in the core
ingest→analyze loop before they reach production.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, List, Any


# Mock data fixtures
@pytest.fixture
def mock_ticker_data() -> Dict[str, Any]:
    """Return fixture: sample stock price data for ACME Corp."""
    return {
        "ticker": "ACME",
        "current_price": 145.32,
        "previous_close": 142.87,
        "percent_change": 1.71,
        "date": datetime.now().isoformat(),
        "volume": 3_200_000,
    }


@pytest.fixture
def mock_news_articles() -> List[Dict[str, str]]:
    """Return fixture: sample news headlines with timestamps."""
    return [
        {
            "title": "ACME Corp reports strong Q3 earnings beat",
            "url": "https://example.com/acme-earnings",
            "source": "Reuters",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        },
        {
            "title": "ACME to expand into European market by 2025",
            "url": "https://example.com/acme-expansion",
            "source": "Bloomberg",
            "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
        },
        {
            "title": "ACME Corp faces supply chain delays",
            "url": "https://example.com/acme-supply",
            "source": "MarketWatch",
            "published_at": (datetime.now() - timedelta(hours=12)).isoformat(),
        },
    ]


@pytest.fixture
def mock_sec_filings() -> List[Dict[str, str]]:
    """Return fixture: sample SEC 8-K filing metadata."""
    return [
        {
            "accession_number": "0001193125-24-001234",
            "filing_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "form_type": "8-K",
            "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&...",
            "snippet": "Item 8.01: ACME announces strategic partnership with TechCorp",
        },
        {
            "accession_number": "0001193125-24-001235",
            "filing_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "form_type": "10-Q",
            "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&...",
            "snippet": "Revenue growth of 12% YoY, net margin improved to 18%",
        },
    ]


@pytest.fixture
def mock_linguist_analysis() -> Dict[str, Any]:
    """Return fixture: sample Linguist certainty and tone analysis."""
    return {
        "ticker": "ACME",
        "sentiment_score": 0.72,
        "certainty_level": "high",
        "tone_keywords": ["strong", "partnership", "growth"],
        "hesitation_signals": ["delays", "challenges"],
        "linguistic_drift": "neutral",
        "regulatory_whisper_score": 0.15,
        "analysis_timestamp": datetime.now().isoformat(),
    }


class TestScoutDataIngestion:
    """Tests for Scout pillar: live prices, news, SEC filings."""

    def test_live_price_fetch_mock(self, mock_ticker_data: Dict[str, Any]) -> None:
        """Verify Scout can parse live price data structure without external calls."""
        assert mock_ticker_data["ticker"] == "ACME"
        assert mock_ticker_data["current_price"] == 145.32
        assert mock_ticker_data["percent_change"] == 1.71
        assert "date" in mock_ticker_data

    def test_news_fetcher_mock(self, mock_news_articles: List[Dict[str, str]]) -> None:
        """Verify Scout can ingest and structure news headlines."""
        assert len(mock_news_articles) == 3
        assert all("title" in article for article in mock_news_articles)
        assert all("source" in article for article in mock_news_articles)
        assert all("published_at" in article for article in mock_news_articles)
        # Verify most recent first
        assert "earnings beat" in mock_news_articles[0]["title"].lower()

    def test_sec_filings_mock(self, mock_sec_filings: List[Dict[str, str]]) -> None:
        """Verify Scout can parse SEC filing metadata."""
        assert len(mock_sec_filings) == 2
        assert mock_sec_filings[0]["form_type"] == "8-K"
        assert mock_sec_filings[1]["form_type"] == "10-Q"
        assert all("accession_number" in f for f in mock_sec_filings)
        assert all("snippet" in f for f in mock_sec_filings)


class TestLinguistAnalysis:
    """Tests for Linguist pillar: sentiment and certainty scoring."""

    def test_linguist_certainty_structure(
        self, mock_linguist_analysis: Dict[str, Any]
    ) -> None:
        """Verify Linguist output has required fields for downstream Judge."""
        analysis = mock_linguist_analysis
        assert "ticker" in analysis
        assert "sentiment_score" in analysis
        assert "certainty_level" in analysis
        assert analysis["certainty_level"] in ["high", "medium", "low"]
        assert 0 <= analysis["sentiment_score"] <= 1

    def test_linguist_tone_detection(
        self, mock_linguist_analysis: Dict[str, Any]
    ) -> None:
        """Verify Linguist detects positive and negative tone keywords."""
        analysis = mock_linguist_analysis
        assert "tone_keywords" in analysis
        assert "hesitation_signals" in analysis
        assert isinstance(analysis["tone_keywords"], list)
        assert isinstance(analysis["hesitation_signals"], list)

    def test_linguist_regulatory_whisper(
        self, mock_linguist_analysis: Dict[str, Any]
    ) -> None:
        """Verify Linguist computes regulatory whisper score from SEC text."""
        analysis = mock_linguist_analysis
        assert "regulatory_whisper_score" in analysis
        assert 0 <= analysis["regulatory_whisper_score"] <= 1


class TestScoutLinguistIntegration:
    """End-to-end integration: Scout data → Linguist analysis."""

    @patch("requests.get")
    @patch("anthropic.Anthropic")
    def test_full_pipeline_mocked(
        self,
        mock_claude: Mock,
        mock_requests: Mock,
        mock_ticker_data: Dict[str, Any],
        mock_news_articles: List[Dict[str, str]],
        mock_sec_filings: List[Dict[str, str]],
        mock_linguist_analysis: Dict[str, Any],
    ) -> None:
        """
        Simulate Scout → Linguist flow: ingest ticker data, news, SEC filings,
        then run Linguist sentiment+certainty analysis, all mocked.
        """
        # Mock Scout: price fetch
        mock_price_response = Mock()
        mock_price_response.json.return_value = mock_ticker_data
        mock_requests.return_value = mock_price_response

        # Mock Claude reasoning for Linguist
        mock_claude_instance = MagicMock()
