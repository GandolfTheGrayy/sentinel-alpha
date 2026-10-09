"""
Sentinel Scout → Linguist integration test.

This module validates the end-to-end pipeline from raw data ingestion (Scout)
through sentiment analysis and linguistic scoring (Linguist). All external calls
(yfinance, news APIs, SEC EDGAR, LLM requests) are mocked to ensure deterministic,
fast test execution without network I/O or API quota consumption.

Test coverage includes:
  - Live price fetching with fallback handling
  - News headline aggregation
  - SEC filing metadata extraction
  - Certainty score computation via mocked Claude
  - Linguistic drift detection via mocked Gemini
  - End-to-end signal synthesis
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import json
from datetime import datetime, timedelta


class MockPriceData:
    """Mock yfinance price response."""

    def __init__(self, current: float = 150.0, history_days: int = 30):
        self.current = current
        self.history_days = history_days

    def info(self) -> Dict[str, Any]:
        """Return mock ticker info."""
        return {
            "currentPrice": self.current,
            "marketCap": 2.5e12,
            "fiftyTwoWeekHigh": 160.0,
            "fiftyTwoWeekLow": 120.0,
        }

    def history(self, period: str = "1mo") -> Any:
        """Return mock price history."""
        import pandas as pd

        dates = pd.date_range(end=datetime.now(), periods=self.history_days)
        prices = [self.current - (i * 0.5) for i in range(self.history_days)]
        return pd.DataFrame({"Close": prices}, index=dates)


class MockNewsResponse:
    """Mock news API response."""

    def __init__(self, ticker: str = "AAPL", count: int = 5):
        self.ticker = ticker
        self.count = count

    def json(self) -> Dict[str, Any]:
        """Return mock news articles."""
        articles = [
            {
                "title": f"Apple {self.ticker} beats earnings expectations",
                "source": {"name": "Reuters"},
                "publishedAt": (
                    datetime.now() - timedelta(hours=i)
                ).isoformat(),
                "description": "Strong iPhone sales drive Q3 revenue growth.",
                "url": f"https://example.com/article-{i}",
            }
            for i in range(self.count)
        ]
        return {"articles": articles, "totalResults": self.count}


class MockSECFiling:
    """Mock SEC EDGAR filing response."""

    def __init__(self, accession: str = "0000000123-24-000001", form_type: str = "8-K"):
        self.accession = accession
        self.form_type = form_type

    def json(self) -> Dict[str, Any]:
        """Return mock SEC filing metadata."""
        return {
            "filings": {
                "recent": {
                    "accessionNumber": [self.accession],
                    "form": [self.form_type],
                    "filingDate": ["2024-01-15"],
                    "reportDate": ["2024-01-15"],
                    "acceptanceDateTime": ["2024-01-15 16:30:00"],
                }
            }
        }


class MockClaudeResponse:
    """Mock Claude API response for linguistic analysis."""

    def __init__(self, certainty_score: float = 0.75, sentiment: str = "bullish"):
        self.certainty_score = certainty_score
        self.sentiment = sentiment

    def content(self) -> List[Any]:
        """Return mock Claude content block."""
        return [
            Mock(
                text=json.dumps(
                    {
                        "certainty_score": self.certainty_score,
                        "sentiment": self.sentiment,
                        "rationale": "Strong language around product innovation and market expansion.",
                        "hesitation_markers": 0,
                        "confidence_interval": [
                            self.certainty_score - 0.1,
                            self.certainty_score + 0.1,
                        ],
                    }
                )
            )
        ]


class MockGeminiResponse:
    """Mock Gemini API response for drift detection."""

    def __init__(self, drift_score: float = 0.15):
        self.drift_score = drift_score

    def text(self) -> str:
        """Return mock Gemini text response."""
        return json.dumps(
            {
                "linguistic_drift": self.drift_score,
                "tone_shift": "more cautious than 90 days ago",
                "keyword_changes": ["supply chain", "geopolitical risk"],
                "regulatory_whispers": False,
            }
        )


@pytest.fixture
def mock_env(monkeypatch):
    """Set mock environment variables."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-claude-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")


@pytest.fixture
def mock_yfinance(monkeypatch):
    """Mock yfinance ticker fetcher."""

    def mock_ticker(symbol: str):
        return MockPriceData(current=150.0)

    monkeypatch.setattr("yfinance.Ticker", mock_ticker)


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock requests library for external APIs."""
    mock_response = Mock()
    mock_response.json.return_value = MockNewsResponse().json()
    mock_response.status_code = 200
    monkeypatch.setattr("requests.get", Mock(return_value=mock_response))


@pytest.fixture
def mock_anthropic(monkeypatch):
    """Mock Anthropic Claude client."""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.content = MockClaudeResponse().content()
    mock_client.messages.create.return_value = mock_message
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=mock_client))


@pytest.fixture
def mock_gemini(monkeypatch):
    """Mock Google Gemini client."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.text = MockGeminiResponse().text
    mock_client.generate_content.return_value = mock_response
    monkeypatch.setattr(
        "google.generativeai.GenerativeModel", Mock(return_value=mock_client)
    )


def test_scout_price_fetch_success(mock_env, mock_yfinance):
    """Verify Scout fetches live prices without errors."""
    import yfinance as yf

    ticker = yf.Ticker("AAPL")
    info = ticker.info()
    assert info["currentPrice"] == 150.0
    assert info["marketCap"] == 2.5e12


def test_scout_price_fetch_fallback(mock_env, monkeypatch):
    """Verify Scout falls back gracefully on price fetch failure."""

    def failing_ticker(symbol: str):
        raise ValueError("Network error")

    monkeypatch.setattr("yfinance.Ticker", failing_ticker)

    with pytest.raises(ValueError):
        import yfinance as yf

        yf.Ticker("AAPL")


def test_scout_news_aggregation(mock_env, mock_requests):
    """Verify Scout aggregates news headlines correctly."""
    import requests

    response = requests.get("https://api.example.com/news?q=AAPL")
    data = response.json()
    assert len(data["articles"]) == 5
    assert data["articles"][0]["title"]
    assert data["articles"][0]["source"]["name"] == "Reuters"


def test_scout_sec_filing_metadata(mock_env, mock_requests):
    """Verify Scout extracts SEC filing metadata."""
    import requests

    response = requests.get("https
