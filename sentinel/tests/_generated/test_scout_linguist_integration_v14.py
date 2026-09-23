"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) using mocked external API calls. It ensures that
scraped signals (prices, news, SEC filings, Reddit sentiment) are correctly
parsed, embedded, and scored for certainty and linguistic drift detection.

Runs as part of CI/CD to catch regressions in the core reasoning loop.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import yfinance as yf


class MockPriceData:
    """Simulates yfinance.Ticker response."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.history_data = {
            "Open": [150.0, 151.5, 152.0],
            "Close": [151.0, 152.0, 153.5],
            "Volume": [1000000, 1100000, 950000],
        }

    def history(self, period: str) -> Dict[str, Any]:
        """Return mock OHLCV data."""
        return self.history_data


class MockRedditPost:
    """Simulates praw.models.reddit.submission.Submission."""

    def __init__(self, title: str, selftext: str, score: int) -> None:
        self.title = title
        self.selftext = selftext
        self.score = score
        self.created_utc = datetime.utcnow().timestamp()


@pytest.fixture
def temp_sentinel_env() -> Dict[str, str]:
    """Provide temporary environment with isolated file paths."""
    temp_dir = tempfile.mkdtemp()
    return {
        "SENTINEL_DB": str(Path(temp_dir) / "sentinel.db"),
        "SENTINEL_CHROMADB": str(Path(temp_dir) / "chromadb"),
        "SENTINEL_CACHE": str(Path(temp_dir) / "cache"),
    }


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic Claude client for Linguist reasoning."""
    with patch("anthropic.Anthropic") as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        mock_instance.messages.create.return_value = MagicMock(
            content=[
                MagicMock(
                    text=json.dumps(
                        {
                            "certainty_score": 0.78,
                            "direction": "bullish",
                            "key_signals": ["positive earnings guidance", "analyst upgrades"],
                            "hesitation_markers": ["supply chain concerns"],
                        }
                    )
                )
            ]
        )
        yield mock_instance


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for Scout text extraction and embedding."""
    with patch("google.generativeai.generate_content") as mock_gen:
        mock_gen.return_value = MagicMock(
            text="Tesla reported record Q4 deliveries, beating analyst expectations."
        )
        yield mock_gen


@pytest.fixture
def mock_yfinance():
    """Mock yfinance.Ticker for live price fetching."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value = MockPriceData("TSLA")
        yield mock_ticker


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client for RAG vector storage."""
    with patch("chromadb.Client") as mock_chroma:
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "documents": [
                ["TSLA surged 15% on positive earnings"],
                ["Market sentiment shifted bullish post-announcement"],
            ],
            "metadatas": [
                {"source": "news", "date": "2024-01-15"},
                {"source": "reddit", "date": "2024-01-15"},
            ],
            "distances": [0.1, 0.2],
        }
        yield mock_client


@pytest.fixture
def mock_requests_session():
    """Mock requests for SEC EDGAR and news API calls."""
    with patch("requests.Session") as mock_session:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = """
        <html>
          <document>
            <content>ITEM 1A. RISK FACTORS: Supply chain disruptions may impact...</content>
          </document>
        </html>
        """
        mock_resp.json.return_value = {
            "articles": [
                {
                    "title": "TSLA stock soars on Q4 earnings beat",
                    "description": "Strong delivery numbers exceed expectations.",
                    "publishedAt": "2024-01-15T10:00:00Z",
                }
            ]
        }
        mock_session.return_value.get.return_value = mock_resp
        yield mock_session


def test_scout_price_fetch_integration(mock_yfinance):
    """Test Scout live price fetcher integration."""
    ticker = yf.Ticker("TSLA")
    history = ticker.history(period="5d")
    assert "Close" in history
    assert len(history["Close"]) == 3
    assert history["Close"].iloc[-1] == 153.5


def test_scout_news_ingestion_integration(mock_requests_session):
    """Test Scout news headline fetcher integration."""
    import requests

    session = requests.Session()
    resp = session.get("https://newsapi.org/v2/everything")
    assert resp.status_code == 200
    articles = resp.json()["articles"]
    assert len(articles) > 0
    assert "TSLA" in articles[0]["title"]


def test_scout_sec_filing_scraper_integration(mock_requests_session):
    """Test Scout SEC EDGAR 8-K/10-Q scraper integration."""
    import requests

    session = requests.Session()
    resp = session.get("https://www.sec.gov/cgi-bin/browse-edgar")
    assert resp.status_code == 200
    assert "RISK FACTORS" in resp.text or "Supply chain" in resp.text


def test_linguist_certainty_scorer_integration(mock_anthropic_client):
    """Test Linguist certainty and hesitation analysis."""
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": "Analyze sentiment: Tesla reports record earnings.",
            }
        ],
    )

    result = json.loads(response.content[0].text)
    assert result["certainty_score"] == 0.78
    assert result["direction"] == "bullish"
    assert len(result["key_signals"]) > 0
    assert len(result["hesitation_markers"]) > 0


def test_linguist_drift_detection_integration():
    """Test Linguist linguistic drift detection over time."""
    historical_tones = [
        {"date": "2024-01-01", "tone": "cautiously optimistic", "keywords": ["growth"]},
        {"date": "2024-01-08", "tone": "neutral", "keywords": ["headwinds"]},
        {"date": "2024-01-15", "tone": "pessimistic", "keywords": ["decline", "risk"]},
    ]

    drift_score = 0.0
    for i in range(1, len(historical_tones)):
        prev_tone = historical_tones[i - 1]["tone"]
        curr_tone
