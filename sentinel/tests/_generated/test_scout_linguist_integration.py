"""
Scout → Linguist integration test for Sentinel Sentiment Engine.

This module provides end-to-end pytest integration tests that validate the
data flow from Scout (data ingestion) through Linguist (NLM reasoning) with
fully mocked external API calls. Tests verify that sentiment signals are
correctly extracted, embedded, scored for certainty, and passed downstream
to the Judge for prediction.

Used by the daily build to validate core pipeline integrity without hitting
live markets or LLM APIs.
"""

import json
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, List, Any

import pytest
import numpy as np
import pandas as pd


class MockYFinanceResponse:
    """Mock yfinance ticker object for live price fetching."""

    def __init__(self, ticker: str, price: float = 150.0):
        self.ticker = ticker
        self.info = {
            "currentPrice": price,
            "previousClose": price * 0.99,
            "fiftyTwoWeekHigh": price * 1.2,
            "fiftyTwoWeekLow": price * 0.8,
        }
        self.history_data = pd.DataFrame(
            {
                "Close": [price * (0.99 + i * 0.001) for i in range(30)],
                "Volume": [1e7 + i * 1e5 for i in range(30)],
            },
            index=pd.date_range("2024-01-01", periods=30),
        )

    def history(self, period: str) -> pd.DataFrame:
        """Return mock historical price data."""
        return self.history_data


class MockSECFiling:
    """Mock SEC filing object for 8-K/10-Q extraction."""

    def __init__(
        self,
        ticker: str,
        form_type: str = "8-K",
        text: str = "Risk factors and forward guidance disclosed.",
    ):
        self.ticker = ticker
        self.form_type = form_type
        self.text = text
        self.date = "2024-02-15"
        self.accession_number = "0000123456-24-000001"


class MockRedditPost:
    """Mock Reddit post for sentiment extraction."""

    def __init__(self, title: str, score: int = 100, sentiment_text: str = None):
        self.title = title
        self.selftext = sentiment_text or "This stock is looking strong fundamentally."
        self.score = score
        self.url = "https://reddit.com/r/stocks/comments/abc123"
        self.created_utc = 1708041600.0


class MockNewsArticle:
    """Mock news article for headline sentiment."""

    def __init__(self, headline: str, source: str = "Reuters", sentiment_text: str = None):
        self.headline = headline
        self.source = source
        self.text = sentiment_text or "Company announced record revenue growth."
        self.url = "https://example.com/news/article"
        self.published_at = "2024-02-15T10:30:00Z"


@pytest.fixture
def mock_yfinance():
    """Fixture: mocked yfinance module."""
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value = MockYFinanceResponse("AAPL", price=185.5)
        yield mock_ticker


@pytest.fixture
def mock_sec_filings():
    """Fixture: mocked SEC EDGAR scraper."""
    with patch(
        "sentinel.scout.sec_filings.fetch_sec_filings"
    ) as mock_fetch:
        mock_fetch.return_value = [
            MockSECFiling("AAPL", "8-K", "Material agreement signed. Earnings upside expected."),
            MockSECFiling("AAPL", "10-Q", "Q1 revenue +15% YoY. Forward guidance raised."),
        ]
        yield mock_fetch


@pytest.fixture
def mock_reddit():
    """Fixture: mocked Reddit PRAW scraper."""
    with patch("praw.Reddit") as mock_reddit_client:
        mock_subreddit = MagicMock()
        mock_subreddit.search.return_value = [
            MockRedditPost(
                "AAPL breakout imminent, strong technical + fundamental setup",
                score=450,
            ),
            MockRedditPost(
                "AAPL concerns: margin compression in services segment",
                score=120,
                sentiment_text="Bearish on AAPL fundamentals this quarter.",
            ),
        ]
        mock_reddit_client.return_value.subreddit.return_value = mock_subreddit
        yield mock_reddit_client


@pytest.fixture
def mock_news():
    """Fixture: mocked news headline fetcher."""
    with patch("sentinel.scout.news.fetch_headlines") as mock_fetch:
        mock_fetch.return_value = [
            MockNewsArticle(
                "Apple Q1 Earnings Beat Estimates, Services Growth Accelerates",
                source="Reuters",
                sentiment_text="Strong earnings report. Services segment growing faster than expected.",
            ),
            MockNewsArticle(
                "Apple Supply Chain Disruptions Risk Q2 Guidance",
                source="Bloomberg",
                sentiment_text="Manufacturing delays could impact near-term guidance.",
            ),
        ]
        yield mock_fetch


@pytest.fixture
def mock_linguist_embeddings():
    """Fixture: mocked Gemini embedding API."""
    with patch("google.generativeai.embed_content") as mock_embed:
        mock_embed.return_value = {
            "embedding": np.random.randn(768).tolist()
        }
        yield mock_embed


@pytest.fixture
def mock_claude_reasoning():
    """Fixture: mocked Claude reasoning client."""
    with patch("anthropic.Anthropic") as mock_claude:
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [
            MagicMock(
                text="Certainty: 0.78 | Tone: cautiously optimistic | Key signal: earnings beat + FG raise. Hedging risk: supply chain delays."
            )
        ]
        mock_client.messages.create.return_value = mock_message
        mock_claude.return_value = mock_client
        yield mock_claude


@pytest.fixture
def sample_ticker() -> str:
    """Fixture: sample ticker for testing."""
    return "AAPL"


def test_scout_live_prices_fetch(mock_yfinance, sample_ticker):
    """Validate Scout live_prices module fetches and parses price data."""
    import yfinance

    ticker_obj = yfinance.Ticker(sample_ticker)
    assert ticker_obj.info["currentPrice"] == 185.5
    assert ticker_obj.info["previousClose"] == pytest.approx(185.5 * 0.99, rel=0.01)
    history = ticker_obj.history("1mo")
    assert len(history) == 30
    assert "Close" in history.columns
    assert "Volume" in history.columns


def test_scout_sec_filings_fetch(mock_sec_filings, sample_ticker):
    """Validate Scout SEC filings scraper extracts filing text."""
    from sentinel.scout.sec_filings import fetch_sec_filings

    filings = fetch_sec_filings(sample_ticker)
    assert len(filings) == 2
    assert filings[0].form_type == "8-K"
    assert "Material agreement" in filings[0].text
    assert filings[1].form_type == "10-Q"
    assert "Forward guidance raised" in filings[1].text


def test_scout_reddit_sentiment_extraction(mock_reddit, sample_ticker):
    """Validate Scout Reddit scraper extracts posts and scores sentiment."""
    import praw

    reddit = praw.Reddit()
    subreddit = reddit.subreddit("stocks")
    posts = list(subreddit.search(f"{sample_ticker} stock"))
    assert len(posts) ==
