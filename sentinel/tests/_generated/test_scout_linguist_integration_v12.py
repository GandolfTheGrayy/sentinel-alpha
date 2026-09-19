"""
Integration test suite for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) using mocked external calls (yfinance, news APIs,
SEC EDGAR, Reddit). Tests confirm that raw signals are correctly parsed,
embedded, and scored for certainty/hesitation before being passed to Judge.

Part of Sentinel's test harness; runs via pytest.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

# Sentinel modules
import sys
from pathlib import Path

# Add sentinel root to path for imports
sentinel_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(sentinel_root))

from sentinel.scout.live_prices import fetch_live_price
from sentinel.scout.news import fetch_news_headlines
from sentinel.scout.sec_filings import fetch_sec_filings
from sentinel.linguist.sample_score import score_certainty


@pytest.fixture
def mock_yfinance_data() -> Dict[str, Any]:
    """Mock live price data from yfinance."""
    return {
        "AAPL": {
            "price": 189.45,
            "change_pct": 2.34,
            "volume": 52_300_000,
            "timestamp": datetime.utcnow().isoformat(),
        },
        "TSLA": {
            "price": 242.17,
            "change_pct": -1.12,
            "volume": 128_400_000,
            "timestamp": datetime.utcnow().isoformat(),
        },
    }


@pytest.fixture
def mock_news_data() -> Dict[str, List[Dict[str, str]]]:
    """Mock news headlines for sentiment analysis."""
    return {
        "AAPL": [
            {
                "title": "Apple Q1 earnings beat expectations with strong iPhone sales",
                "source": "Reuters",
                "published_at": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "url": "https://example.com/aapl-earnings",
            },
            {
                "title": "Apple announces new AI chip partnership amid market competition",
                "source": "TechCrunch",
                "published_at": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
                "url": "https://example.com/aapl-ai",
            },
        ],
        "TSLA": [
            {
                "title": "Tesla stock falls on delivery miss in Q1 guidance",
                "source": "Bloomberg",
                "published_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "url": "https://example.com/tsla-miss",
            },
            {
                "title": "Musk hints at major product refresh coming next quarter",
                "source": "Reuters",
                "published_at": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                "url": "https://example.com/tsla-refresh",
            },
        ],
    }


@pytest.fixture
def mock_sec_filing_data() -> Dict[str, List[Dict[str, str]]]:
    """Mock SEC filing data from EDGAR."""
    return {
        "AAPL": [
            {
                "form_type": "8-K",
                "accession_number": "0000320193-24-000123",
                "filing_date": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "document_text": (
                    "Item 8.01 Other Events: Apple Inc. announced strategic "
                    "partnerships with leading cloud providers to enhance "
                    "services capabilities. Strong positive outlook for FY2024."
                ),
            },
            {
                "form_type": "10-Q",
                "accession_number": "0000320193-24-000456",
                "filing_date": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "document_text": (
                    "Net income increased 15% YoY. Operating margin expanded due to "
                    "cost optimization. Slight headwinds in Services segment offset "
                    "by Hardware strength."
                ),
            },
        ],
        "TSLA": [
            {
                "form_type": "8-K",
                "accession_number": "0001564590-24-000789",
                "filing_date": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "document_text": (
                    "Item 8.01 Other Events: Tesla Inc. faces increased competitive "
                    "pressure in EV market. Margin compression expected in Q2. "
                    "Supply chain normalized but demand uncertainty persists."
                ),
            },
        ],
    }


@pytest.fixture
def mock_reddit_sentiment() -> Dict[str, List[Dict[str, str]]]:
    """Mock Reddit/social sentiment data."""
    return {
        "AAPL": [
            {
                "source": "reddit_r_stocks",
                "text": "AAPL earnings were fantastic! AI pivot will be huge.",
                "upvotes": 1242,
                "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
            },
            {
                "source": "reddit_investing",
                "text": "Valuation still seems reasonable for a mature tech giant.",
                "upvotes": 856,
                "timestamp": (datetime.utcnow() - timedelta(hours=3)).isoformat(),
            },
        ],
        "TSLA": [
            {
                "source": "reddit_investing",
                "text": "Worried about Tesla's ability to maintain margins long-term.",
                "upvotes": 2104,
                "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            },
            {
                "source": "reddit_stocks",
                "text": "Too much hype, not enough delivery. Musk talk is all speculation.",
                "upvotes": 1876,
                "timestamp": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
            },
        ],
    }


class TestScoutLivePrice:
    """Test Scout module: live price fetching."""

    @patch("sentinel.scout.live_prices.yf.Ticker")
    def test_fetch_live_price_success(self, mock_ticker, mock_yfinance_data) -> None:
        """Verify live price fetch returns expected structure."""
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "currentPrice": mock_yfinance_data["AAPL"]["price"],
            "regularMarketChangePercent": mock_yfinance_data["AAPL"]["change_pct"],
            "volume": mock_yfinance_data["AAPL"]["volume"],
        }
        mock_ticker.return_value = mock_ticker_instance

        result = fetch_live_price("AAPL")

        assert result is not None
        assert result["price"] == pytest.approx(189.45)
        assert result["change_pct"] == pytest.approx(2.34)
        assert "timestamp" in result

    @patch("sentinel.scout.live_prices.yf.Ticker")
    def test_fetch_live_price_fallback_on_error(self, mock_ticker) -> None:
        """Verify fallback behavior when yfinance fails."""
        mock_ticker.side_effect = Exception("Network error")

        result = fetch_live_price("INVALID_TICKER")

        assert result is None or isinstance(result, dict)


class TestScoutNews:
    """Test Scout module: news headline ingestion."""

    @patch("sentinel.scout.news.requests.get")
    def test_fetch_news_headlines_success(self, mock_get, mock_news_data) -> None:
        """Verify news headline fetch returns
