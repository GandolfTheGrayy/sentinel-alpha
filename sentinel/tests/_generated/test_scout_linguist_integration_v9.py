"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout) through
sentiment analysis (Linguist) with mocked external calls. It verifies that:
  1. Scout fetches and normalizes multi-source data (prices, news, SEC filings)
  2. Linguist analyzes text for certainty signals, linguistic drift, and regulatory whispers
  3. Outputs are correctly structured for downstream Judge consumption
  4. Error handling and fallback logic work as designed

This test suite is part of Sentinel's spine validation and runs daily as a
smoke test before live predictions begin.
"""

import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import pytest
from typing import Dict, List, Any


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def mock_price_data() -> Dict[str, Any]:
    """Return mock OHLCV data for a single ticker."""
    return {
        "AAPL": {
            "timestamp": datetime.now().isoformat(),
            "open": 150.0,
            "high": 152.5,
            "low": 149.5,
            "close": 151.2,
            "volume": 52_000_000,
            "change_percent": 0.8,
            "source": "yfinance"
        }
    }


@pytest.fixture
def mock_news_data() -> List[Dict[str, Any]]:
    """Return mock news headlines with timestamps."""
    return [
        {
            "ticker": "AAPL",
            "headline": "Apple beats Q4 earnings expectations with strong iPhone sales",
            "source": "Reuters",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "url": "https://example.com/apple-earnings"
        },
        {
            "ticker": "AAPL",
            "headline": "Apple stock faces headwinds from China market slowdown",
            "source": "Bloomberg",
            "timestamp": (datetime.now() - timedelta(hours=6)).isoformat(),
            "url": "https://example.com/apple-china"
        }
    ]


@pytest.fixture
def mock_sec_filing_data() -> List[Dict[str, Any]]:
    """Return mock SEC 8-K filing text snippets."""
    return [
        {
            "ticker": "AAPL",
            "form_type": "8-K",
            "filing_date": (datetime.now() - timedelta(days=1)).isoformat(),
            "text_snippet": (
                "Item 2.02 Results of Operations and Financial Condition. "
                "The company reported net income of $25.1 billion in Q4 2024, "
                "representing a 15% increase year-over-year. Revenue grew to $123.5 billion. "
                "Management maintains confidence in FY2025 guidance."
            ),
            "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193"
        }
    ]


@pytest.fixture
def scout_output(mock_price_data, mock_news_data, mock_sec_filing_data) -> Dict[str, Any]:
    """Return a complete Scout pipeline output."""
    return {
        "ticker": "AAPL",
        "timestamp": datetime.now().isoformat(),
        "prices": mock_price_data,
        "news": mock_news_data,
        "sec_filings": mock_sec_filing_data,
        "ingestion_status": "success",
        "sources_queried": ["yfinance", "newsapi", "sec_edgar"]
    }


# ============================================================================
# Scout Module Mock (simulates sentinel/scout/*)
# ============================================================================

class MockScout:
    """Mock Scout pipeline for testing without live external calls."""

    @staticmethod
    def fetch_live_prices(tickers: List[str]) -> Dict[str, Any]:
        """Simulate live price fetch from yfinance."""
        result = {}
        for ticker in tickers:
            result[ticker] = {
                "timestamp": datetime.now().isoformat(),
                "close": 151.2,
                "volume": 52_000_000,
                "source": "yfinance"
            }
        return result

    @staticmethod
    def fetch_news(tickers: List[str]) -> List[Dict[str, Any]]:
        """Simulate news headline fetch."""
        headlines = []
        for ticker in tickers:
            headlines.append({
                "ticker": ticker,
                "headline": f"Breaking: {ticker} reports strong quarterly results",
                "source": "Reuters",
                "timestamp": datetime.now().isoformat(),
                "url": "https://example.com"
            })
        return headlines

    @staticmethod
    def fetch_sec_filings(tickers: List[str], form_types: List[str] = None) -> List[Dict[str, Any]]:
        """Simulate SEC EDGAR filing fetch."""
        if form_types is None:
            form_types = ["8-K", "10-Q"]
        filings = []
        for ticker in tickers:
            filings.append({
                "ticker": ticker,
                "form_type": "8-K",
                "filing_date": datetime.now().isoformat(),
                "text_snippet": "Management maintains strong confidence in forward guidance.",
                "url": "https://www.sec.gov/cgi-bin/browse-edgar"
            })
        return filings


# ============================================================================
# Linguist Module Mock (simulates sentinel/linguist/*)
# ============================================================================

class MockLinguist:
    """Mock Linguist pipeline for sentiment and linguistic analysis."""

    @staticmethod
    def score_certainty(text: str) -> Dict[str, Any]:
        """
        Analyze text for certainty vs. hesitation signals.
        Returns scores for: confidence, uncertainty, neutral markers.
        """
        # Simple heuristic: count confidence/uncertainty keywords
        confidence_words = ["strong", "expects", "guidance", "beat", "exceeds"]
        uncertainty_words = ["may", "could", "risk", "challenge", "headwind"]

        confidence_score = sum(1 for word in confidence_words if word in text.lower())
        uncertainty_score = sum(1 for word in uncertainty_words if word in text.lower())

        total = confidence_score + uncertainty_score
        if total == 0:
            certainty_ratio = 0.5
        else:
            certainty_ratio = confidence_score / total

        return {
            "text_sample": text[:100],
            "confidence_score": confidence_score,
            "uncertainty_score": uncertainty_score,
            "certainty_ratio": round(certainty_ratio, 3),
            "signal": "bullish" if certainty_ratio > 0.6 else "bearish" if certainty_ratio < 0.4 else "neutral"
        }

    @staticmethod
    def detect_linguistic_drift(historical_texts: List[str], recent_text: str) -> Dict[str, Any]:
        """
        Detect tone shift in recent communications vs. historical baseline.
        Returns drift magnitude and direction (more/less cautious).
        """
        # Mock: simple word frequency comparison
        historical_avg_length = sum(len(t.split()) for t in historical_texts) / len(historical_texts) if historical_texts else 0
        recent_length = len(recent_text.split())

        drift_pct = ((recent_length - historical_avg_length) / historical_avg_length * 100) if historical_avg_length > 0 else 0

        return {
            "historical_avg_word_count": int(historical_avg_length),
            "recent_word_count": recent_length,
            "drift_percentage": round(drift_pct, 2),
            "direction": "expansion" if drift_pct > 5 else "contraction" if drift_pct < -5 else "stable",
            "anomaly": abs(drift_pct) > 20
        }

    @staticmethod
    def detect_regulatory_whispers(text
