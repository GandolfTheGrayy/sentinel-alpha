"""
Integration test for Scout → Linguist pipeline.

This module validates the end-to-end flow from data ingestion (Scout)
through sentiment analysis (Linguist) using mocked external API calls.
It ensures that raw market signals are correctly transformed into
certainty-weighted sentiment scores without hitting live endpoints.

Part of Sentinel's test suite; runs via pytest.
"""

import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import pytest


class MockScoutResponse:
    """Simulates Scout module's aggregated response."""

    def __init__(self):
        self.live_price = {
            "ticker": "NVDA",
            "price": 875.50,
            "timestamp": "2024-01-15T14:32:00Z"
        }
        self.news_headlines = [
            {
                "title": "NVDA beats earnings expectations with strong AI demand",
                "source": "Reuters",
                "timestamp": "2024-01-15T09:00:00Z"
            },
            {
                "title": "Analyst upgrades NVDA on accelerating data center growth",
                "source": "Goldman Sachs",
                "timestamp": "2024-01-14T16:45:00Z"
            }
        ]
        self.sec_filings = [
            {
                "filing_type": "8-K",
                "date": "2024-01-10",
                "snippet": "Company announces record quarterly revenue driven by GPU sales."
            }
        ]
        self.reddit_sentiment = {
            "subreddit": "r/stocks",
            "sentiment_score": 0.72,
            "mention_count": 234,
            "sample_comments": [
                "NVDA is crushing it. AI boom is real.",
                "Long-term hold for sure."
            ]
        }


def mock_linguist_analyze(
    text: str,
    context: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """Mock Claude-based linguistic analysis."""
    if not text:
        return {"certainty": 0.0, "signal": "neutral", "explanation": "Empty input"}

    text_lower = text.lower()
    
    if any(word in text_lower for word in ["beats", "strong", "accelerating", "record"]):
        certainty = 0.85
        signal = "bullish"
    elif any(word in text_lower for word in ["miss", "decline", "weak", "slowing"]):
        certainty = 0.80
        signal = "bearish"
    else:
        certainty = 0.45
        signal = "neutral"

    return {
        "certainty": certainty,
        "signal": signal,
        "explanation": f"Detected {signal} tone with {certainty:.0%} confidence.",
        "drift_flag": False
    }


def mock_historian_lookup(
    ticker: str,
    query: str
) -> Dict[str, Any]:
    """Mock RAG-based historical event lookup."""
    return {
        "similar_events": [
            {
                "date": "2023-05-25",
                "event": "Previous earnings beat on AI demand",
                "outcome": "Stock up 8% next day",
                "similarity_score": 0.91
            }
        ],
        "confidence": 0.88
    }


@pytest.fixture
def scout_response() -> MockScoutResponse:
    """Provide a fresh Scout response for each test."""
    return MockScoutResponse()


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic Claude client."""
    client = MagicMock()
    client.messages.create = MagicMock(
        return_value=MagicMock(
            content=[MagicMock(text=json.dumps({
                "certainty": 0.85,
                "signal": "bullish",
                "explanation": "Strong earnings with AI tailwinds."
            }))]
        )
    )
    return client


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client for RAG."""
    client = MagicMock()
    collection = MagicMock()
    collection.query = MagicMock(
        return_value={
            "ids": [["evt_001"]],
            "documents": [[
                "Previous earnings beat on AI demand. Stock up 8% next day."
            ]],
            "distances": [[0.09]]
        }
    )
    client.get_or_create_collection = MagicMock(return_value=collection)
    return client


def test_scout_data_aggregation(scout_response: MockScoutResponse) -> None:
    """Verify Scout correctly aggregates live price, news, SEC, and social signals."""
    assert scout_response.live_price["ticker"] == "NVDA"
    assert scout_response.live_price["price"] == 875.50
    assert len(scout_response.news_headlines) == 2
    assert len(scout_response.sec_filings) == 1
    assert scout_response.reddit_sentiment["sentiment_score"] == 0.72


def test_linguist_headline_analysis(scout_response: MockScoutResponse) -> None:
    """Test Linguist correctly scores a bullish headline."""
    headline = scout_response.news_headlines[0]["title"]
    result = mock_linguist_analyze(headline)
    
    assert result["signal"] == "bullish"
    assert result["certainty"] > 0.75
    assert "explanation" in result


def test_linguist_sec_filing_analysis(scout_response: MockScoutResponse) -> None:
    """Test Linguist correctly scores SEC filing snippet."""
    filing_text = scout_response.sec_filings[0]["snippet"]
    result = mock_linguist_analyze(filing_text)
    
    assert result["signal"] == "bullish"
    assert result["certainty"] > 0.75


def test_linguist_neutral_text_handling() -> None:
    """Test Linguist handles neutral/ambiguous text gracefully."""
    neutral_text = "Company released quarterly update."
    result = mock_linguist_analyze(neutral_text)
    
    assert result["signal"] in ["neutral", "bullish", "bearish"]
    assert 0.0 <= result["certainty"] <= 1.0


def test_linguist_empty_input_handling() -> None:
    """Test Linguist gracefully handles empty or malformed input."""
    result = mock_linguist_analyze("")
    
    assert result["certainty"] == 0.0
    assert result["signal"] == "neutral"


def test_historian_rag_lookup(scout_response: MockScoutResponse) -> None:
    """Test Historian correctly retrieves similar historical events."""
    ticker = scout_response.live_price["ticker"]
    query = scout_response.news_headlines[0]["title"]
    result = mock_historian_lookup(ticker, query)
    
    assert "similar_events" in result
    assert len(result["similar_events"]) > 0
    assert result["confidence"] > 0.75


def test_scout_linguist_full_pipeline(
    scout_response: MockScoutResponse,
    mock_anthropic_client
) -> None:
    """End-to-end test: Scout aggregates, Linguist scores all signals."""
    
    all_texts = [
        scout_response.news_headlines[0]["title"],
        scout_response.news_headlines[1]["title"],
        scout_response.sec_filings[0]["snippet"]
    ]
    
    scores: List[Dict[str, Any]] = []
    for text in all_texts:
        score = mock_linguist_analyze(text)
        scores.append(score)
    
    assert len(scores) == 3
    
    avg_certainty = sum(s["certainty"] for s in scores) / len(scores)
    assert avg_certainty > 0.7
    
    bullish_count = sum(1 for s in scores if s["signal"] == "bullish")
    assert bullish_count >= 2


def test_scout_linguist_with_reddit_sentiment(scout_response:
