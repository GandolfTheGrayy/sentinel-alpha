"""
Reddit sentiment scraper for Sentinel Scout pillar.

Uses PRAW to fetch posts and comments from r/wallstreetbets, r/stocks, and
r/investing, analyzing sentiment signals for equities. Outputs normalized
SentimentSignal dataclasses for ingestion by the Linguist pillar.

This module is called by sentinel/scout during daily runs to capture retail
sentiment velocity, bullish/bearish language patterns, and ticker mentions.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import praw
from anthropic import Anthropic


@dataclass
class SentimentSignal:
    """Normalized sentiment output from Scout modules."""

    ticker: str
    source: str  # "reddit"
    timestamp: datetime
    signal_type: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    raw_text: str
    mention_count: int
    score: float  # upvote differential or aggregate metric


def _get_reddit_client() -> praw.Reddit:
    """Instantiate authenticated PRAW Reddit client from environment."""
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        user_agent=os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0"),
    )
    return reddit


def _extract_tickers(text: str) -> set[str]:
    """Extract stock ticker symbols from text using regex and common patterns."""
    # Match $TICKER or standalone uppercase 1-5 letter sequences preceded by word boundary
    patterns = [
        r"\$([A-Z]{1,5})\b",  # $AAPL format
        r"\b([A-Z]{1,5})\s+(?:is|was|will|stock|shares|calls|puts)",  # AAPL is/was pattern
    ]
    tickers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tickers.update(m.upper() for m in matches)
    # Filter out common non-ticker uppercase words
    exclude = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "ALL", "CAN", "GET", "HAS"}
    return tickers - exclude


def _score_sentiment_with_claude(
    text: str, client: Anthropic
) -> tuple[str, float]:
    """
    Use Claude Sonnet to classify sentiment and assign confidence score.

    Returns (signal_type, confidence) where signal_type is 'bullish', 'bearish', or 'neutral'.
    """
    prompt = f"""Analyze the sentiment of this Reddit post/comment about stocks.
Respond with ONLY two lines:
LINE 1: bullish | bearish | neutral
LINE 2: 0.0 to 1.0 confidence

Text: {text[:500]}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = message.content[0].text.strip().split("\n")

    signal_type = "neutral"
    confidence = 0.5

    if len(response_text) >= 2:
        signal_type = response_text[0].strip().lower()
        if signal_type not in ("bullish", "bearish", "neutral"):
            signal_type = "neutral"
        try:
            confidence = float(response_text[1].strip())
            confidence = max(0.0, min(1.0, confidence))
        except ValueError:
            confidence = 0.5

    return signal_type, confidence


def fetch_reddit_sentiment(
    subreddits: Optional[list[str]] = None, limit: int = 100
) -> list[SentimentSignal]:
    """
    Scrape Reddit posts and comments, extract tickers, and score sentiment.

    Args:
        subreddits: List of subreddit names (default: wallstreetbets, stocks, investing).
        limit: Number of posts to fetch per subreddit (default: 100).

    Returns:
        List of SentimentSignal dataclasses normalized for downstream processing.
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]

    reddit = _get_reddit_client()
    claude_client = Anthropic()
    signals: list[SentimentSignal] = []

    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            # Fetch hot posts (balance freshness vs. relevance)
            for post in subreddit.hot(limit=limit):
                # Skip archived or deleted posts
                if post.archived or post.selftext == "[removed]":
                    continue

                text = f"{post.title} {post.selftext}"
                tickers = _extract_tickers(text)

                if not tickers:
                    continue

                # Score sentiment using Claude
                signal_type, confidence = _score_sentiment_with_claude(
                    text, claude_client
                )

                for ticker in tickers:
                    signal = SentimentSignal(
                        ticker=ticker,
                        source="reddit",
                        timestamp=datetime.utcfromtimestamp(post.created_utc),
                        signal_type=signal_type,
                        confidence=confidence,
                        raw_text=text[:300],
                        mention_count=1,
                        score=float(post.score),
                    )
                    signals.append(signal)

                # Optionally fetch top comments for richer signal
                post.comments.replace_more(limit=3)
                for comment in post.comments.list()[:5]:
                    if comment.body == "[removed]" or comment.body == "[deleted]":
                        continue

                    comment_tickers = _extract_tickers(comment.body)
                    if not comment_tickers:
                        continue

                    comment_signal_type, comment_confidence = (
                        _score_sentiment_with_claude(comment.body, claude_client)
                    )

                    for ticker in comment_tickers:
                        comment_signal = SentimentSignal(
                            ticker=ticker,
                            source="reddit",
                            timestamp=datetime.utcfromtimestamp(comment.created_utc),
                            signal_type=comment_signal_type,
                            confidence=comment_confidence,
                            raw_text=comment.body[:300],
                            mention_count=1,
                            score=float(comment.score),
                        )
                        signals.append(comment_signal)

        except Exception as e:
            print(f"Error fetching {subreddit_name}: {e}")
            continue

    return signals


def aggregate_sentiment_by_ticker(signals: list[SentimentSignal]) -> dict:
    """
    Aggregate signals by ticker into a summary dict with weighted sentiment.

    Returns a dict mapping ticker -> {
        'bullish_count': int,
        'bearish_count': int,
        'neutral_count': int,
        'avg_confidence': float,
        'aggregate_score': float,
        'last_update': datetime
    }
    """
    agg: dict = {}

    for signal in signals:
        ticker = signal.ticker
        if ticker not in agg:
            agg[ticker] = {
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "total_confidence": 0.0,
                "total_score": 0.0,
                "signal_count": 0,
                "last_update": signal.timestamp,
            }

        if signal.signal_type == "bullish":
            agg[ticker]["bullish_count"] += 1
        elif signal.signal_type == "bearish":
            agg[ticker]["bearish_count"] += 1
        else:
            agg[ticker]["neutral_
