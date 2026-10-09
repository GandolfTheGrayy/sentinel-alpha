"""
Reddit sentiment scraper for Sentinel Scout pillar.

Fetches comments and posts from r/wallstreetbets, r/stocks, and r/investing,
extracts ticker mentions, and computes normalized sentiment signals using
lexicon-based analysis. Outputs SentimentSignal dataclasses for downstream
Linguist reasoning and RAG enrichment.

Uses PRAW (Reddit API wrapper) for data ingestion. Sentiment scoring via
simple polarity lexicon (approved packages only).
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import praw
import pandas as pd


@dataclass
class SentimentSignal:
    """Normalized sentiment signal for a ticker from Reddit sources."""

    ticker: str
    source: str  # e.g., "reddit:wallstreetbets"
    polarity: float  # [-1.0, 1.0] range
    mention_count: int
    sample_text: str  # Example post/comment
    timestamp: datetime
    confidence: float  # [0.0, 1.0] based on mention volume and recency


def _get_reddit_client() -> praw.Reddit:
    """Initialize PRAW Reddit client from environment variables."""
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "sentinel-scout-v1")

    if not client_id or not client_secret:
        raise ValueError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars required"
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _extract_tickers(text: str) -> list[str]:
    """Extract potential stock tickers from text (e.g., $TSLA, AAPL)."""
    # Match $TICKER or standalone 1-5 letter uppercase words preceded by context
    patterns = [
        r"\$([A-Z]{1,5})\b",  # $TSLA format
        r"\b([A-Z]{1,4})\s+(?:stock|call|put|shares|position|yolo)",  # contextual
    ]
    tickers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        tickers.update(m.upper() for m in matches)
    return list(tickers)


def _compute_polarity(text: str) -> float:
    """Simple lexicon-based polarity scoring [-1.0, 1.0]."""
    positive_words = {
        "bull",
        "moon",
        "lambo",
        "diamond",
        "hands",
        "to",
        "the",
        "moon",
        "win",
        "rocket",
        "bullish",
        "long",
        "buy",
        "undervalued",
        "gem",
        "tendies",
    }
    negative_words = {
        "bear",
        "crash",
        "dump",
        "rekt",
        "loss",
        "bag",
        "holder",
        "bearish",
        "short",
        "sell",
        "overvalued",
        "rug",
        "pull",
        "baghold",
    }

    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    total = pos_count + neg_count

    if total == 0:
        return 0.0

    polarity = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, polarity))


def fetch_reddit_sentiment(
    subreddits: Optional[list[str]] = None,
    limit_per_subreddit: int = 100,
    comment_limit_per_post: int = 10,
) -> list[SentimentSignal]:
    """
    Fetch and analyze sentiment signals from specified Reddit subreddits.

    Args:
        subreddits: List of subreddit names (default: wallstreetbets, stocks, investing)
        limit_per_subreddit: Max posts to fetch per subreddit
        comment_limit_per_post: Max comments to analyze per post

    Returns:
        List of SentimentSignal dataclasses, one per unique ticker.
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]

    reddit = _get_reddit_client()
    ticker_data = {}  # ticker -> {polarity: list, mentions: int, sample: str, source: str}

    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Fetch recent "hot" posts
            for post in subreddit.hot(limit=limit_per_subreddit):
                post_text = f"{post.title} {post.selftext}"
                tickers = _extract_tickers(post_text)
                post_polarity = _compute_polarity(post_text)

                for ticker in tickers:
                    if ticker not in ticker_data:
                        ticker_data[ticker] = {
                            "polarities": [],
                            "mention_count": 0,
                            "sample_text": post_text[:200],
                            "source": f"reddit:{subreddit_name}",
                        }
                    ticker_data[ticker]["polarities"].append(post_polarity)
                    ticker_data[ticker]["mention_count"] += 1

                # Fetch comments on post
                post.comments.replace_more(limit=0)
                for comment in post.comments.list()[:comment_limit_per_post]:
                    comment_text = comment.body
                    comment_tickers = _extract_tickers(comment_text)
                    comment_polarity = _compute_polarity(comment_text)

                    for ticker in comment_tickers:
                        if ticker not in ticker_data:
                            ticker_data[ticker] = {
                                "polarities": [],
                                "mention_count": 0,
                                "sample_text": comment_text[:200],
                                "source": f"reddit:{subreddit_name}",
                            }
                        ticker_data[ticker]["polarities"].append(comment_polarity)
                        ticker_data[ticker]["mention_count"] += 1

        except Exception as e:
            print(f"Error fetching r/{subreddit_name}: {e}")
            continue

    # Convert aggregated data to SentimentSignal objects
    signals = []
    now = datetime.utcnow()

    for ticker, data in ticker_data.items():
        if not data["polarities"]:
            continue

        avg_polarity = sum(data["polarities"]) / len(data["polarities"])
        mention_count = data["mention_count"]

        # Confidence based on mention volume: log scale capped at 1.0
        confidence = min(1.0, (mention_count / 10.0) ** 0.5)

        signal = SentimentSignal(
            ticker=ticker,
            source=data["source"],
            polarity=avg_polarity,
            mention_count=mention_count,
            sample_text=data["sample_text"],
            timestamp=now,
            confidence=confidence,
        )
        signals.append(signal)

    return sorted(signals, key=lambda s: s.confidence, reverse=True)


def reddit_sentiment_to_dataframe(
    signals: list[SentimentSignal],
) -> pd.DataFrame:
    """Convert list of SentimentSignal objects to pandas DataFrame."""
    return pd.DataFrame(
        [
            {
                "ticker": s.ticker,
                "source": s.source,
                "polarity": s.polarity,
                "mention_count": s
