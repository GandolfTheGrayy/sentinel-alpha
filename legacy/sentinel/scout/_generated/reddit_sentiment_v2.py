"""
Reddit sentiment scraper for Sentinel Scout pillar.

Fetches posts and comments from r/wallstreetbets, r/stocks, and r/investing
using PRAW (Python Reddit API Wrapper). Analyzes sentiment signals and outputs
normalized SentimentSignal dataclass instances for downstream Linguist analysis.

Integrates with the Scout data ingestion layer to provide real-time retail
investor sentiment as a niche signal cross-referenced against price movements.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import sqlite3

import praw
import numpy as np


@dataclass
class SentimentSignal:
    """Normalized sentiment signal from Reddit analysis."""
    ticker: str
    source: str
    signal_type: str
    score: float
    confidence: float
    sample_size: int
    timestamp: datetime
    raw_data: dict


class RedditSentimentScraper:
    """Scrapes and analyzes sentiment from Reddit financial communities."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
    SENTIMENT_KEYWORDS = {
        "bullish": ["buy", "moon", "rocket", "bullish", "long", "calls", "pump"],
        "bearish": ["sell", "crash", "dump", "bearish", "short", "puts", "dump"],
    }

    def __init__(self) -> None:
        """Initialize PRAW Reddit API client from environment variables."""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "sentinel-sentiment-engine/1.0")

        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in environment"
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def fetch_posts(
        self, subreddit: str, limit: int = 100, time_filter: str = "day"
    ) -> list[dict]:
        """Fetch top posts from a subreddit within a time window.

        Args:
            subreddit: Subreddit name (without r/).
            limit: Maximum posts to fetch.
            time_filter: Time window ('day', 'week', 'month', 'all').

        Returns:
            List of post dictionaries with metadata.
        """
        posts = []
        try:
            sub = self.reddit.subreddit(subreddit)
            for post in sub.top(time_filter=time_filter, limit=limit):
                posts.append(
                    {
                        "title": post.title,
                        "selftext": post.selftext,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "created_utc": datetime.fromtimestamp(post.created_utc),
                        "url": post.url,
                    }
                )
        except Exception as e:
            print(f"Error fetching posts from r/{subreddit}: {e}")

        return posts

    def extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols (uppercase 1-5 char sequences) from text.

        Args:
            text: Text to parse for ticker mentions.

        Returns:
            List of unique ticker symbols found.
        """
        import re

        ticker_pattern = r"\b[A-Z]{1,5}\b"
        candidates = re.findall(ticker_pattern, text)

        # Filter out common non-ticker words
        blacklist = {
            "AND",
            "THE",
            "FOR",
            "ARE",
            "BUT",
            "NOT",
            "ALL",
            "OUT",
            "CAN",
            "GET",
            "HAS",
            "HER",
            "HIS",
            "HOW",
            "ITS",
            "MAY",
            "NEW",
            "NOW",
            "OLD",
            "ONE",
            "OUR",
            "OUT",
            "OWN",
            "SAY",
            "SHE",
            "TOO",
            "TWO",
            "USE",
            "WAY",
            "WHO",
            "WHY",
            "YES",
        }

        return [t for t in set(candidates) if t not in blacklist]

    def analyze_sentiment(self, text: str) -> tuple[float, int]:
        """Compute sentiment score and keyword count from text.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (sentiment_score [-1, 1], keyword_count).
        """
        text_lower = text.lower()
        bullish_count = sum(
            text_lower.count(kw) for kw in self.SENTIMENT_KEYWORDS["bullish"]
        )
        bearish_count = sum(
            text_lower.count(kw) for kw in self.SENTIMENT_KEYWORDS["bearish"]
        )

        total_keywords = bullish_count + bearish_count
        if total_keywords == 0:
            return 0.0, 0

        sentiment_score = (bullish_count - bearish_count) / total_keywords
        return np.clip(sentiment_score, -1.0, 1.0), total_keywords

    def scrape_and_aggregate(
        self, days: int = 1, limit_per_sub: int = 100
    ) -> dict[str, SentimentSignal]:
        """Aggregate Reddit sentiment across subreddits for all mentioned tickers.

        Args:
            days: Number of days to lookback.
            limit_per_sub: Post limit per subreddit.

        Returns:
            Dictionary mapping ticker -> SentimentSignal.
        """
        time_filter = "day" if days == 1 else ("week" if days <= 7 else "month")
        ticker_sentiments: dict[str, list[float]] = {}
        ticker_posts: dict[str, int] = {}
        timestamp = datetime.utcnow()

        for subreddit in self.SUBREDDITS:
            posts = self.fetch_posts(subreddit, limit=limit_per_sub, time_filter=time_filter)

            for post in posts:
                combined_text = f"{post['title']} {post['selftext']}"
                tickers = self.extract_tickers(combined_text)
                sentiment, keyword_count = self.analyze_sentiment(combined_text)

                if tickers and keyword_count > 0:
                    for ticker in tickers:
                        if ticker not in ticker_sentiments:
                            ticker_sentiments[ticker] = []
                            ticker_posts[ticker] = 0

                        ticker_sentiments[ticker].append(sentiment)
                        ticker_posts[ticker] += 1

        signals: dict[str, SentimentSignal] = {}
        for ticker, scores in ticker_sentiments.items():
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
            confidence = min(1.0, len(scores) / 100.0) * (1.0 - std_score)

            signals[ticker] = SentimentSignal(
                ticker=ticker,
                source="reddit",
                signal_type="retail_sentiment",
                score=mean_score,
                confidence=max(0.0, confidence),
                sample_size=len(scores),
                timestamp=timestamp,
                raw_data={
                    "subreddits": self.SUBREDDITS,
                    "post_count": ticker_posts[ticker],
                    "mean_sentiment": mean_score,
                    "std_sentiment": std_score,
                },
            )

        return signals

    def persist_signals(
        self, signals: dict[str, SentimentSignal], db_path: str = ":memory:"
    ) -> None:
        """Store sentiment signals in SQLite for
