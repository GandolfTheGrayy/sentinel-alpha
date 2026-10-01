"""
Reddit sentiment scraper for Sentinel Scout pillar.

Uses PRAW to ingest posts and comments from r/wallstreetbets, r/stocks, and
r/investing, analyzing sentiment signals tied to ticker mentions. Outputs
normalized SentimentSignal dataclass instances for downstream Linguist analysis.

Integrates with Scout's multi-source signal aggregation: raw sentiment scores
are weighted by subreddit authority, post recency, and upvote velocity, then
cross-referenced against historical baselines in Historian RAG.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import praw
from praw.exceptions import PrawException


@dataclass
class SentimentSignal:
    """Normalized sentiment signal for a single ticker from Reddit."""

    ticker: str
    """Stock ticker symbol (e.g., 'AAPL')."""

    source: str
    """Source identifier: 'reddit_wsb', 'reddit_stocks', 'reddit_investing'."""

    sentiment_score: float
    """Normalized sentiment in range [-1.0, 1.0]: -1=bearish, 0=neutral, 1=bullish."""

    confidence: float
    """Confidence score in range [0.0, 1.0] based on signal strength."""

    volume: int
    """Total mention count (posts + comments) for this ticker in sample window."""

    timestamp: datetime
    """UTC timestamp when signal was generated."""

    raw_text_sample: str
    """Representative snippet of source text for audit trail."""

    upvote_velocity: float
    """Average upvotes per mention; proxy for community engagement."""


class RedditSentimentScraper:
    """PRAW-based scraper for multi-subreddit ticker sentiment extraction."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
    LOOKBACK_HOURS = 24
    MIN_MENTION_THRESHOLD = 3

    def __init__(self) -> None:
        """Initialize PRAW client from environment credentials."""
        client_id = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        user_agent = os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0")

        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in environment"
            )

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        self.sentiment_lexicon = self._build_sentiment_lexicon()

    @staticmethod
    def _build_sentiment_lexicon() -> dict[str, float]:
        """Build simple lexicon for keyword-based sentiment scoring."""
        bullish = {
            "moon": 0.8,
            "rocket": 0.8,
            "lambo": 0.9,
            "bullish": 0.7,
            "buy": 0.6,
            "hold": 0.3,
            "squeeze": 0.7,
            "breakout": 0.6,
            "gap": 0.5,
            "calls": 0.6,
        }
        bearish = {
            "dump": -0.8,
            "crash": -0.9,
            "bearish": -0.7,
            "sell": -0.6,
            "rug": -0.9,
            "bankrupt": -1.0,
            "puts": -0.6,
            "shorting": -0.7,
            "tank": -0.8,
            "collapse": -0.9,
        }
        return {**bullish, **bearish}

    def _extract_tickers(self, text: str) -> set[str]:
        """Extract potential stock tickers from text using regex."""
        # Match $TICKER or standalone 1-5 letter uppercase words
        pattern = r"\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{1,5})(?:\s|$|[^\w])"
        matches = re.findall(pattern, text.upper())
        tickers = set()
        for match in matches:
            ticker = match[0] if match[0] else match[1]
            if len(ticker) >= 1 and len(ticker) <= 5:
                tickers.add(ticker)
        return tickers

    def _compute_sentiment_score(self, text: str) -> float:
        """Compute sentiment score [-1, 1] using lexicon-based approach."""
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        if not words:
            return 0.0

        scores = []
        for word in words:
            if word in self.sentiment_lexicon:
                scores.append(self.sentiment_lexicon[word])

        if not scores:
            return 0.0

        raw_score = sum(scores) / len(scores)
        return max(-1.0, min(1.0, raw_score))

    def _fetch_subreddit_mentions(self, subreddit_name: str) -> dict:
        """Fetch recent posts/comments from a subreddit and aggregate ticker mentions."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.LOOKBACK_HOURS)
        ticker_data = {}

        try:
            subreddit = self.reddit.subreddit(subreddit_name)

            # Fetch top recent posts
            for post in subreddit.new(limit=100):
                if post.created_utc < cutoff_time.timestamp():
                    continue

                post_text = f"{post.title} {post.selftext}"
                tickers = self._extract_tickers(post_text)
                sentiment = self._compute_sentiment_score(post_text)

                for ticker in tickers:
                    if ticker not in ticker_data:
                        ticker_data[ticker] = {
                            "scores": [],
                            "volumes": [],
                            "samples": [],
                            "upvotes": [],
                        }

                    ticker_data[ticker]["scores"].append(sentiment)
                    ticker_data[ticker]["volumes"].append(1)
                    ticker_data[ticker]["upvotes"].append(post.score)
                    ticker_data[ticker]["samples"].append(post_text[:200])

                # Fetch top-level comments on the post
                try:
                    post.comments.replace_more(limit=0)
                    for comment in post.comments[:50]:
                        if comment.created_utc < cutoff_time.timestamp():
                            continue

                        comment_text = comment.body
                        tickers = self._extract_tickers(comment_text)
                        sentiment = self._compute_sentiment_score(comment_text)

                        for ticker in tickers:
                            if ticker not in ticker_data:
                                ticker_data[ticker] = {
                                    "scores": [],
                                    "volumes": [],
                                    "samples": [],
                                    "upvotes": [],
                                }

                            ticker_data[ticker]["scores"].append(sentiment)
                            ticker_data[ticker]["volumes"].append(1)
                            ticker_data[ticker]["upvotes"].append(comment.score)
                            ticker_data[ticker]["samples"].append(comment_text[:200])
                except Exception:
                    # Comments may fail to load; skip gracefully
                    pass

        except PrawException as e:
            print(f"PRAW error fetching {subreddit_name}: {e}")

        return ticker_data

    def scrape(self) -> list[SentimentSignal]:
        """Scrape all target subreddits and return aggregated SentimentSignal list."""
        all_signals = {}

        for subreddit_name in self.SUBREDDITS:
            print(f"Scraping r/{subreddit_name}...")
            ticker_data = self._fetch_subreddit_mentions(subreddit_name)

            for ticker, data
