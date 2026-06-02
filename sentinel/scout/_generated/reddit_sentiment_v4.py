"""
Reddit sentiment scraper for Sentinel Scout pillar.

Fetches posts and comments from r/wallstreetbets, r/stocks, and r/investing,
analyzes sentiment signals (bullish/bearish keywords, upvote ratios, comment tone),
and outputs normalized SentimentSignal dataclasses for downstream Linguist analysis.

Uses PRAW (Reddit API wrapper) to stream live discussion and historical posts.
Implements rate-limiting and deduplication to avoid API exhaustion.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import sqlite3

import praw
import pandas as pd


@dataclass
class SentimentSignal:
    """Normalized sentiment signal for a ticker from Reddit discussion."""
    ticker: str
    source: str  # "reddit"
    subreddit: str  # e.g. "wallstreetbets"
    signal_type: str  # "post" or "comment"
    text: str
    sentiment_score: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    upvotes: int
    timestamp: datetime
    post_id: Optional[str] = None
    author: Optional[str] = None


class RedditSentimentScraper:
    """Scrapes Reddit sentiment signals from financial subreddits."""

    # Bullish indicators
    BULLISH_KEYWORDS = {
        r'\bto\s+the\s+moon\b': 2.0,
        r'\brocket\b': 1.5,
        r'\blong\b': 1.2,
        r'\bbullish\b': 1.5,
        r'\bhold\b': 0.8,
        r'\bhodl\b': 1.0,
        r'\bstrong\s+buy\b': 1.8,
        r'\blego\b': 1.0,
        r'\nlambo\b': 1.2,
        r'\btendie\b': 1.0,
        r'\btrend\s+up\b': 1.3,
        r'\bgain\b': 0.9,
    }

    # Bearish indicators
    BEARISH_KEYWORDS = {
        r'\bcrash\b': -1.8,
        r'\brip\b': -1.5,
        r'\bdump\b': -1.4,
        r'\bshort\b': -1.2,
        r'\bbearish\b': -1.5,
        r'\bsell\b': -0.8,
        r'\brisk\b': -0.6,
        r'\bloss\b': -1.0,
        r'\bread\s+flag\b': -1.4,
        r'\bbagholder\b': -1.6,
        r'\bpump\s+and\s+dump\b': -1.8,
        r'\bfud\b': -1.2,
    }

    def __init__(self, reddit_client: Optional[praw.Reddit] = None):
        """Initialize scraper with PRAW Reddit client."""
        if reddit_client is None:
            client_id = os.getenv("REDDIT_CLIENT_ID", "")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
            user_agent = os.getenv("REDDIT_USER_AGENT", "sentinel-sentiment-engine")
            if not client_id or not client_secret:
                raise ValueError(
                    "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in env"
                )
            reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )
        self.reddit = reddit_client
        self.subreddits = ["wallstreetbets", "stocks", "investing"]
        self._seen_ids = set()

    def _extract_tickers(self, text: str) -> set[str]:
        """Extract stock ticker symbols (all-caps, 1–5 chars) from text."""
        pattern = r'\$?([A-Z]{1,5})\b'
        matches = re.findall(pattern, text)
        # Filter out common words
        common_words = {"THE", "AND", "FOR", "ARE", "BUT", "ALL", "THIS", "THAT"}
        return {m for m in matches if m not in common_words}

    def _compute_sentiment_score(self, text: str) -> tuple[float, float]:
        """
        Compute sentiment score and confidence from text.

        Returns (sentiment_score, confidence) where sentiment_score is -1.0 to +1.0
        and confidence is 0.0 to 1.0.
        """
        text_lower = text.lower()
        bullish_score = 0.0
        bearish_score = 0.0
        bullish_matches = 0
        bearish_matches = 0

        for pattern, weight in self.BULLISH_KEYWORDS.items():
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            if matches > 0:
                bullish_score += weight * matches
                bullish_matches += matches

        for pattern, weight in self.BEARISH_KEYWORDS.items():
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            if matches > 0:
                bearish_score += abs(weight) * matches
                bearish_matches += matches

        # Normalize to -1.0 to +1.0
        total_signal = bullish_score - bearish_score
        max_possible = max(
            sum(self.BULLISH_KEYWORDS.values()),
            sum(abs(w) for w in self.BEARISH_KEYWORDS.values()),
        )
        sentiment = total_signal / max_possible if max_possible > 0 else 0.0
        sentiment = max(-1.0, min(1.0, sentiment))

        # Confidence based on signal intensity and text length
        signal_intensity = abs(sentiment)
        text_length_factor = min(1.0, len(text) / 500.0)
        confidence = (signal_intensity + text_length_factor) / 2.0

        return sentiment, confidence

    def scrape_subreddit_posts(
        self, subreddit_name: str, limit: int = 50
    ) -> list[SentimentSignal]:
        """Scrape recent posts from a subreddit and return sentiment signals."""
        signals = []
        try:
            sub = self.reddit.subreddit(subreddit_name)
            for post in sub.new(limit=limit):
                if post.id in self._seen_ids:
                    continue
                self._seen_ids.add(post.id)

                tickers = self._extract_tickers(post.title + " " + post.selftext)
                if not tickers:
                    continue

                sentiment, confidence = self._compute_sentiment_score(
                    post.title + " " + post.selftext
                )

                for ticker in tickers:
                    signal = SentimentSignal(
                        ticker=ticker,
                        source="reddit",
                        subreddit=subreddit_name,
                        signal_type="post",
                        text=post.title,
                        sentiment_score=sentiment,
                        confidence=confidence,
                        upvotes=post.score,
                        timestamp=datetime.fromtimestamp(post.created_utc),
                        post_id=post.id,
                        author=post.author.name if post.author else None,
                    )
                    signals.append(signal)
        except Exception as e:
            print(f"Error scraping {subreddit_name}: {e}")

        return signals

    def scrape_subreddit_comments(
        self, subreddit_name: str, post_limit: int = 10, comment_limit: int = 20
    ) -> list[SentimentSignal]:
        """Scrape recent comments from posts in a subreddit."""
        signals =
