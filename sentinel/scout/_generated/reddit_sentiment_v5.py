"""
Reddit sentiment scraper for Sentinel Scout pillar.

Uses PRAW to ingest discussions from r/wallstreetbets, r/stocks, and r/investing,
extracting sentiment signals (bullish/bearish intensity, ticker mentions, confidence)
and normalizing them into SentimentSignal dataclasses for consumption by Linguist
and Historian modules.

Designed for high-volume parsing; integrates with the main pipeline via
scout.py aggregation layer.
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
    """Normalized sentiment observation from Reddit discussions."""
    source: str  # "reddit"
    ticker: str
    sentiment: float  # -1.0 (bearish) to +1.0 (bullish)
    intensity: float  # 0.0 to 1.0, confidence/conviction magnitude
    text_sample: str  # excerpt for audit trail
    url: str  # permalink to source post/comment
    timestamp: datetime
    subreddit: str
    author: str
    score: int  # upvote count as proxy for agreement
    

class RedditSentimentScraper:
    """Scrapes Reddit discussions and extracts sentiment signals for target stocks."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
    TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5})\b")
    BULLISH_KEYWORDS = {
        "moon", "diamond hands", "rocket", "bull", "calls", "long",
        "bullish", "buy", "undervalued", "strong buy", "breakout",
        "pump", "squeeze", "gain", "profit", "lambo"
    }
    BEARISH_KEYWORDS = {
        "crash", "dump", "puts", "short", "bearish", "sell", "overvalued",
        "decline", "drop", "loss", "bear", "bag holder", "bagholder",
        "rug pull", "bankruptcy", "worthless"
    }

    def __init__(self):
        """Initialize PRAW client from environment credentials."""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0"),
        )

    def extract_tickers(self, text: str) -> list[str]:
        """Extract potential stock tickers from text using regex."""
        candidates = self.TICKER_PATTERN.findall(text.upper())
        # Filter out common non-ticker words
        exclude = {"EDIT", "TLDR", "UPDATE", "DISCLAIMER", "HOLD"}
        return [t for t in candidates if t not in exclude and len(t) >= 1]

    def compute_sentiment(self, text: str) -> tuple[float, float]:
        """
        Compute sentiment score and intensity from text.
        
        Returns (sentiment, intensity) where sentiment in [-1, 1] and intensity in [0, 1].
        """
        text_lower = text.lower()
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)

        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return 0.0, 0.0

        sentiment = (bullish_count - bearish_count) / total_signals
        intensity = min(1.0, total_signals / 10.0)  # cap at 10 signals for intensity
        return sentiment, intensity

    def fetch_posts(self, limit: int = 100) -> list[praw.reddit.Submission]:
        """Fetch recent posts from target subreddits."""
        posts = []
        for subreddit_name in self.SUBREDDITS:
            sub = self.reddit.subreddit(subreddit_name)
            for post in sub.new(limit=limit):
                posts.append(post)
        return posts

    def fetch_comments(self, posts: list[praw.reddit.Submission], depth: int = 2) -> list[praw.reddit.Comment]:
        """Recursively fetch top-level and child comments from posts."""
        comments = []
        for post in posts:
            post.comments.replace_more(limit=depth)
            for comment in post.comments.list()[:50]:  # limit per post
                comments.append(comment)
        return comments

    def process_text(
        self, text: str, author: str, score: int, url: str, 
        subreddit: str, timestamp: datetime
    ) -> list[SentimentSignal]:
        """Convert a single Reddit post/comment into SentimentSignal objects per ticker."""
        signals = []
        tickers = self.extract_tickers(text)
        if not tickers:
            return signals

        sentiment, intensity = self.compute_sentiment(text)
        text_sample = text[:200] if len(text) > 200 else text

        for ticker in set(tickers):  # deduplicate
            signal = SentimentSignal(
                source="reddit",
                ticker=ticker,
                sentiment=sentiment,
                intensity=intensity,
                text_sample=text_sample,
                url=url,
                timestamp=timestamp,
                subreddit=subreddit,
                author=author,
                score=score,
            )
            signals.append(signal)

        return signals

    def scrape_and_normalize(self, post_limit: int = 50) -> list[SentimentSignal]:
        """
        Main entry point: scrape Reddit, extract signals, normalize.
        
        Returns list of SentimentSignal objects aggregated across all subreddits.
        """
        all_signals = []

        posts = self.fetch_posts(limit=post_limit)
        for post in posts:
            signals = self.process_text(
                text=post.selftext or post.title,
                author=post.author.name if post.author else "[deleted]",
                score=post.score,
                url=post.permalink,
                subreddit=post.subreddit.display_name,
                timestamp=datetime.fromtimestamp(post.created_utc),
            )
            all_signals.extend(signals)

        comments = self.fetch_comments(posts, depth=2)
        for comment in comments:
            signals = self.process_text(
                text=comment.body,
                author=comment.author.name if comment.author else "[deleted]",
                score=comment.score,
                url=comment.permalink,
                subreddit=comment.subreddit.display_name,
                timestamp=datetime.fromtimestamp(comment.created_utc),
            )
            all_signals.extend(signals)

        return all_signals


def save_signals_to_sqlite(signals: list[SentimentSignal], db_path: str = "sentinel_reddit.db") -> None:
    """Persist SentimentSignal objects to SQLite for audit trail and RAG indexing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reddit_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            ticker TEXT,
            sentiment REAL,
            intensity REAL,
            text_sample TEXT,
            url TEXT,
            timestamp TEXT,
            subreddit TEXT,
            author TEXT,
            score INTEGER,
            created_at TEXT
        )
    """)
    
    for signal in signals:
        cursor.execute("""
            INSERT INTO reddit_signals
            (source, ticker, sentiment, intensity, text_sample, url, timestamp, subreddit, author, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.source,
            signal.ticker,
            signal.sentiment,
