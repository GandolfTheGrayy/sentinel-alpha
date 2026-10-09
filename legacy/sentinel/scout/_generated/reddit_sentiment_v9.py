"""
Reddit sentiment scraper for Sentinel Scout pillar.

Uses PRAW to ingest posts and comments from r/wallstreetbets, r/stocks, and r/investing,
then normalizes sentiment signals into SentimentSignal dataclass for downstream
Linguist and Historian pipelines. Handles rate limiting, deduplication, and
temporal weighting of discussion volume.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import praw
from praw.exceptions import PrawException


@dataclass
class SentimentSignal:
    """
    Normalized sentiment signal from Reddit sources.
    
    Attributes:
        ticker: Stock symbol (e.g., 'AAPL')
        subreddit: Source subreddit (e.g., 'wallstreetbets')
        signal_type: 'bullish', 'bearish', or 'neutral'
        confidence: Float 0.0–1.0 indicating strength of signal
        volume: Count of relevant posts/comments
        timestamp: When signal was collected
        sample_text: Representative quote from discussion
    """
    ticker: str
    subreddit: str
    signal_type: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    volume: int
    timestamp: datetime
    sample_text: str


class RedditSentimentScraper:
    """PRAW-based Reddit sentiment scraper for Sentinel."""

    def __init__(self) -> None:
        """Initialize PRAW Reddit API client from environment credentials."""
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "sentinel-scout/1.0"),
        )
        self.subreddits = ["wallstreetbets", "stocks", "investing"]
        self.bullish_keywords = {
            "buy", "long", "bullish", "moon", "diamond hands", "rocket", "undervalued",
            "dip", "hold", "strong", "upside", "rally", "breakout", "boom"
        }
        self.bearish_keywords = {
            "sell", "short", "bearish", "crash", "dump", "red", "loss", "tank",
            "weak", "downside", "collapse", "baghold", "overvalued", "rug pull"
        }

    def fetch_recent_posts(
        self, 
        subreddit_name: str, 
        limit: int = 100, 
        time_filter: str = "week"
    ) -> list[dict]:
        """
        Fetch recent posts from a subreddit using PRAW.
        
        Args:
            subreddit_name: Name of subreddit (e.g., 'wallstreetbets')
            limit: Max posts to retrieve (default 100)
            time_filter: Time window ('day', 'week', 'month', 'all')
        
        Returns:
            List of dicts with post metadata (title, score, timestamp)
        """
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                posts.append({
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.fromtimestamp(post.created_utc),
                    "url": post.url,
                })
            time.sleep(1)  # Respect rate limits
        except PrawException as e:
            print(f"[REDDIT] Error fetching {subreddit_name}: {e}")
        return posts

    def extract_ticker_mentions(self, text: str) -> list[str]:
        """
        Extract stock ticker symbols from text (simple regex-like heuristic).
        
        Looks for all-caps words of length 1–5 preceded by $ or in known ticker lists.
        
        Args:
            text: Raw text to parse
        
        Returns:
            List of detected ticker symbols
        """
        import re
        tickers = set()
        # Match $TICKER pattern
        for match in re.finditer(r"\$([A-Z]{1,5})\b", text):
            tickers.add(match.group(1))
        # Match common tickers in context (heuristic)
        common_tickers = {"AAPL", "MSFT", "TSLA", "GME", "AMC", "NVDA", "META", "GOOGL"}
        for ticker in common_tickers:
            if ticker in text.upper():
                tickers.add(ticker)
        return list(tickers)

    def score_sentiment(self, text: str) -> tuple[str, float]:
        """
        Score sentiment of text as 'bullish', 'bearish', or 'neutral' with confidence.
        
        Uses keyword overlap; confidence is ratio of sentiment words to total words.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (signal_type, confidence_float)
        """
        text_lower = text.lower()
        words = text_lower.split()
        
        bullish_count = sum(1 for w in words if any(b in w for b in self.bullish_keywords))
        bearish_count = sum(1 for w in words if any(b in w for b in self.bearish_keywords))
        
        total_sentiment = bullish_count + bearish_count
        if total_sentiment == 0:
            return ("neutral", 0.0)
        
        if bullish_count > bearish_count:
            confidence = min(1.0, bullish_count / max(1, len(words)))
            return ("bullish", confidence)
        elif bearish_count > bullish_count:
            confidence = min(1.0, bearish_count / max(1, len(words)))
            return ("bearish", confidence)
        else:
            return ("neutral", 0.3)

    def scrape_subreddit(
        self, 
        subreddit_name: str, 
        limit: int = 100
    ) -> list[SentimentSignal]:
        """
        Scrape sentiment signals from a single subreddit.
        
        Aggregates posts and comments, extracts tickers, scores sentiment,
        and groups by ticker and signal type.
        
        Args:
            subreddit_name: Target subreddit name
            limit: Max posts to ingest
        
        Returns:
            List of SentimentSignal objects
        """
        posts = self.fetch_recent_posts(subreddit_name, limit=limit)
        signals_by_ticker = {}
        
        for post in posts:
            combined_text = f"{post['title']} {post['selftext']}"
            tickers = self.extract_ticker_mentions(combined_text)
            signal_type, confidence = self.score_sentiment(combined_text)
            
            for ticker in tickers:
                key = (ticker, signal_type)
                if key not in signals_by_ticker:
                    signals_by_ticker[key] = {
                        "volume": 0,
                        "confidence_sum": 0.0,
                        "sample_texts": []
                    }
                signals_by_ticker[key]["volume"] += 1
                signals_by_ticker[key]["confidence_sum"] += confidence
                if len(signals_by_ticker[key]["sample_texts"]) < 3:
                    signals_by_ticker[key]["sample_texts"].append(post["title"][:100])
        
        signals = []
        for (ticker, signal_type), agg in signals_by_ticker.items():
            avg_confidence = agg["confidence_sum"] / max(1, agg["volume"])
            signals.append(SentimentSignal(
                ticker=ticker,
                subreddit=subreddit_name,
                signal_type=
