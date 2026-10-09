"""
Reddit sentiment scraper for Sentinel Scout pillar.

Ingests posts and comments from r/wallstreetbets, r/stocks, and r/investing,
computing normalized sentiment signals via keyword extraction and valence scoring.
Output is a SentimentSignal dataclass cross-referenced by ticker and timestamp,
ready for Linguist analysis and Historian RAG enrichment.

Uses PRAW (Reddit API wrapper) for live data; avoids LLM calls here (Gemini/Claude
reserved for reasoning in Linguist/Judge). Sentiment is heuristic: keyword-based
polarity + confidence weighting by post karma and comment depth.
"""

import os
import re
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import praw
import pandas as pd


@dataclass
class SentimentSignal:
    """
    Normalized sentiment signal from Reddit sources.
    
    Attributes:
        ticker: Stock symbol (e.g., 'AAPL').
        source: 'reddit' identifier.
        subreddit: Name of subreddit (e.g., 'wallstreetbets').
        sentiment_score: Float in [-1.0, 1.0]; -1 = bearish, 0 = neutral, 1 = bullish.
        confidence: Float in [0.0, 1.0]; higher = more reliable signal.
        post_count: Number of posts analyzed for this ticker.
        comment_count: Number of comments analyzed.
        top_keywords: List of (keyword, frequency) tuples driving sentiment.
        timestamp: ISO 8601 datetime when signal was computed.
        raw_urls: List of top post URLs contributing to signal.
    """
    ticker: str
    source: str = "reddit"
    subreddit: str = ""
    sentiment_score: float = 0.0
    confidence: float = 0.0
    post_count: int = 0
    comment_count: int = 0
    top_keywords: List[tuple] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    raw_urls: List[str] = field(default_factory=list)


class RedditSentimentScraper:
    """Scrapes Reddit for sentiment signals across target subreddits."""
    
    # Sentiment lexicons (simple keyword-based valence).
    BULLISH_KEYWORDS = {
        'moon', 'hodl', 'diamond hands', 'strong buy', 'bullish', 'undervalued',
        'bargain', 'gem', 'breakout', 'rally', 'surge', 'pump', 'squeeze',
        'profit', 'gain', 'up', 'bull', 'lambo', 'tendies', 'catalyst', 'rocket'
    }
    
    BEARISH_KEYWORDS = {
        'crash', 'dump', 'bearish', 'overvalued', 'sell', 'short', 'decline',
        'loss', 'down', 'bear', 'red', 'rip', 'bagholder', 'dead', 'baghold',
        'risky', 'scam', 'bubble', 'collapse', 'tank', 'plunge', 'disaster'
    }
    
    TARGET_SUBREDDITS = ['wallstreetbets', 'stocks', 'investing']
    TICKER_PATTERN = re.compile(r'\b([A-Z]{1,5})\b')
    
    def __init__(self, lookback_hours: int = 24):
        """
        Initialize Reddit scraper with PRAW client from env credentials.
        
        Args:
            lookback_hours: Only ingest posts from the past N hours.
        """
        self.lookback_hours = lookback_hours
        self.reddit = self._init_praw()
    
    def _init_praw(self) -> praw.Reddit:
        """Initialize PRAW Reddit client from environment variables."""
        client_id = os.getenv("REDDIT_CLIENT_ID", "")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        user_agent = os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0 (by sentinel-bot)")
        
        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars required."
            )
        
        return praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
    
    def scrape_subreddit(self, subreddit_name: str) -> Dict[str, List[Dict]]:
        """
        Scrape a single subreddit for posts and comments in the lookback window.
        
        Returns dict: {
            'posts': [{'title', 'selftext', 'score', 'url', 'created_utc', ...}],
            'comments': [{'body', 'score', 'created_utc', ...}]
        }
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=self.lookback_hours)
        posts_data = []
        comments_data = []
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Fetch recent hot/top posts.
            for post in subreddit.hot(limit=100):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if post_time < cutoff_time:
                    break
                
                posts_data.append({
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'url': post.url,
                    'created_utc': post.created_utc,
                    'num_comments': post.num_comments,
                })
                
                # Fetch comments (limited to avoid rate limits).
                post.comments.replace_more(limit=5)
                for comment in post.comments.list()[:50]:
                    comment_time = datetime.utcfromtimestamp(comment.created_utc)
                    if comment_time < cutoff_time:
                        break
                    comments_data.append({
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'post_url': post.url,
                    })
        
        except Exception as e:
            print(f"Error scraping r/{subreddit_name}: {e}")
        
        return {'posts': posts_data, 'comments': comments_data}
    
    def extract_tickers(self, text: str) -> set:
        """Extract ticker symbols from text using regex pattern matching."""
        matches = self.TICKER_PATTERN.findall(text.upper())
        # Filter out common non-ticker words.
        exclude = {'THE', 'AND', 'FOR', 'YOU', 'ARE', 'NOT', 'CAN', 'HAS'}
        return {m for m in matches if m not in exclude and len(m) <= 5}
    
    def score_text_sentiment(self, text: str) -> tuple:
        """
        Score sentiment of text as (polarity, confidence).
        
        Returns:
            (float, float): polarity in [-1, 1], confidence in [0, 1].
        """
        text_lower = text.lower()
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return (0.0, 0.0)
        
        polarity = (bullish_count - bearish_count) / total
        confidence = min(total / 10.0, 1.0)  # Cap confidence at 1.
