"""
Reddit Sentiment Scraper for Sentinel Scout.

Ingests sentiment signals from r/wallstreetbets, r/stocks, and r/investing
using PRAW (Python Reddit API Wrapper). Outputs normalized SentimentSignal
dataclasses tagged with source subreddit, post/comment volume, and aggregate
sentiment polarity. Feeds into Linguist pipeline for certainty scoring and
Historian RAG synthesis.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import praw
from praw.models import Submission, Comment


@dataclass
class SentimentSignal:
    """Normalized sentiment signal from Reddit."""

    ticker: str
    subreddit: str
    signal_type: str  # "post", "comment", "aggregate"
    polarity: float  # -1.0 to +1.0
    confidence: float  # 0.0 to 1.0
    text_sample: str
    post_count: int
    comment_count: int
    upvote_ratio: float
    timestamp: datetime
    url: Optional[str] = None


def _extract_tickers(text: str) -> list[str]:
    """Extract stock tickers ($AAPL, $TSLA) from text.
    
    Uses regex to find $-prefixed alphanumeric sequences (length 1-5).
    """
    matches = re.findall(r'\$([A-Z]{1,5})\b', text)
    return list(set(matches))


def _simple_sentiment(text: str) -> float:
    """Compute simple heuristic sentiment polarity from text.
    
    Returns float in [-1.0, 1.0]. Counts bullish keywords (moon, lambo, hodl)
    vs bearish (crash, dump, rekt). Normalizes by text length.
    """
    bullish_words = {
        'moon', 'lambo', 'hodl', 'diamond', 'hands', 'rocket', 'bull',
        'bullish', 'long', 'buy', 'pump', 'gains', 'tendies', 'up',
        'green', 'win', 'profit', 'surge', 'rally', 'momentum'
    }
    bearish_words = {
        'crash', 'dump', 'rekt', 'bearish', 'short', 'sell', 'drop',
        'red', 'loss', 'bagholder', 'rug', 'collapse', 'tank', 'down',
        'plunge', 'weak', 'fail', 'bubble', 'overvalued'
    }
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return 0.0
    
    bullish_count = sum(1 for w in words if w in bullish_words)
    bearish_count = sum(1 for w in words if w in bearish_words)
    
    total = bullish_count + bearish_count
    if total == 0:
        return 0.0
    
    return (bullish_count - bearish_count) / max(total, 1)


def scrape_reddit_sentiment(
    subreddits: list[str],
    limit: int = 100,
    time_filter: str = "day"
) -> list[SentimentSignal]:
    """Scrape Reddit sentiment from specified subreddits.
    
    Fetches top posts (by upvote_ratio) from given subreddits within time_filter
    window. Extracts tickers, computes sentiment polarity, aggregates comment
    sentiment. Returns list of SentimentSignal objects.
    
    Args:
        subreddits: List of subreddit names (e.g. ["wallstreetbets", "stocks"]).
        limit: Max posts per subreddit to fetch (default 100).
        time_filter: Timespan for "top" posts ("day", "week", "month").
    
    Returns:
        List of SentimentSignal dataclasses, one per (ticker, subreddit) pair.
    """
    reddit_id = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    reddit_user_agent = os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0")
    
    if not (reddit_id and reddit_secret):
        raise RuntimeError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET env vars")
    
    reddit = praw.Reddit(
        client_id=reddit_id,
        client_secret=reddit_secret,
        user_agent=reddit_user_agent
    )
    
    signals: list[SentimentSignal] = []
    ticker_data: dict[tuple[str, str], dict] = {}  # (ticker, subreddit) -> aggregates
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            posts = list(subreddit.top(time_filter=time_filter, limit=limit))
        except Exception as e:
            print(f"Error fetching r/{subreddit_name}: {e}")
            continue
        
        for post in posts:
            # Extract tickers from post title + selftext
            text = f"{post.title} {post.selftext}"
            tickers = _extract_tickers(text)
            
            # Skip if no tickers mentioned
            if not tickers:
                continue
            
            # Compute post sentiment
            post_sentiment = _simple_sentiment(text)
            
            # Aggregate comment sentiment (up to 20 top comments)
            comment_sentiments = []
            try:
                post.comments.replace_more(limit=0)
                for comment in list(post.comments)[:20]:
                    if isinstance(comment, Comment):
                        comment_sentiment = _simple_sentiment(comment.body)
                        comment_sentiments.append(comment_sentiment)
            except Exception as e:
                print(f"Error fetching comments for {post.id}: {e}")
            
            # Average sentiment across post + comments
            all_sentiments = [post_sentiment] + comment_sentiments
            avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0.0
            
            # Confidence based on engagement (upvote_ratio + comment count)
            confidence = min(1.0, (post.upvote_ratio * 0.5) + (len(comment_sentiments) / 100 * 0.5))
            
            # Record per ticker
            for ticker in tickers:
                key = (ticker, subreddit_name)
                if key not in ticker_data:
                    ticker_data[key] = {
                        "sentiments": [],
                        "post_count": 0,
                        "comment_count": 0,
                        "sample_text": "",
                        "url": post.url,
                        "upvote_ratio": post.upvote_ratio
                    }
                
                ticker_data[key]["sentiments"].append(avg_sentiment)
                ticker_data[key]["post_count"] += 1
                ticker_data[key]["comment_count"] += len(comment_sentiments)
                if not ticker_data[key]["sample_text"]:
                    ticker_data[key]["sample_text"] = text[:200]
        
    # Convert aggregates to SentimentSignal objects
    for (ticker, subreddit), data in ticker_data.items():
        avg_sentiment = sum(data["sentiments"]) / len(data["sentiments"])
        confidence = min(1.0, (data["upvote_ratio"] * 0.5) + (data["comment_count"] / 100 * 0.5))
        
        signal = SentimentSignal(
            ticker=ticker,
            subreddit=subreddit,
            signal_type="aggregate",
            polarity=avg_sentiment,
            confidence=confidence,
            text_sample=data["sample_
