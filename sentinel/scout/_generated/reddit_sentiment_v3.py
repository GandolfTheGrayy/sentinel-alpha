"""
Reddit sentiment scraper for Sentinel Scout pillar.

Ingests posts and comments from r/wallstreetbets, r/stocks, and r/investing,
extracting ticker mentions and computing normalized sentiment signals via
Gemini extraction + basic lexical scoring. Outputs SentimentSignal dataclass
instances for cross-referencing with historical market data in Historian.

Uses PRAW (Reddit API client) for data collection and Gemini for high-volume
mention extraction and context parsing.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import praw
from google.generativeai import GenerativeModel


@dataclass
class SentimentSignal:
    """
    Normalized sentiment signal for a ticker extracted from Reddit.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL").
        source: Data source identifier (e.g., "reddit:wallstreetbets").
        timestamp: UTC datetime of signal acquisition.
        sentiment_score: Float in [-1.0, 1.0]; -1=strong bearish, 0=neutral, 1=strong bullish.
        mention_count: Integer count of mentions in scraped posts/comments.
        context_snippet: Representative text excerpt (up to 200 chars).
        confidence: Float in [0.0, 1.0] indicating scoring reliability.
    """
    ticker: str
    source: str
    timestamp: datetime
    sentiment_score: float
    mention_count: int
    context_snippet: str
    confidence: float


def _extract_tickers(text: str) -> list[str]:
    """Extract potential stock tickers (all-caps 1-5 char symbols with $) from text."""
    # Match $SYMBOL or standalone SYMBOL patterns common on Reddit
    pattern = r'\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{1,5})(?:\s|$|[.,!?])'
    matches = re.findall(pattern, text)
    tickers = [m[0] or m[1] for m in matches if m[0] or m[1]]
    # Filter to known ticker-like symbols (simple heuristic: no common words)
    common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
    return [t for t in tickers if t not in common_words]


def _sentiment_lexical_score(text: str) -> float:
    """
    Compute basic sentiment score via lexical signal (word frequency).
    
    Returns float in [-1.0, 1.0].
    """
    bullish = r'\b(moon|lambo|tendies|bull|bullish|long|buy|strong|gains|profit|up|rocket|🚀|win|beat|crush|excellent|awesome)\b'
    bearish = r'\b(crash|dump|bear|bearish|short|sell|weak|loss|down|rekt|bad|terrible|awful|miss|tank|doomed)\b'
    
    text_lower = text.lower()
    bull_count = len(re.findall(bullish, text_lower))
    bear_count = len(re.findall(bearish, text_lower))
    
    total = bull_count + bear_count
    if total == 0:
        return 0.0
    
    return (bull_count - bear_count) / total


def fetch_reddit_sentiment(
    subreddits: list[str] | None = None,
    hours_back: int = 24,
    post_limit: int = 100,
    comment_limit: int = 50
) -> list[SentimentSignal]:
    """
    Fetch sentiment signals from specified Reddit subreddits.
    
    Args:
        subreddits: List of subreddit names (default: ['wallstreetbets', 'stocks', 'investing']).
        hours_back: Lookback window in hours (default 24).
        post_limit: Max posts to scrape per subreddit.
        comment_limit: Max top comments per post to analyze.
    
    Returns:
        List of SentimentSignal instances, one per (ticker, source) pair.
    """
    if subreddits is None:
        subreddits = ['wallstreetbets', 'stocks', 'investing']
    
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID', ''),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET', ''),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'sentinel-scout-v1')
    )
    
    signals: dict[tuple[str, str], list[str]] = {}  # (ticker, source) -> [contexts]
    mention_counts: dict[tuple[str, str], int] = {}
    
    cutoff = datetime.utcnow() - timedelta(hours=hours_back)
    
    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            source = f"reddit:{sub_name}"
            
            for post in subreddit.hot(limit=post_limit):
                if datetime.utcfromtimestamp(post.created_utc) < cutoff:
                    continue
                
                post_text = f"{post.title} {post.selftext}"
                tickers = _extract_tickers(post_text)
                
                for ticker in tickers:
                    key = (ticker, source)
                    if key not in signals:
                        signals[key] = []
                        mention_counts[key] = 0
                    mention_counts[key] += 1
                    # Store snippet (first 200 chars of title for context)
                    signals[key].append(post.title[:200])
                
                # Top comments
                post.comments.replace_more(limit=0)
                for comment in list(post.comments)[:comment_limit]:
                    tickers = _extract_tickers(comment.body)
                    for ticker in tickers:
                        key = (ticker, source)
                        if key not in signals:
                            signals[key] = []
                            mention_counts[key] = 0
                        mention_counts[key] += 1
                        signals[key].append(comment.body[:200])
        
        except Exception as e:
            print(f"Error scraping {sub_name}: {e}")
            continue
    
    # Aggregate and score
    results: list[SentimentSignal] = []
    for (ticker, source), contexts in signals.items():
        # Combine contexts for sentiment analysis
        combined_text = ' '.join(contexts[:10])  # Use first 10 contexts
        
        # Lexical sentiment (fast baseline)
        lex_score = _sentiment_lexical_score(combined_text)
        
        # Gemini-enhanced extraction of nuanced sentiment (optional, higher confidence)
        gemini_score = None
        try:
            model = GenerativeModel('gemini-3.1-flash-lite-preview')
            prompt = f"""Analyze sentiment toward ${ticker} in this Reddit text. 
Reply with ONLY a float between -1.0 (strong bearish) and 1.0 (strong bullish), e.g., 0.35

Text: {combined_text[:1000]}"""
            response = model.generate_content(prompt)
            try:
                gemini_score = float(response.text.strip())
                gemini_score = max(-1.0, min(1.0, gemini_score))
            except ValueError:
