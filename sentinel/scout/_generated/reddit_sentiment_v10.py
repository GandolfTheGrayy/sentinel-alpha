"""
Reddit sentiment scraper for Sentinel Scout pillar.

Uses PRAW to fetch posts and comments from r/wallstreetbets, r/stocks, and
r/investing, analyzing sentiment signals for equities. Outputs normalized
SentimentSignal dataclass with score, volume, and confidence metrics tied
to ticker symbols. Designed to feed into Historian RAG and Judge prediction.
"""

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import re

import praw
from praw.exceptions import PrawException


@dataclass
class SentimentSignal:
    """Normalized sentiment signal from Reddit analysis."""
    
    ticker: str
    source: str
    score: float
    confidence: float
    volume: int
    timestamp: datetime
    raw_text: str
    summary: str


def _initialize_reddit_client() -> praw.Reddit:
    """Initialize PRAW Reddit client from environment credentials."""
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent = os.getenv("REDDIT_USER_AGENT", "sentinel-scout/1.0")
    
    if not client_id or not client_secret:
        raise ValueError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in env"
        )
    
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )


def _extract_tickers(text: str) -> list[str]:
    """Extract potential stock ticker symbols from text."""
    # Match $SYMBOL or bare 1–5 uppercase letter sequences surrounded by word boundaries
    pattern = r"\$[A-Z]{1,5}\b|(?:^|\s)([A-Z]{1,5})(?:\s|$|\.|\,)"
    matches = re.findall(pattern, text, re.MULTILINE)
    tickers = []
    for match in matches:
        if isinstance(match, tuple):
            ticker = match[0] if match[0] else None
        else:
            ticker = match.replace("$", "")
        if ticker and len(ticker) <= 5:
            tickers.append(ticker)
    return list(set(tickers))  # deduplicate


def _score_sentiment(text: str) -> tuple[float, float]:
    """
    Score sentiment of text on [-1, 1] scale with confidence [0, 1].
    
    Returns (score, confidence).
    """
    text_lower = text.lower()
    
    bullish_words = [
        "moon", "rocket", "bull", "buy", "long", "gains", "lambo",
        "diamond hands", "hodl", "bullish", "squeeze", "tendies",
        "to the moon", "rip", "call"
    ]
    bearish_words = [
        "crash", "dump", "bear", "sell", "short", "loss", "bag holder",
        "paper hands", "bearish", "tank", "rip", "puts", "downside"
    ]
    
    bullish_count = sum(1 for word in bullish_words if word in text_lower)
    bearish_count = sum(1 for word in bearish_words if word in text_lower)
    
    total_signals = bullish_count + bearish_count
    
    if total_signals == 0:
        return 0.0, 0.0
    
    score = (bullish_count - bearish_count) / total_signals
    confidence = min(total_signals / 10.0, 1.0)
    
    return score, confidence


def fetch_reddit_sentiment(
    subreddits: Optional[list[str]] = None,
    hours_back: int = 24,
    limit_per_sub: int = 100
) -> list[SentimentSignal]:
    """
    Fetch and analyze sentiment from Reddit subreddits.
    
    Args:
        subreddits: List of subreddit names (default: wallstreetbets, stocks, investing)
        hours_back: Hours of historical posts to scan (default: 24)
        limit_per_sub: Max posts per subreddit (default: 100)
    
    Returns:
        List of SentimentSignal objects with ticker, score, confidence, volume.
    """
    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing"]
    
    reddit = _initialize_reddit_client()
    signals_by_ticker = {}
    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    
    for subreddit_name in subreddits:
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Fetch new posts
            for post in subreddit.new(limit=limit_per_sub):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if post_time < cutoff_time:
                    continue
                
                # Combine title and selftext for analysis
                combined_text = f"{post.title}\n{post.selftext}"
                
                tickers = _extract_tickers(combined_text)
                score, confidence = _score_sentiment(combined_text)
                
                if not tickers:
                    continue
                
                for ticker in tickers:
                    if ticker not in signals_by_ticker:
                        signals_by_ticker[ticker] = {
                            "total_score": 0.0,
                            "total_confidence": 0.0,
                            "volume": 0,
                            "texts": [],
                            "subreddit": subreddit_name,
                            "timestamp": post_time
                        }
                    
                    signals_by_ticker[ticker]["total_score"] += score
                    signals_by_ticker[ticker]["total_confidence"] += confidence
                    signals_by_ticker[ticker]["volume"] += 1
                    signals_by_ticker[ticker]["texts"].append(combined_text[:200])
                    signals_by_ticker[ticker]["timestamp"] = max(
                        signals_by_ticker[ticker]["timestamp"], post_time
                    )
        
        except PrawException as e:
            print(f"PRAW error fetching r/{subreddit_name}: {e}")
            continue
    
    # Normalize and convert to SentimentSignal objects
    result = []
    now = datetime.utcnow()
    
    for ticker, data in signals_by_ticker.items():
        avg_score = data["total_score"] / data["volume"] if data["volume"] > 0 else 0.0
        avg_confidence = data["total_confidence"] / data["volume"] if data["volume"] > 0 else 0.0
        
        signal = SentimentSignal(
            ticker=ticker,
            source=f"reddit/{data['subreddit']}",
            score=avg_score,
            confidence=avg_confidence,
            volume=data["volume"],
            timestamp=now,
            raw_text="\n---\n".join(data["texts"][:3]),
            summary=f"{data['volume']} mentions across {data['subreddit']}, "
                    f"avg sentiment {avg_score:.2f}, confidence {avg_confidence:.2f}"
        )
        result.append(signal)
    
    return result


if __name__ == "__main__":
    signals = fetch_reddit_sentiment()
    for sig in signals:
        print(f"{sig.ticker}: score={sig.score:.2f}, "
              f"confidence={sig.confidence:.2f}, volume={sig.volume}")
