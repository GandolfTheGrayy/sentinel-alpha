"""
Sentinel Scout: Reddit Sentiment Scraper

Ingests live sentiment signals from r/wallstreetbets, r/stocks, and r/investing
using PRAW (Python Reddit API Wrapper). Analyzes post/comment volume, upvote ratios,
and comment sentiment to produce normalized SentimentSignal dataclasses keyed by ticker.

Feeds into Linguist for tone analysis and Judge for prediction weighting.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import praw
from praw.exceptions import PrawException


@dataclass
class SentimentSignal:
    """Normalized sentiment signal for a single ticker."""

    ticker: str
    source: str  # "reddit"
    timestamp: datetime
    raw_score: float  # [-1.0, 1.0]: -1 bearish, 0 neutral, +1 bullish
    volume_signal: int  # post/comment count (absolute)
    confidence: float  # [0.0, 1.0]: how certain is this signal
    metadata: Dict[str, any]  # upvote_ratio, comment_count, post_count, subreddits


def _extract_tickers_from_text(text: str) -> List[str]:
    """Extract stock tickers (e.g., $AAPL, TSLA) from text using regex."""
    # Match $TICKER or standalone TICKER (4 letters or less, uppercase)
    pattern = r"\$([A-Z]{1,5})\b|(?:^|\s)([A-Z]{1,5})(?:\s|$)"
    matches = re.findall(pattern, text)
    tickers = []
    for match in matches:
        ticker = match[0] or match[1]
        if ticker and len(ticker) <= 5:
            tickers.append(ticker.strip())
    return list(set(tickers))


def _score_sentiment(
    text: str, upvote_ratio: float, num_comments: int
) -> tuple[float, float]:
    """
    Score sentiment from text + engagement metrics.

    Returns: (raw_score, confidence)
    - raw_score: [-1.0, 1.0], -1=bearish, 0=neutral, +1=bullish
    - confidence: [0.0, 1.0], higher with more comments/engagement
    """
    bullish_keywords = [
        "moon",
        "rocket",
        "diamond hands",
        "buy",
        "hold",
        "bullish",
        "pump",
        "long",
        "undervalued",
        "strong",
        "growth",
        "profit",
    ]
    bearish_keywords = [
        "crash",
        "dump",
        "sell",
        "bearish",
        "short",
        "overvalued",
        "weak",
        "loss",
        "panic",
        "red",
        "downside",
        "risk",
    ]

    text_lower = text.lower()
    bullish_count = sum(text_lower.count(kw) for kw in bullish_keywords)
    bearish_count = sum(text_lower.count(kw) for kw in bearish_keywords)

    # Net sentiment score
    if bullish_count + bearish_count == 0:
        raw_score = 0.0
    else:
        raw_score = (bullish_count - bearish_count) / (bullish_count + bearish_count)
    raw_score = max(-1.0, min(1.0, raw_score))

    # Confidence from engagement: more comments = higher confidence
    engagement_confidence = min(1.0, num_comments / 100.0)
    # Upvote ratio also signals consensus
    ratio_confidence = abs(upvote_ratio - 0.5) * 2  # 0.5 ratio = 0 conf, 1.0 = 1.0
    confidence = (engagement_confidence + ratio_confidence) / 2.0

    return raw_score, confidence


def scrape_reddit_sentiment(
    subreddit_names: Optional[List[str]] = None,
    lookback_hours: int = 24,
    limit_posts_per_subreddit: int = 100,
) -> List[SentimentSignal]:
    """
    Scrape Reddit sentiment from specified subreddits over the past N hours.

    Args:
        subreddit_names: List of subreddit names (default: r/wallstreetbets, r/stocks, r/investing)
        lookback_hours: How far back to scan (default: 24 hours)
        limit_posts_per_subreddit: Max posts to analyze per subreddit (default: 100)

    Returns:
        List of SentimentSignal objects, one per unique ticker found.

    Raises:
        PrawException: If Reddit API authentication or rate limiting fails.
    """
    if subreddit_names is None:
        subreddit_names = ["wallstreetbets", "stocks", "investing"]

    # Initialize PRAW client from environment
    try:
        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0"),
        )
    except PrawException as e:
        raise PrawException(f"Failed to authenticate with Reddit API: {e}")

    # Aggregate sentiment by ticker
    ticker_signals: Dict[str, Dict] = {}
    now = datetime.utcnow()
    cutoff = now - timedelta(hours=lookback_hours)

    for subreddit_name in subreddit_names:
        try:
            subreddit = reddit.subreddit(subreddit_name)

            # Fetch hot/new posts
            posts = list(subreddit.hot(limit=limit_posts_per_subreddit))

            for post in posts:
                # Filter by timestamp
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if post_time < cutoff:
                    continue

                # Extract tickers from title + selftext
                full_text = f"{post.title} {post.selftext}"
                tickers = _extract_tickers_from_text(full_text)

                if not tickers:
                    continue

                # Score sentiment
                upvote_ratio = post.upvote_ratio if post.upvote_ratio else 0.5
                num_comments = post.num_comments if post.num_comments else 0
                raw_score, confidence = _score_sentiment(
                    full_text, upvote_ratio, num_comments
                )

                # Aggregate per ticker
                for ticker in tickers:
                    if ticker not in ticker_signals:
                        ticker_signals[ticker] = {
                            "scores": [],
                            "volumes": [],
                            "upvote_ratios": [],
                            "comment_counts": [],
                            "post_count": 0,
                            "subreddits": set(),
                        }

                    ticker_signals[ticker]["scores"].append(raw_score)
                    ticker_signals[ticker]["volumes"].append(1)
                    ticker_signals[ticker]["upvote_ratios"].append(upvote_ratio)
                    ticker_signals[ticker]["comment_counts"].append(num_comments)
                    ticker_signals[ticker]["post_count"] += 1
                    ticker_signals[ticker]["subreddits"].add(subreddit_name)

        except PrawException as e:
            print(f"Warning: Failed to scrape r/{subreddit_name}: {e}")
            continue

    # Normalize and convert to SentimentSignal objects
    signals = []
    for ticker, data in ticker_signals.items():
        if not data["scores"]:
            continue

        avg_raw_score = sum(data["scores"]) / len(data["scores"])
        avg_confidence = sum(data["upvote_ratios"]) / len(data["upvote_ratios"])
        total_volume
