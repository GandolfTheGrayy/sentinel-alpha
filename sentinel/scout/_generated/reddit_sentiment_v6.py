"""
Reddit sentiment scraper for Sentinel Scout pillar.

Fetches posts and comments from r/wallstreetbets, r/stocks, and r/investing,
computing aggregate sentiment signals per ticker using lexicon-based scoring.
Outputs normalized SentimentSignal dataclass for downstream Linguist analysis.

Part of the Sentinel Sentiment Engine's multi-source signal aggregation.
"""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import praw
from praw.models import Submission


@dataclass
class SentimentSignal:
    """Normalized sentiment signal for a ticker from Reddit."""

    ticker: str
    source: str
    signal_type: str
    score: float
    confidence: float
    mention_count: int
    timestamp: datetime
    raw_text_sample: str


def _get_reddit_client() -> praw.Reddit:
    """Initialize PRAW Reddit API client from environment variables."""
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
        user_agent=os.getenv("REDDIT_USER_AGENT", "sentinel-scout/1.0"),
    )


def _extract_tickers(text: str) -> List[str]:
    """Extract stock tickers (e.g., $AAPL, AAPL) from text."""
    pattern = r"\$?([A-Z]{1,5})\b"
    matches = re.findall(pattern, text)
    filtered = [m for m in matches if len(m) >= 1 and len(m) <= 5]
    return list(set(filtered))


def _score_sentiment(text: str) -> float:
    """
    Score text sentiment on [-1, 1] scale using simple lexicon matching.
    
    Positive indicators: moon, diamond, hold, strong, buy, bullish, rocket, gains.
    Negative indicators: dump, bag, weak, sell, bearish, crash, loss, rip.
    """
    positive_words = {
        "moon",
        "diamond",
        "hold",
        "strong",
        "buy",
        "bullish",
        "rocket",
        "gains",
        "lambo",
        "tendies",
        "pump",
        "bull",
    }
    negative_words = {
        "dump",
        "bag",
        "weak",
        "sell",
        "bearish",
        "crash",
        "loss",
        "rip",
        "bear",
        "short",
        "panic",
        "rug",
    }

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    score = (pos_count - neg_count) / total
    return max(-1.0, min(1.0, score))


def scrape_subreddit_sentiment(
    subreddit_name: str,
    hours_back: int = 24,
    limit: int = 100,
) -> List[SentimentSignal]:
    """
    Scrape sentiment signals from a single subreddit.
    
    Returns list of SentimentSignal objects, one per ticker mentioned.
    """
    reddit = _get_reddit_client()
    subreddit = reddit.subreddit(subreddit_name)

    cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
    ticker_scores: Dict[str, Dict] = {}

    try:
        for submission in subreddit.new(limit=limit):
            submission_time = datetime.fromtimestamp(submission.created_utc)
            if submission_time < cutoff_time:
                continue

            combined_text = submission.title + " " + submission.selftext
            tickers = _extract_tickers(combined_text)
            sentiment_score = _score_sentiment(combined_text)

            for ticker in tickers:
                if ticker not in ticker_scores:
                    ticker_scores[ticker] = {
                        "scores": [],
                        "mentions": 0,
                        "sample": combined_text[:200],
                    }
                ticker_scores[ticker]["scores"].append(sentiment_score)
                ticker_scores[ticker]["mentions"] += 1
                ticker_scores[ticker]["sample"] = combined_text[:200]

    except Exception as e:
        print(f"Error scraping r/{subreddit_name}: {e}")
        return []

    signals = []
    for ticker, data in ticker_scores.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        confidence = min(0.95, len(data["scores"]) / 10.0)

        signal = SentimentSignal(
            ticker=ticker,
            source=f"reddit:{subreddit_name}",
            signal_type="aggregated_sentiment",
            score=avg_score,
            confidence=confidence,
            mention_count=data["mentions"],
            timestamp=datetime.utcnow(),
            raw_text_sample=data["sample"],
        )
        signals.append(signal)

    return signals


def scrape_all_wsb_sentiment(
    hours_back: int = 24,
    limit_per_subreddit: int = 100,
) -> List[SentimentSignal]:
    """
    Aggregate sentiment signals across r/wallstreetbets, r/stocks, r/investing.
    
    Returns combined list of SentimentSignal objects.
    """
    subreddits = ["wallstreetbets", "stocks", "investing"]
    all_signals = []

    for subreddit_name in subreddits:
        signals = scrape_subreddit_sentiment(
            subreddit_name=subreddit_name,
            hours_back=hours_back,
            limit=limit_per_subreddit,
        )
        all_signals.extend(signals)

    return all_signals


def aggregate_signals_by_ticker(
    signals: List[SentimentSignal],
) -> Dict[str, SentimentSignal]:
    """
    Consolidate multiple SentimentSignal objects per ticker into one.
    
    Averages scores and confidence, sums mentions, uses most recent sample.
    """
    ticker_map: Dict[str, Dict] = {}

    for signal in signals:
        if signal.ticker not in ticker_map:
            ticker_map[signal.ticker] = {
                "scores": [],
                "confidences": [],
                "mentions": 0,
                "samples": [],
                "sources": [],
            }

        ticker_map[signal.ticker]["scores"].append(signal.score)
        ticker_map[signal.ticker]["confidences"].append(signal.confidence)
        ticker_map[signal.ticker]["mentions"] += signal.mention_count
        ticker_map[signal.ticker]["samples"].append(signal.raw_text_sample)
        ticker_map[signal.ticker]["sources"].append(signal.source)

    aggregated = {}
    for ticker, data in ticker_map.items():
        avg_score = sum(data["scores"]) / len(data["scores"])
        avg_confidence = sum(data["confidences"]) / len(data["confidences"])

        aggregated[ticker] = SentimentSignal(
            ticker=ticker,
            source=",".join(set(data["sources"])),
            signal_type="aggregated_reddit_sentiment",
            score=avg_score,
            confidence=avg_confidence,
            mention_count=data["mentions"],
            timestamp=datetime.utcnow(),
            raw_text_sample=data["samples"][0] if data["samples"] else "",
        )

    return aggregated


if __name__ == "__main__":
    signals = scrape_all_wsb_sentiment(hours_back=24, limit_per_subreddit=50)
    aggregated = aggregate_signals_by_ticker(signals)

    print(f"Collected {len
