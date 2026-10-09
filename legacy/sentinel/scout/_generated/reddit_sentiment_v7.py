"""
Reddit sentiment scraper for Sentinel Scout pillar.

Fetches posts and comments from r/wallstreetbets, r/stocks, and r/investing
using PRAW. Analyzes sentiment via keyword heuristics and LLM classification.
Outputs normalized SentimentSignal dataclass for downstream Linguist analysis.

Role in Sentinel: Captures retail investor sentiment as a niche signal
cross-referenced with institutional signals (SEC filings, news) in Judge.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional
import sqlite3
from datetime import datetime, timedelta

import praw
import numpy as np
from anthropic import Anthropic


@dataclass
class SentimentSignal:
    """Normalized sentiment output from Reddit scraper."""
    ticker: str
    subreddit: str
    signal_type: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    volume: int  # number of posts/comments analyzed
    raw_score: float  # aggregate sentiment score
    timestamp: str
    sample_text: str  # representative post excerpt


class RedditSentimentScraper:
    """Scrapes Reddit sentiment for stock tickers across three subreddits."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
    BULLISH_KEYWORDS = {
        "moon", "rocket", "🚀", "to the moon", "bull", "bullish", "long",
        "hold", "diamond hands", "buy", "buying", "pump", "gains", "win",
        "profit", "surge", "spike", "breakout", "undervalued", "bargain"
    }
    BEARISH_KEYWORDS = {
        "crash", "tank", "bearish", "short", "sell", "selling", "dump",
        "loss", "plunge", "drop", "red", "fear", "panic", "overvalued",
        "bubble", "collapse", "dead", "rug pull", "scam", "bag holder"
    }

    def __init__(self):
        """Initialize PRAW Reddit client from environment variables."""
        try:
            self.reddit = praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID", ""),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
                user_agent=os.getenv("REDDIT_USER_AGENT", "Sentinel/1.0"),
            )
            self._anthropic = Anthropic()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Reddit client: {e}")

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract stock ticker symbols (e.g., $AAPL) from text."""
        matches = re.findall(r"\$([A-Z]{1,5})\b", text)
        return list(set(matches))

    def _heuristic_sentiment(self, text: str) -> tuple[float, str]:
        """
        Compute sentiment score via keyword heuristics.
        Returns (score, signal_type) where score is -1.0 to 1.0.
        """
        text_lower = text.lower()
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        net_score = (bullish_count - bearish_count) / (bullish_count + bearish_count + 1)
        
        if net_score > 0.2:
            signal_type = "bullish"
        elif net_score < -0.2:
            signal_type = "bearish"
        else:
            signal_type = "neutral"
        
        return net_score, signal_type

    def _llm_refine_sentiment(self, text: str, ticker: str) -> tuple[float, str, float]:
        """
        Use Claude to refine sentiment score and confidence.
        Returns (refined_score, signal_type, confidence).
        """
        prompt = f"""
Analyze sentiment for ${ticker} in this Reddit post/comment. 
Be concise. Output JSON: {{"score": -1.0 to 1.0, "signal": "bullish"|"bearish"|"neutral", "confidence": 0.0 to 1.0}}

Text:
{text[:500]}
"""
        try:
            msg = self._anthropic.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = msg.content[0].text
            
            import json
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return (
                    float(data.get("score", 0.0)),
                    data.get("signal", "neutral"),
                    float(data.get("confidence", 0.5))
                )
        except Exception:
            pass
        
        return 0.0, "neutral", 0.5

    def scrape_subreddit(
        self, subreddit_name: str, limit: int = 50
    ) -> dict[str, list[SentimentSignal]]:
        """
        Scrape posts from a subreddit and aggregate sentiment by ticker.
        Returns dict mapping ticker -> list of SentimentSignal.
        """
        signals_by_ticker: dict[str, list[SentimentSignal]] = {}
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts = list(subreddit.new(limit=limit))
        except Exception as e:
            print(f"Error fetching {subreddit_name}: {e}")
            return signals_by_ticker
        
        for post in posts:
            text = f"{post.title}\n{post.selftext}"
            tickers = self._extract_tickers(text)
            
            if not tickers:
                continue
            
            heur_score, heur_type = self._heuristic_sentiment(text)
            llm_score, llm_type, confidence = self._llm_refine_sentiment(text, tickers[0])
            
            final_score = 0.6 * llm_score + 0.4 * heur_score
            final_type = llm_type if confidence > 0.6 else heur_type
            
            timestamp = datetime.fromtimestamp(post.created_utc).isoformat()
            
            for ticker in tickers:
                signal = SentimentSignal(
                    ticker=ticker,
                    subreddit=subreddit_name,
                    signal_type=final_type,
                    confidence=confidence,
                    volume=1,
                    raw_score=final_score,
                    timestamp=timestamp,
                    sample_text=text[:200]
                )
                
                if ticker not in signals_by_ticker:
                    signals_by_ticker[ticker] = []
                signals_by_ticker[ticker].append(signal)
        
        return signals_by_ticker

    def scrape_all(self, limit_per_subreddit: int = 50) -> dict[str, list[SentimentSignal]]:
        """Scrape all three subreddits and aggregate sentiment signals."""
        all_signals: dict[str, list[SentimentSignal]] = {}
        
        for subreddit_name in self.SUBREDDITS:
            signals = self.scrape_subreddit(subreddit_name, limit=limit_per_subreddit)
            for ticker, ticker_signals in signals.items():
                if ticker not in all_signals:
                    all_signals[ticker] = []
                all_signals[ticker].extend(ticker_signals)
        
        return all_signals

    def aggregate_signals(
        self, signals: list[SentimentSignal]
    ) -> Optional[SentimentSignal]:
        """Aggregate multiple signals into a single
