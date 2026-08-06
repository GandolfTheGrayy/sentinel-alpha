"""
Hacker News scraper for Sentinel Scout pillar.

Ingests 'Ask HN' posts mentioning tech companies, extracts developer community
sentiment signals (hiring, layoffs, tech stack sentiment, company culture).
Uses Gemini (flash-lite) for HTML parsing and comment extraction; outputs
structured sentiment records for Linguist downstream analysis.

Integrates with the Sentinel pipeline as a niche sentiment source,
complementing traditional news and SEC filings.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional
import re

import requests
import google.generativeai as genai
import sqlite3
from pathlib import Path


# Initialize Gemini client
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))

# HN API base URL
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_WEB_BASE = "https://news.ycombinator.com"

# Database setup
DB_PATH = Path(__file__).parent.parent.parent / "data" / "hn_sentiment.db"


def _init_db() -> None:
    """Initialize SQLite schema for HN sentiment records."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hn_posts (
            id INTEGER PRIMARY KEY,
            hn_id INTEGER UNIQUE,
            title TEXT,
            author TEXT,
            timestamp INTEGER,
            score INTEGER,
            comment_count INTEGER,
            url TEXT,
            fetched_at REAL
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hn_sentiments (
            id INTEGER PRIMARY KEY,
            post_id INTEGER,
            ticker TEXT,
            sentiment_label TEXT,
            confidence REAL,
            keywords TEXT,
            raw_text TEXT,
            analyzed_at REAL,
            FOREIGN KEY (post_id) REFERENCES hn_posts(hn_id)
        )
    """
    )
    conn.commit()
    conn.close()


def fetch_ask_hn_ids(limit: int = 30) -> list[int]:
    """
    Fetch recent 'Ask HN' post IDs from HN API.

    Args:
        limit: Maximum number of post IDs to retrieve.

    Returns:
        List of HN post IDs.
    """
    try:
        resp = requests.get(f"{HN_API_BASE}/askstories.json", timeout=10)
        resp.raise_for_status()
        all_ids = resp.json() or []
        return all_ids[:limit]
    except Exception as e:
        print(f"Error fetching Ask HN IDs: {e}")
        return []


def fetch_post_details(post_id: int) -> Optional[dict]:
    """
    Fetch full post details (title, author, score, comment count) from HN API.

    Args:
        post_id: HN post ID.

    Returns:
        Dict with post metadata, or None on error.
    """
    try:
        resp = requests.get(f"{HN_API_BASE}/item/{post_id}.json", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data and data.get("type") == "story":
            return {
                "id": post_id,
                "title": data.get("title", ""),
                "author": data.get("by", "unknown"),
                "timestamp": data.get("time", 0),
                "score": data.get("score", 0),
                "comment_count": len(data.get("kids", [])),
                "url": f"{HN_WEB_BASE}/item?id={post_id}",
            }
    except Exception as e:
        print(f"Error fetching post {post_id}: {e}")
    return None


def extract_tech_companies(text: str) -> list[str]:
    """
    Extract potential tech company mentions from text using regex + keyword list.

    Args:
        text: Raw text (title or comments).

    Returns:
        List of company names/tickers found.
    """
    # Common tech companies and their aliases
    company_patterns = {
        "AAPL": [r"\bapple\b", r"\biphone\b", r"\bipad\b"],
        "MSFT": [r"\bmicrosoft\b", r"\bwindows\b", r"\bazure\b"],
        "GOOGL": [r"\bgoogle\b", r"\bgmail\b"],
        "AMZN": [r"\bamazon\b", r"\baws\b"],
        "META": [r"\bmeta\b", r"\bfacebook\b", r"\binstagram\b"],
        "TSLA": [r"\btesla\b", r"\belon\b"],
        "NVDA": [r"\bnvidia\b", r"\bcuda\b"],
        "AMD": [r"\bamd\b", r"\bzen\b"],
        "INTC": [r"\bintel\b", r"\bx86\b"],
        "CRM": [r"\bsalesforce\b"],
        "ADBE": [r"\badobe\b"],
        "NFLX": [r"\bnetflix\b"],
        "PYPL": [r"\bpaypal\b"],
        "UBER": [r"\buber\b"],
        "LYFT": [r"\blyft\b"],
        "SNAP": [r"\bsnap\b", r"\bsnapchat\b"],
        "SPOT": [r"\bspotify\b"],
        "COIN": [r"\bcoinbase\b"],
        "RBLX": [r"\broblox\b"],
        "ZM": [r"\bzoom\b"],
    }

    found = set()
    text_lower = text.lower()
    for ticker, patterns in company_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found.add(ticker)
    return list(found)


def analyze_post_sentiment(post: dict) -> list[dict]:
    """
    Use Gemini to parse post title/comments, extract sentiment signals per company.

    Args:
        post: Dict with 'title', 'id', 'url' keys.

    Returns:
        List of sentiment records {ticker, label, confidence, keywords}.
    """
    title = post.get("title", "")
    companies = extract_tech_companies(title)

    if not companies:
        return []

    prompt = f"""
Analyze this Hacker News 'Ask HN' post title for developer/tech community sentiment.

Title: {title}
URL: {post.get('url', 'N/A')}

For each company mentioned, provide:
1. Sentiment (positive/negative/neutral)
2. Confidence (0.0–1.0)
3. Key signals (hiring, layoffs, tech stack, culture, regulatory, etc.)

Output as JSON array:
[
  {{"ticker": "AAPL", "sentiment": "positive", "confidence": 0.85, "signals": ["hiring announcement", "ecosystem strength"]}},
  ...
]

If no clear sentiment, return empty array [].
"""

    try:
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
        text = response.text.strip()

        # Extract JSON from response
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            sentiments = json.loads(json_match.group())
            # Validate and filter
            result = []
            for item in sentiments:
                if (
                    isinstance(item, dict)
                    and "ticker" in item
                    and item["ticker"] in companies
                ):
                    result.append(
                        {
                            "ticker": item["ticker"],
                            "sentiment_label": item.get("sentiment",
