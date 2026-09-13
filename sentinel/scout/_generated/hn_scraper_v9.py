"""
Hacker News scraper for Sentinel Scout.

Targets 'Ask HN' posts mentioning tech companies to extract developer sentiment.
Uses Gemini for parsing and relevance filtering. Scores posts by upvotes, comment
count, and linguistic signals (complaint vs. praise keywords). Results feed into
the Linguist for certainty analysis and Historian for RAG context.

Role in Sentinel:
  - Ingests HN discussions as a niche sentiment signal complementary to news/Reddit.
  - Developer community perspective often precedes mainstream market moves.
  - Outputs scored sentiment tuples: (ticker, post_id, title, score, confidence).
"""

import os
import re
import time
import sqlite3
from typing import Optional
from datetime import datetime
import requests
import google.generativeai as genai

# Initialize Gemini client for parsing HN HTML.
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

# Approved tech company tickers we listen for.
TECH_TICKERS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "PYPL": "PayPal",
    "SQ": "Square",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "INTC": "Intel",
    "AMD": "AMD",
    "UBER": "Uber",
    "LYFT": "Lyft",
}

# Keywords indicating negative/positive sentiment in HN discussions.
NEGATIVE_KEYWORDS = [
    "bug", "broken", "crash", "fail", "error", "slow", "outage", "exploit",
    "security", "vulnerability", "lawsuit", "scandal", "layoff", "decline",
]
POSITIVE_KEYWORDS = [
    "great", "excellent", "innovation", "growth", "profit", "success",
    "partnership", "acquisition", "upgrade", "impressive", "breakthrough",
]


def fetch_hn_ask_posts(limit: int = 50) -> list[dict]:
    """Fetch recent 'Ask HN' posts from HN API; return raw post metadata."""
    try:
        # Get top Ask HN post IDs.
        url = "https://hacker-news.firebaseio.com/v0/askstories.json"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        post_ids = resp.json()[:limit]

        posts = []
        for post_id in post_ids:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{post_id}.json"
            item_resp = requests.get(item_url, timeout=5)
            item_resp.raise_for_status()
            posts.append(item_resp.json())
            time.sleep(0.1)  # Be respectful to HN API.

        return posts
    except Exception as e:
        print(f"Error fetching HN posts: {e}")
        return []


def parse_company_mentions(title: str, text: Optional[str] = None) -> list[str]:
    """Extract ticker symbols and company names from post title/text."""
    mentions = []
    full_text = f"{title} {text or ''}".lower()

    for ticker, name in TECH_TICKERS.items():
        if ticker.lower() in full_text or name.lower() in full_text:
            mentions.append(ticker)

    return list(set(mentions))


def score_sentiment(title: str, text: Optional[str] = None) -> float:
    """
    Score sentiment of HN post as float in [-1.0, 1.0].
    Negative: -1.0, Neutral: 0.0, Positive: 1.0.
    """
    full_text = f"{title} {text or ''}".lower()

    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in full_text)
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in full_text)

    if neg_count + pos_count == 0:
        return 0.0

    return (pos_count - neg_count) / (pos_count + neg_count)


def score_engagement(post: dict) -> float:
    """Score post credibility via upvotes + comment volume; range [0.0, 1.0]."""
    score = post.get("score", 0)
    descendants = post.get("descendants", 0)

    # Normalize: 100 points = 0.5, 500 descendants = 0.5.
    score_component = min(score / 100.0, 1.0) * 0.5
    descendants_component = min(descendants / 500.0, 1.0) * 0.5

    return score_component + descendants_component


def classify_relevance_with_gemini(title: str, text: Optional[str] = None) -> bool:
    """
    Use Gemini to classify if post is genuinely about tech company
    fundamentals (not spam/noise). Returns True if relevant.
    """
    try:
        prompt = f"""You are a classifier for Hacker News discussions about tech companies.
Determine if the following 'Ask HN' post is genuinely asking about or discussing
a tech company's product, performance, or ecosystem. Return only "yes" or "no".

Title: {title}
Text: {text or "(no text)"}

Response (yes/no):"""

        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
        response = model.generate_content(prompt, generation_config={"max_output_tokens": 10})
        return "yes" in response.text.lower()
    except Exception as e:
        print(f"Gemini relevance check failed: {e}")
        return True  # Default to including if classifier fails.


def scrape_and_score_hn(db_path: str = "sentinel/data/hn_sentiment.db", limit: int = 50) -> list[dict]:
    """
    Main entry point: fetch HN Ask posts, filter for tech companies,
    score sentiment/engagement, and store in SQLite. Return list of scored posts.
    """
    # Initialize DB.
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hn_posts (
            post_id INTEGER PRIMARY KEY,
            title TEXT,
            ticker TEXT,
            sentiment_score REAL,
            engagement_score REAL,
            combined_score REAL,
            posted_at INTEGER,
            fetched_at INTEGER,
            url TEXT
        )
    """)
    conn.commit()

    posts = fetch_hn_ask_posts(limit=limit)
    scored_posts = []

    for post in posts:
        post_id = post.get("id")
        title = post.get("title", "")
        text = post.get("text", "")
        post_time = post.get("time")
        url = f"https://news.ycombinator.com/item?id={post_id}"

        # Filter for relevance.
        if not classify_relevance_with_gemini(title, text):
            continue

        # Extract company mentions.
        tickers = parse_company_mentions(title, text)
        if not tickers:
            continue

        # Score sentiment & engagement.
        sentiment = score_sentiment(title, text)
        engagement = score_engagement(post)
        combined = (sentiment + 1) / 2 * engagement  # Normalize sentiment to [0, 1], weight by engagement.

        for ticker in tickers:
            record = {
                "post_id": post_id,
                "title": title,
                "ticker": ticker,
                "sentiment_score": sentiment,
                "engagement_score": engagement,
                "combined_score": combined,
                "posted_at": post_
