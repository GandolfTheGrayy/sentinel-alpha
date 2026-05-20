"""
Hacker News scraper for Sentinel Scout pillar.

Targets 'Ask HN' posts mentioning tech companies to extract developer community
sentiment signals. Uses Gemini for HTML parsing and entity extraction.
Integrates with the main pipeline via scout module interfaces.
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional
import requests
from google.generativeai import GenerativeModel
import sqlite3


# HN API endpoint
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_ITEM_URL = f"{HN_API_BASE}/item"
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

# Tech company keywords and aliases
TECH_COMPANIES = {
    "apple": ["apple", "aapl", "macos", "ios", "iphone"],
    "microsoft": ["microsoft", "msft", "windows", "azure", "github"],
    "google": ["google", "alphabet", "googl", "goog", "android"],
    "meta": ["meta", "facebook", "threads", "instagram"],
    "amazon": ["amazon", "amzn", "aws"],
    "nvidia": ["nvidia", "nvda", "gpu", "cuda"],
    "tesla": ["tesla", "tsla", "elon"],
    "openai": ["openai", "chatgpt", "gpt"],
    "anthropic": ["anthropic", "claude"],
    "xai": ["xai", "grok"],
}


def _init_db(db_path: str = "/tmp/sentinel_hn.db") -> None:
    """Initialize SQLite DB for caching HN posts."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS hn_posts (
            id INTEGER PRIMARY KEY,
            hn_id INTEGER UNIQUE,
            title TEXT,
            text TEXT,
            company TEXT,
            sentiment_score REAL,
            created_at TIMESTAMP,
            fetched_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def _get_cached_post(hn_id: int, db_path: str = "/tmp/sentinel_hn.db") -> Optional[dict]:
    """Retrieve cached HN post from SQLite."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM hn_posts WHERE hn_id = ?", (hn_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "hn_id": row[1],
            "title": row[2],
            "text": row[3],
            "company": row[4],
            "sentiment_score": row[5],
            "created_at": row[6],
            "fetched_at": row[7],
        }
    return None


def _cache_post(
    hn_id: int,
    title: str,
    text: str,
    company: str,
    sentiment_score: float,
    created_at: str,
    db_path: str = "/tmp/sentinel_hn.db",
) -> None:
    """Cache HN post in SQLite."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute(
            """
            INSERT INTO hn_posts
            (hn_id, title, text, company, sentiment_score, created_at, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (hn_id, title, text, company, sentiment_score, created_at, datetime.utcnow()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    finally:
        conn.close()


def _extract_company_mentions(text: str) -> list[str]:
    """Extract mentioned tech companies from text using keyword matching."""
    text_lower = text.lower()
    mentioned = set()
    for company, keywords in TECH_COMPANIES.items():
        for keyword in keywords:
            if re.search(r"\b" + keyword + r"\b", text_lower):
                mentioned.add(company)
                break
    return list(mentioned)


def _score_sentiment_with_gemini(title: str, text: str) -> tuple[float, str]:
    """
    Use Gemini to score developer sentiment in HN post (0.0 to 1.0, 0.5 = neutral).
    Returns (score, reasoning).
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return 0.5, "No GEMINI_API_KEY; defaulting to neutral."

    model = GenerativeModel("gemini-3.1-flash-lite-preview")
    prompt = f"""
You are a sentiment analyst for developer community signals.
Analyze this Hacker News "Ask HN" post and score developer sentiment on a scale 0.0–1.0:
- 0.0–0.33: Negative (frustration, bugs, concerns)
- 0.34–0.66: Neutral (informational, balanced)
- 0.67–1.0: Positive (excitement, praise, adoption)

Title: {title}
Text: {text[:1000]}

Respond with ONLY a JSON object on one line:
{{"score": <float>, "reasoning": "<brief explanation>"}}
"""
    try:
        response = model.generate_content(prompt, request_options={"timeout": 10})
        text_response = response.text.strip()
        # Extract JSON from response
        import json
        json_match = re.search(r"\{.*\}", text_response)
        if json_match:
            data = json.loads(json_match.group())
            score = float(data.get("score", 0.5))
            reasoning = data.get("reasoning", "")
            return max(0.0, min(1.0, score)), reasoning
    except Exception as e:
        pass
    return 0.5, f"Gemini error: {str(e)}"


def fetch_hn_ask_posts(
    hours: int = 24, min_score: int = 10, max_posts: int = 50
) -> list[dict]:
    """
    Fetch recent 'Ask HN' posts mentioning tech companies from Hacker News.
    
    Args:
        hours: Look back this many hours for recent posts.
        min_score: Only include posts with at least this HN score.
        max_posts: Return at most this many posts.
    
    Returns:
        List of dicts: {hn_id, title, text, companies, sentiment_score, url}.
    """
    _init_db()

    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    cutoff_unix = int(cutoff_time.timestamp())

    results = []

    try:
        # Use Algolia HN search for 'Ask HN'
        search_url = f"{HN_SEARCH_URL}?query=Ask+HN&type=story&numericFilters=created_at_i>{cutoff_unix}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", [])

        for hit in hits[:max_posts * 2]:  # Fetch 2x to account for filtering
            hn_id = hit.get("objectID")
            title = hit.get("title", "")
            score = hit.get("points", 0)

            if score < min_score:
                continue

            # Check cache first
            cached = _get_cached_post(int(hn_id))
            if cached:
                results.append(cached)
                if len(results) >= max_posts:
                    return results
                continue

            # Fetch full item from HN API
            item_url = f"{HN_ITEM_URL}/{hn_id}.json"
