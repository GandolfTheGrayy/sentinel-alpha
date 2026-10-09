"""
Hacker News sentiment scraper for Sentinel Scout.

Targets 'Ask HN' posts mentioning tech companies, extracts developer community
sentiment signals, and scores tone/enthusiasm. Feeds into the broader Scout
data ingestion pipeline for cross-referencing with historical market moves.

Uses Gemini (flash-lite) for HTML parsing and text extraction; sentiment
scoring is deferred to Linguist modules for consistency.
"""

import os
import re
import time
from typing import Optional
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# Initialize Gemini client from env
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))


def fetch_hn_frontpage() -> Optional[list[dict]]:
    """Fetch top stories from Hacker News frontpage via official API."""
    try:
        resp = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
        resp.raise_for_status()
        story_ids = resp.json()[:30]  # Top 30
        return story_ids
    except Exception as e:
        print(f"HN frontpage fetch failed: {e}")
        return None


def fetch_hn_story(story_id: int) -> Optional[dict]:
    """Fetch individual HN story metadata by ID."""
    try:
        resp = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
            timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"HN story {story_id} fetch failed: {e}")
        return None


def fetch_hn_comments(story_id: int, max_depth: int = 2) -> list[dict]:
    """
    Recursively fetch HN comments for a story up to max_depth.
    
    Returns list of comment dicts with 'text', 'score', 'by', 'kids' fields.
    """
    comments = []

    def recurse(item_id: int, depth: int) -> None:
        if depth > max_depth or not item_id:
            return
        item = fetch_hn_story(item_id)
        if not item or item.get("type") != "comment":
            return
        comments.append({
            "text": item.get("text", ""),
            "score": item.get("score", 0),
            "by": item.get("by", ""),
            "time": item.get("time", 0),
        })
        for kid_id in item.get("kids", [])[:3]:  # Limit children per comment
            recurse(kid_id, depth + 1)
            time.sleep(0.1)  # Rate limit

    story = fetch_hn_story(story_id)
    if story:
        for kid_id in story.get("kids", [])[:10]:  # Top 10 comments
            recurse(kid_id, 0)
            time.sleep(0.1)

    return comments


def is_ask_hn_about_company(title: str, text: str = "") -> bool:
    """Check if HN post is an 'Ask HN' about a tech company."""
    ask_pattern = r"(?i)(ask\s+hn|who's\s+hiring|show\s+hn)"
    company_keywords = [
        "google", "meta", "apple", "microsoft", "amazon", "nvidia", "tesla",
        "openai", "anthropic", "stripe", "databricks", "figma", "notion",
        "github", "gitlab", "slack", "discord", "coinbase", "robinhood",
        "reddit", "twitter", "x corp", "palantir", "databricks", "hugging",
        "inflection", "stability", "midjourney", "runway", "cursor"
    ]
    combined = f"{title} {text}".lower()
    is_ask = bool(re.search(ask_pattern, title))
    has_company = any(kw in combined for kw in company_keywords)
    return is_ask and has_company


def extract_sentiment_from_comments(comments: list[dict], story_title: str) -> dict:
    """
    Use Gemini to extract developer sentiment themes from HN comments.
    
    Returns dict with 'themes', 'tone', 'keywords', 'community_score'.
    """
    if not comments:
        return {
            "themes": [],
            "tone": "neutral",
            "keywords": [],
            "community_score": 0.0,
            "comment_count": 0,
        }

    # Aggregate comment text, capped at ~10k chars for API
    comment_texts = [c["text"] for c in comments if c["text"]]
    aggregated = "\n".join(comment_texts)[:10000]

    prompt = f"""Analyze this developer community discussion about a tech company.
    
Story: {story_title}

Comments:
{aggregated}

Extract:
1. Top 3 sentiment themes (e.g., "product quality concerns", "hiring process praise")
2. Overall tone: positive, neutral, or negative
3. Top 5 keywords that indicate sentiment direction
4. Community enthusiasm score (0-100, where 100 is highest enthusiasm)

Return a JSON block with keys: themes (list), tone (str), keywords (list), community_score (int).
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, temperature=0.3)
        raw_text = response.text

        # Parse JSON from response
        import json
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            parsed["comment_count"] = len(comments)
            return parsed
    except Exception as e:
        print(f"Gemini sentiment extraction failed: {e}")

    return {
        "themes": [],
        "tone": "neutral",
        "keywords": [],
        "community_score": 0,
        "comment_count": len(comments),
    }


def scrape_hn_for_company_sentiment(
    company_name: str,
    lookback_days: int = 7,
    max_stories: int = 10
) -> list[dict]:
    """
    Scrape HN for Ask HN posts about a company, extract sentiment signals.
    
    Args:
        company_name: e.g. "OpenAI", "Anthropic", "Google"
        lookback_days: how far back to search (API constraint ~30 days)
        max_stories: max stories to analyze
    
    Returns:
        List of dicts with story_id, title, score, sentiment_data, timestamp.
    """
    results = []
    story_ids = fetch_hn_frontpage()
    if not story_ids:
        return results

    analyzed = 0
    for story_id in story_ids:
        if analyzed >= max_stories:
            break

        story = fetch_hn_story(story_id)
        if not story:
            continue

        title = story.get("title", "")
        text = story.get("text", "")
        score = story.get("score", 0)
        timestamp = story.get("time", 0)

        # Check if relevant
        if not is_ask_hn_about_company(title, text):
            continue

        # Fuzzy company match
        if company_name.lower() not in f"{title} {text}".lower():
            continue

        # Fetch comments
        comments = fetch_hn_comments(story_id, max_depth=1)

        # Extract sentiment
        sentiment = extract_sentiment_from_comments(comments, title)

        results.append({
            "story_id": story_id,
            "title": title,
            "score": score,
            "timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "comment_count": len(comments),
