"""
Hacker News sentiment scraper for Sentinel Scout.

Ingests 'Ask HN' posts mentioning tech companies, extracts developer sentiment
via comment analysis, and scores community perception (bullish/bearish/neutral).
Uses Gemini for high-volume HTML parsing and comment extraction.
Output: structured sentiment records keyed by company mention + timestamp.
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd


# Initialize Gemini client
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def fetch_hn_ask_posts(limit: int = 30) -> list[dict]:
    """Fetch recent 'Ask HN' post IDs from Hacker News frontpage."""
    try:
        resp = requests.get("https://news.ycombinator.com/newest", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[HN Scout] Failed to fetch HN frontpage: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    posts = []
    rows = soup.find_all("tr", class_="athing")

    for row in rows[:limit]:
        title_cell = row.find("span", class_="titleline")
        if not title_cell:
            continue

        link = title_cell.find("a")
        if not link:
            continue

        title = link.get_text(strip=True)
        # Filter for "Ask HN" posts
        if not title.lower().startswith("ask hn"):
            continue

        # Extract post ID from the row
        post_id = row.get("id", "").replace("story_", "")
        if not post_id:
            continue

        posts.append({
            "id": post_id,
            "title": title,
            "url": f"https://news.ycombinator.com/item?id={post_id}",
            "fetched_at": datetime.utcnow().isoformat()
        })

    return posts


def fetch_hn_comments(post_id: str) -> list[dict]:
    """Fetch all comments for a given HN post ID."""
    try:
        url = f"https://news.ycombinator.com/item?id={post_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[HN Scout] Failed to fetch post {post_id}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    comments = []

    # HN comment structure: <div class="comment">
    comment_cells = soup.find_all("div", class_="comment")

    for cell in comment_cells:
        text_elem = cell.find("div", class_="commtext")
        if not text_elem:
            continue

        text = text_elem.get_text(strip=True)
        if text:
            comments.append({
                "text": text,
                "post_id": post_id
            })

    return comments


def extract_company_mentions(text: str) -> list[str]:
    """Extract tech company ticker symbols and names from text."""
    # Common tech company patterns
    patterns = [
        r"\b(AAPL|MSFT|GOOGL|AMZN|TSLA|NVDA|META|AMD|INTC|NFLX|PYPL|SNOW|CRM|ADBE|COIN)\b",
        r"\b(Apple|Microsoft|Google|Amazon|Tesla|NVIDIA|Meta|Intel|Netflix|PayPal|Snowflake|Salesforce|Adobe|Coinbase)\b",
    ]

    mentions = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        mentions.extend(matches)

    return list(set(mentions))


def score_comment_sentiment_with_gemini(comment_text: str, company: str) -> dict:
    """
    Use Gemini to score sentiment of a comment toward a specific company.
    Returns {score: -1.0 to 1.0, label: 'bearish'|'neutral'|'bullish', reasoning: str}
    """
    if not GEMINI_API_KEY:
        print("[HN Scout] GEMINI_API_KEY not set; skipping sentiment scoring")
        return {"score": 0.0, "label": "neutral", "reasoning": "API key missing"}

    prompt = f"""Analyze the sentiment of this Hacker News comment toward {company}.
Return a JSON object with:
  - score: a float from -1.0 (bearish) to 1.0 (bullish)
  - label: one of "bearish", "neutral", "bullish"
  - reasoning: a one-sentence explanation

Comment:
"{comment_text}"

Respond ONLY with valid JSON, no markdown or extra text."""

    try:
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
        response = model.generate_content(prompt, request_options={"timeout": 15})
        text = response.text.strip()

        # Parse JSON response
        import json
        # Remove markdown code block if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        result = json.loads(text)
        return {
            "score": float(result.get("score", 0.0)),
            "label": result.get("label", "neutral"),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"[HN Scout] Gemini sentiment scoring failed: {e}")
        return {"score": 0.0, "label": "neutral", "reasoning": f"Error: {e}"}


def scrape_hn_sentiment(hours_back: int = 24, limit_posts: int = 20) -> pd.DataFrame:
    """
    Main scraper: fetch recent Ask HN posts, extract comments, score sentiment.
    Returns DataFrame with columns: [timestamp, company, comment_text, sentiment_score, sentiment_label, reasoning]
    """
    print(f"[HN Scout] Scraping Ask HN posts from last {hours_back} hours...")

    posts = fetch_hn_ask_posts(limit=limit_posts)
    if not posts:
        print("[HN Scout] No Ask HN posts found")
        return pd.DataFrame()

    rows = []

    for post in posts:
        print(f"[HN Scout] Processing post: {post['title']}")
        comments = fetch_hn_comments(post["id"])

        for comment in comments:
            companies = extract_company_mentions(comment["text"])

            for company in companies:
                sentiment = score_comment_sentiment_with_gemini(comment["text"], company)

                rows.append({
                    "timestamp": post["fetched_at"],
                    "post_id": post["id"],
                    "post_title": post["title"],
                    "company": company,
                    "comment_text": comment["text"][:500],  # Truncate for storage
                    "sentiment_score": sentiment["score"],
                    "sentiment_label": sentiment["label"],
                    "reasoning": sentiment["reasoning"]
                })

            # Rate limit to avoid hammering Gemini API
            time.sleep(0.5)

        # Rate limit between posts
        time.sleep(1)

    df = pd.DataFrame(rows)
    print(f"[HN Scout] Extracted {len(df)} sentiment records across {df['company'].nunique()} companies")

    return df


def aggregate_hn_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment scores by company.
    Returns
