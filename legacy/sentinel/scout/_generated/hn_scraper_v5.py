"""
Hacker News sentiment scraper for Sentinel Scout.

Fetches 'Ask HN' posts mentioning tech companies, extracts developer sentiment
signals (enthusiasm, concern, adoption signals), and returns structured data
for upstream Linguist analysis. Uses Gemini for robust HTML parsing and
comment extraction.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


@dataclass
class HNComment:
    """A single HN comment with sentiment markers."""
    author: str
    text: str
    score: int
    timestamp: str


@dataclass
class HNPost:
    """A single 'Ask HN' post about a tech company."""
    post_id: str
    title: str
    url: str
    author: str
    score: int
    comment_count: int
    timestamp: str
    company_mentions: list[str]
    comments: list[HNComment]
    raw_sentiment_score: float


def fetch_hn_frontpage() -> list[dict]:
    """Fetch latest HN posts from the front page via official API."""
    try:
        resp = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10)
        resp.raise_for_status()
        top_ids = resp.json()[:30]
        return top_ids
    except Exception as e:
        print(f"[HN] Error fetching frontpage: {e}")
        return []


def fetch_hn_post_details(post_id: int) -> Optional[dict]:
    """Fetch full post details including comments via HN API."""
    try:
        resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{post_id}.json", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[HN] Error fetching post {post_id}: {e}")
        return None


def fetch_hn_comment_details(comment_id: int) -> Optional[dict]:
    """Fetch a single comment via HN API."""
    try:
        resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[HN] Error fetching comment {comment_id}: {e}")
        return None


def is_ask_hn_tech_post(title: str, company_keywords: list[str]) -> bool:
    """Check if post is 'Ask HN' about tech companies."""
    title_lower = title.lower()
    if not title_lower.startswith("ask hn"):
        return False
    for keyword in company_keywords:
        if keyword.lower() in title_lower:
            return True
    return False


def extract_company_mentions(text: str, company_keywords: list[str]) -> list[str]:
    """Extract company mentions from text."""
    text_lower = text.lower()
    mentions = []
    for keyword in company_keywords:
        if keyword.lower() in text_lower:
            mentions.append(keyword)
    return list(set(mentions))


def analyze_sentiment_with_gemini(post_title: str, comments_text: str) -> float:
    """Use Gemini to score developer sentiment (0.0=negative, 1.0=positive)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[HN] GEMINI_API_KEY not set, returning neutral score")
        return 0.5

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""Analyze the following Hacker News 'Ask HN' post and developer comments.
Score the overall developer sentiment on a scale of 0.0 (very negative/dismissive) to 1.0 (very positive/enthusiastic).

Title: {post_title}

Comments (first 3000 chars):
{comments_text[:3000]}

Respond with ONLY a single float between 0.0 and 1.0, e.g., 0.73"""

    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.3})
        score_str = response.text.strip()
        score = float(score_str)
        return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[HN] Gemini sentiment analysis failed: {e}")
        return 0.5


def scrape_hn_sentiment(
    company_keywords: list[str],
    hours_back: int = 24,
    max_posts: int = 10
) -> list[HNPost]:
    """
    Scrape HN for Ask HN posts about tech companies and score developer sentiment.

    Args:
        company_keywords: List of company names/keywords to search for (e.g., ['OpenAI', 'Anthropic'])
        hours_back: Only consider posts from the last N hours
        max_posts: Maximum number of posts to return

    Returns:
        List of HNPost objects with sentiment scores
    """
    cutoff_timestamp = time.time() - (hours_back * 3600)
    results = []

    print(f"[HN] Fetching top stories...")
    post_ids = fetch_hn_frontpage()

    for post_id in post_ids[:50]:
        if len(results) >= max_posts:
            break

        post_data = fetch_hn_post_details(post_id)
        if not post_data:
            continue

        post_time = post_data.get("time", 0)
        if post_time < cutoff_timestamp:
            continue

        title = post_data.get("title", "")
        if not is_ask_hn_tech_post(title, company_keywords):
            continue

        print(f"[HN] Found matching post: {title[:60]}...")

        company_mentions = extract_company_mentions(title, company_keywords)
        if not company_mentions:
            continue

        comments_data = []
        for comment_id in post_data.get("kids", [])[:20]:
            comment_data = fetch_hn_comment_details(comment_id)
            if comment_data and comment_data.get("type") == "comment":
                try:
                    comments_data.append(
                        HNComment(
                            author=comment_data.get("by", "unknown"),
                            text=comment_data.get("text", ""),
                            score=comment_data.get("score", 0),
                            timestamp=datetime.fromtimestamp(comment_data.get("time", 0)).isoformat()
                        )
                    )
                except Exception as e:
                    print(f"[HN] Error parsing comment: {e}")
                    continue

        comments_combined = " ".join([c.text for c in comments_data])
        sentiment_score = analyze_sentiment_with_gemini(title, comments_combined)

        hn_post = HNPost(
            post_id=str(post_id),
            title=title,
            url=f"https://news.ycombinator.com/item?id={post_id}",
            author=post_data.get("by", "unknown"),
            score=post_data.get("score", 0),
            comment_count=len(comments_data),
            timestamp=datetime.fromtimestamp(post_time).isoformat(),
            company_mentions=company_mentions,
            comments=comments_data,
            raw_sentiment_score=sentiment_score
        )

        results.append(hn_post)
        time.sleep(0.5)

    print(f"[HN] Scraped {len(results)} posts")
    return results


def to_json(posts: list[HNPost]) -> str
