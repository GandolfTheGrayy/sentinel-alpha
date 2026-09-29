"""
Hacker News scraper for Sentinel Scout pillar.

Ingests 'Ask HN' posts mentioning tech companies, extracts developer sentiment
from comments, and scores community perception. Uses Gemini for HTML parsing
and comment analysis. Results feed into the Linguist pillar for tone analysis.
"""

import os
import re
import time
from typing import Optional
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


def _get_gemini_client() -> genai.GenerativeModel:
    """Initialize and return Gemini client from GEMINI_API_KEY env var."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-3.1-flash-lite-preview")


def fetch_ask_hn_posts(limit: int = 10) -> list[dict]:
    """Fetch recent 'Ask HN' posts from Hacker News frontpage."""
    url = "https://news.ycombinator.com/newest?p=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch HN frontpage: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    posts = []
    rows = soup.find_all("tr", class_="athing")

    for row in rows[:limit * 2]:  # Fetch extra to filter for "Ask HN"
        title_cell = row.find("span", class_="titleline")
        if not title_cell:
            continue
        title_text = title_cell.get_text(strip=True)
        if not title_text.lower().startswith("ask hn"):
            continue

        post_id_elem = row.get("id")
        post_id = post_id_elem if post_id_elem else None

        if len(posts) < limit:
            posts.append({"id": post_id, "title": title_text, "url": f"https://news.ycombinator.com/item?id={post_id}"})

    return posts


def fetch_post_comments(post_id: str) -> list[dict]:
    """Fetch all comments from a single HN post."""
    if not post_id:
        return []
    url = f"https://news.ycombinator.com/item?id={post_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch post {post_id}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    comments = []
    comment_cells = soup.find_all("td", class_="comment")

    for cell in comment_cells[:50]:  # Cap at 50 comments per post
        comment_text = cell.get_text(strip=True)
        if comment_text:
            comments.append({"text": comment_text})

    return comments


def extract_company_mentions(text: str) -> list[str]:
    """Extract potential tech company names (uppercase acronyms, common names) from text."""
    tech_companies = {
        "Apple",
        "Google",
        "Microsoft",
        "Amazon",
        "Meta",
        "Tesla",
        "Netflix",
        "NVIDIA",
        "AMD",
        "Intel",
        "Stripe",
        "Shopify",
        "Airbnb",
        "Uber",
        "Lyft",
        "Slack",
        "Zoom",
        "Figma",
        "Discord",
        "Notion",
        "Canva",
        "OpenAI",
        "Anthropic",
        "Databricks",
        "Hugging Face",
    }
    found = []
    for company in tech_companies:
        if re.search(r"\b" + re.escape(company) + r"\b", text, re.IGNORECASE):
            found.append(company)
    return found


def analyze_comment_sentiment(comment_text: str, company_name: str) -> Optional[dict]:
    """
    Use Gemini to analyze sentiment of a comment toward a specific company.
    Returns dict with sentiment, confidence, and key phrases or None if error.
    """
    client = _get_gemini_client()
    prompt = f"""Analyze the sentiment of this developer community comment about {company_name}.

Comment: "{comment_text}"

Respond in JSON format only:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "confidence": 0.0 to 1.0,
  "key_phrase": "short phrase capturing main sentiment"
}}

Be strict: only call it positive if praise is clear, negative if criticism is clear."""

    try:
        response = client.generate_content(prompt, generation_config={"temperature": 0.2})
        if response.text:
            import json
            result = json.loads(response.text)
            return result
    except Exception as e:
        print(f"Gemini analysis failed: {e}")
    return None


def score_developer_sentiment(
    post_title: str, comments: list[dict], company_name: str
) -> dict:
    """Score overall developer sentiment for a company from HN comments."""
    sentiments = []
    for comment in comments:
        analysis = analyze_comment_sentiment(comment["text"], company_name)
        if analysis:
            sentiments.append(analysis)
        time.sleep(0.1)  # Rate limit

    if not sentiments:
        return {
            "company": company_name,
            "post_title": post_title,
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "comment_count": 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

    positive_count = sum(1 for s in sentiments if s.get("sentiment") == "positive")
    negative_count = sum(1 for s in sentiments if s.get("sentiment") == "negative")
    neutral_count = sum(1 for s in sentiments if s.get("sentiment") == "neutral")
    avg_confidence = sum(s.get("confidence", 0) for s in sentiments) / len(sentiments)

    net_sentiment = (positive_count - negative_count) / len(sentiments)

    return {
        "company": company_name,
        "post_title": post_title,
        "sentiment_score": net_sentiment,
        "confidence": avg_confidence,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "comment_count": len(sentiments),
        "timestamp": datetime.utcnow().isoformat(),
    }


def scrape_and_score_hn(target_companies: Optional[list[str]] = None) -> list[dict]:
    """
    Main entry point: scrape Ask HN posts, extract company mentions,
    analyze sentiment per company, return scored results.
    """
    if not target_companies:
        target_companies = [
            "Apple",
            "Google",
            "Microsoft",
            "Amazon",
            "Meta",
            "Tesla",
            "NVIDIA",
        ]

    posts = fetch_ask_hn_posts(limit=10)
    results = []

    for post in posts:
        comments = fetch_post_comments(post["id"])
        mentioned_companies = set
