"""
Hacker News sentiment scraper for Sentinel Scout pillar.

Targets 'Ask HN' posts mentioning tech companies, extracts developer community
sentiment signals (enthusiasm, concerns, adoption signals), and returns structured
data for linguistic analysis. Uses Gemini for high-volume HTML parsing and
sentiment extraction from HN thread comments.

Integrated into the Scout data-ingestion layer to feed Linguist reasoning modules.
"""

import os
import time
import sqlite3
from typing import Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import pandas as pd


def _get_hn_frontpage_urls() -> list[str]:
    """Fetch top 30 Hacker News story URLs from frontpage."""
    url = "https://news.ycombinator.com/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        story_rows = soup.select("tr.athing")
        urls = []
        for row in story_rows[:30]:
            link_elem = row.select_one("span.titleline > a")
            if link_elem:
                story_url = link_elem.get("href", "")
                if story_url:
                    urls.append(story_url)
        return urls
    except Exception as e:
        print(f"Error fetching HN frontpage: {e}")
        return []


def _get_hn_ask_threads(limit: int = 50) -> list[dict]:
    """
    Fetch 'Ask HN' thread metadata from Hacker News.
    
    Returns list of dicts with 'id', 'title', 'url', 'points', 'comments'.
    """
    try:
        # Query HN API for Ask HN posts
        url = "https://hacker-news.firebaseio.com/v0/askstories.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        story_ids = response.json()[:limit]
        
        threads = []
        for story_id in story_ids:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            try:
                item_response = requests.get(item_url, timeout=5)
                item_response.raise_for_status()
                item = item_response.json()
                if item and item.get("type") == "story":
                    threads.append({
                        "id": story_id,
                        "title": item.get("title", ""),
                        "url": f"https://news.ycombinator.com/item?id={story_id}",
                        "points": item.get("score", 0),
                        "comments": item.get("descendants", 0),
                        "timestamp": item.get("time", 0)
                    })
            except Exception as e:
                print(f"Error fetching item {story_id}: {e}")
                continue
            time.sleep(0.1)  # Rate limit
        return threads
    except Exception as e:
        print(f"Error fetching Ask HN threads: {e}")
        return []


def _extract_hn_thread_comments(thread_id: int, max_comments: int = 50) -> list[str]:
    """Fetch comment text from a single HN thread via HN API."""
    comments = []
    try:
        item_url = f"https://hacker-news.firebaseio.com/v0/item/{thread_id}.json"
        response = requests.get(item_url, timeout=5)
        response.raise_for_status()
        item = response.json()
        
        if not item:
            return comments
        
        kid_ids = item.get("kids", [])[:max_comments]
        for kid_id in kid_ids:
            try:
                kid_url = f"https://hacker-news.firebaseio.com/v0/item/{kid_id}.json"
                kid_response = requests.get(kid_url, timeout=5)
                kid_response.raise_for_status()
                kid_item = kid_response.json()
                if kid_item and kid_item.get("type") == "comment":
                    text = kid_item.get("text", "")
                    if text:
                        comments.append(text)
            except Exception as e:
                print(f"Error fetching kid {kid_id}: {e}")
                continue
            time.sleep(0.05)
        return comments
    except Exception as e:
        print(f"Error extracting comments from thread {thread_id}: {e}")
        return []


def score_hn_sentiment(
    thread_title: str,
    comments: list[str],
    company_name: str
) -> dict:
    """
    Use Gemini to score developer sentiment in HN comments about a company.
    
    Returns dict with 'enthusiasm', 'concern', 'adoption_signal', 'raw_analysis'.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    
    # Truncate for API limits
    comment_text = "\n---\n".join(comments[:20])
    
    prompt = f"""
Analyze developer sentiment in these Hacker News comments about {company_name}.

Thread Title: {thread_title}

Comments:
{comment_text}

Score on 0–10 scales:
1. Enthusiasm (0=dismissive, 10=excited adoption)
2. Concern (0=no concerns, 10=critical issues)
3. Adoption Signal (0=no traction, 10=strong adoption signal)

Respond ONLY as JSON:
{{
  "enthusiasm": <int>,
  "concern": <int>,
  "adoption_signal": <int>,
  "summary": "<one-sentence summary>"
}}
"""
    
    try:
        response = model.generate_content(prompt, temperature=0.3)
        text = response.text.strip()
        
        # Parse JSON from response
        import json
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            result = json.loads(json_str)
            return {
                "enthusiasm": result.get("enthusiasm", 5),
                "concern": result.get("concern", 5),
                "adoption_signal": result.get("adoption_signal", 5),
                "summary": result.get("summary", "")
            }
    except Exception as e:
        print(f"Error scoring sentiment with Gemini: {e}")
    
    return {
        "enthusiasm": 5,
        "concern": 5,
        "adoption_signal": 5,
        "summary": "Error in sentiment scoring"
    }


def scrape_hn_for_company(
    company_name: str,
    days_back: int = 7,
    max_threads: int = 10
) -> pd.DataFrame:
    """
    Scrape Hacker News for 'Ask HN' posts mentioning company, score sentiment.
    
    Returns DataFrame with columns: thread_id, title, sentiment_scores, timestamp.
    """
    threads = _get_hn_ask_threads(limit=100)
    
    # Filter for threads mentioning company name
    relevant = [
        t for t in threads
        if company_name.lower() in t["title"].lower()
    ][:max_threads]
    
    results = []
    for thread in relevant:
        print(f"Processing HN thread: {thread['title']}")
        comments = _extract_hn_thread_comments(thread["id"], max_comments=50)
        
        if comments:
            sentiment = score_hn_sentiment(thread["title"], comments, company_name)
            results.append({
                "thread_id": thread["id"],
                "title": thread["
