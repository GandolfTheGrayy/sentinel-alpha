"""
Hacker News scraper for Sentinel Scout.

Fetches 'Ask HN' posts mentioning tech companies, extracts sentiment signals
from comment threads, and scores developer community perception. Uses Gemini
for HTML parsing and comment extraction due to HN's dynamic structure.

Integrated into the Scout pillar to capture niche developer sentiment that
often precedes mainstream market moves in tech stocks.
"""

import os
import time
from typing import Optional
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


def _get_gemini_client() -> genai.GenerativeModel:
    """Initialize Gemini client from GEMINI_API_KEY env var."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def fetch_ask_hn_posts(limit: int = 20) -> list[dict]:
    """
    Fetch recent 'Ask HN' posts from Hacker News.
    
    Args:
        limit: Maximum number of posts to fetch.
    
    Returns:
        List of dicts with keys: id, title, url, points, num_comments, author, time_ago.
    """
    url = "https://news.ycombinator.com/newest?p=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching HN frontpage: {e}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    posts = []
    
    for row in soup.find_all("tr", class_="athing"):
        title_cell = row.find("span", class_="titleline")
        if not title_cell:
            continue
        
        title_text = title_cell.get_text(strip=True)
        
        # Filter for "Ask HN" posts
        if not title_text.lower().startswith("ask hn"):
            continue
        
        post_id = row.get("id", "")
        link = title_cell.find("a")
        post_url = link.get("href", "") if link else ""
        
        # Get metadata (points, comments) from next row
        meta_row = row.find_next("tr")
        if meta_row:
            meta_text = meta_row.get_text(strip=True)
            parts = meta_text.split()
            points = int(parts[0]) if parts and parts[0].isdigit() else 0
            num_comments = 0
            for i, part in enumerate(parts):
                if "comment" in part.lower() and i > 0:
                    try:
                        num_comments = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
                    break
        else:
            points = 0
            num_comments = 0
        
        posts.append({
            "id": post_id,
            "title": title_text,
            "url": f"https://news.ycombinator.com/item?id={post_id}",
            "points": points,
            "num_comments": num_comments,
            "author": "",
            "time_ago": ""
        })
        
        if len(posts) >= limit:
            break
    
    return posts


def fetch_post_comments(post_id: str, limit: int = 30) -> list[str]:
    """
    Fetch top comments from a single HN post.
    
    Args:
        post_id: Hacker News post ID.
        limit: Maximum number of comments to fetch.
    
    Returns:
        List of comment text strings.
    """
    url = f"https://news.ycombinator.com/item?id={post_id}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching HN post {post_id}: {e}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    comments = []
    
    # HN comments are in <span class="commtext c00">
    for span in soup.find_all("span", class_="commtext"):
        comment_text = span.get_text(strip=True)
        if comment_text:
            comments.append(comment_text)
        
        if len(comments) >= limit:
            break
    
    return comments


def extract_company_mentions(text: str) -> list[str]:
    """
    Extract tech company mentions from text using simple keyword matching.
    
    Args:
        text: Input text to scan.
    
    Returns:
        List of company names/tickers found.
    """
    companies = [
        "apple", "aapl",
        "google", "alphabet", "googl", "goog",
        "microsoft", "msft",
        "meta", "facebook", "fb",
        "tesla", "tsla",
        "nvidia", "nvda",
        "amazon", "amzn",
        "openai", "chatgpt",
        "anthropic",
        "stripe",
        "figma",
        "notion",
        "vercel",
        "fly.io",
        "cloudflare", "net",
        "shopify", "shop",
        "datadog", "ddog",
        "elastic", "estc",
        "mongo", "mdb",
        "docker",
        "kubernetes",
        "rust",
        "python",
        "javascript",
        "golang",
    ]
    
    text_lower = text.lower()
    found = []
    for company in companies:
        if company in text_lower:
            found.append(company)
    
    return list(set(found))


def score_sentiment_with_gemini(
    title: str,
    comments: list[str],
    companies: list[str]
) -> dict:
    """
    Use Gemini to score developer sentiment from HN post and comments.
    
    Args:
        title: 'Ask HN' post title.
        comments: List of comment text strings.
        companies: List of company names/tickers mentioned.
    
    Returns:
        Dict with keys: sentiment_score (float -1 to 1), confidence, summary, flags.
    """
    client = _get_gemini_client()
    
    comments_text = "\n---\n".join(comments[:10])  # Limit to first 10
    
    prompt = f"""
You are analyzing developer community sentiment from a Hacker News 'Ask HN' post.

Post Title: {title}

Top Comments:
{comments_text}

Companies/Products Mentioned: {', '.join(companies)}

Score the overall sentiment from developers on a scale of -1 (very negative) to +1 (very positive).
Consider:
- Technical concerns vs. praise
- Adoption momentum
- Trust and reliability signals
- Frustration with pricing, support, or product direction

Respond with a JSON object:
{{
  "sentiment_score": <float -1 to 1>,
  "confidence": <float 0 to 1>,
  "summary": "<one sentence summary>",
  "flags": ["<concern1>", "<concern2>"]
}}
"""
    
    try:
        response = client.generate_content(prompt)
        text = response.text.strip()
        
        # Extract JSON from response
        import json
        start = text.find("{")
