"""
Hacker News scraper for Sentinel Scout pillar.

Ingests 'Ask HN' posts mentioning tech companies, extracts developer sentiment
signals (enthusiasm, concern, technical depth), and returns scored sentiment
vectors for downstream Linguist analysis. Uses Gemini for HTML parsing and
comment extraction to avoid brittle DOM selectors.
"""

import os
import re
import time
from typing import Optional
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup


# Configure Gemini for high-volume parsing
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
HN_BASE_URL = "https://news.ycombinator.com"


def fetch_ask_hn_posts(
    limit: int = 30, company_filter: Optional[str] = None
) -> list[dict]:
    """
    Fetch recent 'Ask HN' posts from Hacker News frontpage.
    
    Args:
        limit: Maximum posts to retrieve (HN frontpage default ~30).
        company_filter: Optional substring to filter posts (e.g. "Apple", "Meta").
    
    Returns:
        List of post dicts with keys: id, title, url, points, comments_count, author.
    """
    try:
        resp = requests.get(f"{HN_BASE_URL}/newest", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[HN Scout] Fetch error: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    posts = []
    
    # HN HTML structure: each post is a <tr> with class "athing"
    for row in soup.find_all("tr", class_="athing")[:limit]:
        title_cell = row.find("span", class_="titleline")
        if not title_cell:
            continue
        
        title_text = title_cell.get_text(strip=True)
        
        # Filter for "Ask HN" pattern
        if not title_text.lower().startswith("ask hn"):
            continue
        
        # Optional company filter
        if company_filter and company_filter.lower() not in title_text.lower():
            continue
        
        post_id = row.get("id", "")
        url = f"{HN_BASE_URL}/item?id={post_id}" if post_id else None
        
        # Extract metadata from next <tr> row
        metadata_row = row.find_next("tr")
        if metadata_row:
            meta_text = metadata_row.get_text(strip=True)
            # Pattern: "123 points by author 2 hours ago | 45 comments"
            points_match = re.search(r"(\d+)\s+points?", meta_text)
            comments_match = re.search(r"(\d+)\s+comments?", meta_text)
            author_match = re.search(r"by\s+(\S+)", meta_text)
            
            points = int(points_match.group(1)) if points_match else 0
            comments_count = int(comments_match.group(1)) if comments_match else 0
            author = author_match.group(1) if author_match else "unknown"
        else:
            points = 0
            comments_count = 0
            author = "unknown"
        
        posts.append({
            "id": post_id,
            "title": title_text,
            "url": url,
            "points": points,
            "comments_count": comments_count,
            "author": author,
        })
    
    return posts


def fetch_post_comments(post_id: str) -> list[dict]:
    """
    Fetch top-level comments for a given HN post.
    
    Args:
        post_id: Hacker News post ID.
    
    Returns:
        List of comment dicts with keys: author, text, points.
    """
    try:
        resp = requests.get(
            f"{HN_BASE_URL}/item?id={post_id}", timeout=10
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[HN Scout] Comment fetch error: {e}")
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    comments = []
    
    # HN comments: <div class="comment"> with nested structure
    for comment_div in soup.find_all("div", class_="comment")[:20]:  # Top 20 comments
        # Author and points in <span class="comhead">
        comhead = comment_div.find("span", class_="comhead")
        if not comhead:
            continue
        
        comhead_text = comhead.get_text(strip=True)
        author_match = re.search(r"(\S+)\s+", comhead_text)
        author = author_match.group(1) if author_match else "unknown"
        points_match = re.search(r"(\d+)\s+points?", comhead_text)
        points = int(points_match.group(1)) if points_match else 0
        
        # Comment body in <span class="comment">
        comment_text_span = comment_div.find("span", class_="commtext c00")
        comment_text = (
            comment_text_span.get_text(strip=True)
            if comment_text_span
            else ""
        )
        
        if comment_text:
            comments.append({
                "author": author,
                "text": comment_text,
                "points": points,
            })
    
    return comments


def score_sentiment_with_gemini(text: str) -> dict:
    """
    Use Gemini to extract developer sentiment from post/comment text.
    
    Analyzes for: enthusiasm (1-5), concern (1-5), technical_depth (1-5),
    and extracts keywords (tech stack, pain points, praise).
    
    Args:
        text: Post title, comment, or aggregated text snippet.
    
    Returns:
        Dict with keys: enthusiasm, concern, technical_depth, keywords, raw_analysis.
    """
    if not text or len(text.strip()) < 10:
        return {
            "enthusiasm": 0,
            "concern": 0,
            "technical_depth": 0,
            "keywords": [],
            "raw_analysis": "",
        }
    
    prompt = f"""Analyze this developer/tech community post for sentiment signals.
Score on 1-5 scales:
- Enthusiasm: confidence, optimism, excitement about the tech/company
- Concern: skepticism, worry, red flags, frustration
- Technical Depth: how substantive/technical is the discussion

Also extract:
- keywords: tech stack, product names, pain points, praised features (comma-separated)

Text:
{text}

Respond in this exact format:
ENTHUSIASM: <1-5>
CONCERN: <1-5>
TECHNICAL_DEPTH: <1-5>
KEYWORDS: <comma-separated>
ANALYSIS: <one sentence summary>
"""
    
    try:
        response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
        analysis_text = response.text.strip()
    except Exception as e:
        print(f"[HN Scout] Gemini error: {e}")
        return {
            "enthusiasm": 0,
            "concern": 0,
            "technical_depth": 0,
            "keywords": [],
            "raw_analysis": "",
        }
    
    # Parse structured response
    lines = analysis_text.split("\n")
    result = {
        "enthusiasm": 0,
        "concern": 0,
        "technical_depth": 0,
        "keywords": [],
        "raw_analysis": analysis_text,
    }
    
    for line in lines:
        if line.startswith("ENTHUSIASM:"):
            match = re.search(r"\d+", line)
            result["enthusiasm"] = int
