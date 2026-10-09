"""
Hacker News scraper for Sentinel Scout pillar.

Targets 'Ask HN' posts mentioning tech companies to extract developer community
sentiment signals. Uses Gemini (via google-generativeai) for HTML parsing and
text extraction due to HN's dynamic content and high-volume requirements.

Sentiment scores are stored in ChromaDB alongside metadata for RAG retrieval
by the Historian pillar. Complements live_prices.py, news.py, and sec_filings.py
as a niche signal source for tech-focused ticker analysis.
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import chromadb
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup


def _get_hn_client() -> chromadb.HttpClient | chromadb.PersistentClient:
    """Return ChromaDB client for HN sentiment storage."""
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
        return client
    except Exception:
        persist_path = os.getenv("CHROMADB_PERSIST_PATH", "./chroma_data")
        return chromadb.PersistentClient(path=persist_path)


def _extract_company_mentions(text: str) -> list[str]:
    """Extract likely tech company names from text via simple heuristics."""
    # Common patterns: "Company Name", "COMPANY", or known tickers
    known_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
        "PYPL", "SQ", "CRWD", "OKTA", "SNOW", "DDOG", "TWLO", "SHOP"
    ]
    mentions = []
    for ticker in known_tickers:
        if re.search(rf'\b{ticker}\b', text, re.IGNORECASE):
            mentions.append(ticker)
    return list(set(mentions))


def _score_sentiment_gemini(text: str, company: str) -> dict:
    """
    Use Gemini to score developer sentiment in HN text about a company.
    Returns dict with keys: sentiment_score (0-1), tone, confidence.
    """
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    
    prompt = f"""Analyze the following Hacker News discussion snippet for developer sentiment about {company}.
    
Text: "{text}"

Return a JSON object with:
- sentiment_score (float 0-1, where 0=very negative, 0.5=neutral, 1=very positive)
- tone (string: "bullish", "bearish", "mixed", or "neutral")
- confidence (float 0-1)
- key_phrases (list of 2-3 phrases capturing the sentiment)

Only return valid JSON, no markdown or preamble."""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Gemini sentiment scoring error: {e}")
    
    return {
        "sentiment_score": 0.5,
        "tone": "neutral",
        "confidence": 0.0,
        "key_phrases": []
    }


def fetch_ask_hn_posts(limit: int = 50) -> list[dict]:
    """
    Fetch recent 'Ask HN' posts from HN API with tech company references.
    Returns list of dicts with keys: id, title, url, score, descendants, text.
    """
    posts = []
    try:
        # Fetch top Ask HN story IDs
        url = "https://hacker-news.firebaseio.com/v0/askstories.json"
        response = requests.get(url, timeout=10)
        story_ids = response.json()[:limit * 2]  # Fetch 2x to filter
        
        for story_id in story_ids:
            item_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
            try:
                item_response = requests.get(item_url, timeout=5)
                item = item_response.json()
                
                # Filter for posts with text and likely tech relevance
                if item.get("text"):
                    posts.append({
                        "id": item.get("id"),
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "score": item.get("score", 0),
                        "descendants": item.get("descendants", 0),
                        "text": item.get("text", ""),
                        "time": item.get("time"),
                        "by": item.get("by", "")
                    })
                
                if len(posts) >= limit:
                    break
            except Exception as e:
                print(f"Error fetching HN item {story_id}: {e}")
                continue
            
            time.sleep(0.1)  # Rate limit
        
    except Exception as e:
        print(f"Error fetching HN stories: {e}")
    
    return posts


def score_and_store_posts(posts: list[dict]) -> None:
    """
    Score each post for company sentiment and store in ChromaDB + SQLite.
    """
    client = _get_hn_client()
    collection_name = "hn_sentiment"
    
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Hacker News Ask HN sentiment for tech companies"}
        )
    except Exception as e:
        print(f"Error creating ChromaDB collection: {e}")
        return
    
    # SQLite for historical record
    db_path = os.getenv("SENTINEL_DB_PATH", "./sentinel.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hn_posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            company TEXT,
            sentiment_score REAL,
            tone TEXT,
            confidence REAL,
            hn_score INTEGER,
            descendants INTEGER,
            fetched_at TIMESTAMP,
            by TEXT
        )
    """)
    conn.commit()
    
    for post in posts:
        companies = _extract_company_mentions(post["text"] + " " + post["title"])
        
        for company in companies:
            sentiment_result = _score_sentiment_gemini(post["text"][:1000], company)
            
            doc_id = f"hn_{post['id']}_{company}"
            
            # Store in ChromaDB
            try:
                collection.add(
                    ids=[doc_id],
                    documents=[post["text"][:2000]],
                    metadatas=[{
                        "source": "hacker_news",
                        "company": company,
                        "title": post["title"],
                        "sentiment_score": sentiment_result["sentiment_score"],
                        "tone": sentiment_result["tone"],
                        "confidence": sentiment_result["confidence"],
                        "hn_score": post["score"],
                        "descendants": post["descendants"],
                        "fetched_at": datetime.utcnow().isoformat()
                    }]
                )
            except Exception as e:
                print(f"Error storing in ChromaDB: {e}")
            
            # Store in SQLite
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO hn_posts
                    (id, title, text, company, sentiment_score, tone, confidence,
                     hn_score,
