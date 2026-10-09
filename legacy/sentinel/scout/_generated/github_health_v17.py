"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests repository signals (stars, commit velocity, issue open rate)
for a given GitHub org/repo to supplement sentiment analysis with developer
ecosystem health. Data is collected via the GitHub REST API and cached locally
to avoid rate-limit exhaustion.

Integrated into scout/ pillar for cross-referencing with price movements and
linguistic drift in linguist/ (e.g., "declining developer interest + negative
sentiment = high confidence short signal").
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional

import requests


# ============================================================================
# Constants & Configuration
# ============================================================================

GITHUB_API_BASE = "https://api.github.com"
DB_PATH = "sentinel_data/github_cache.db"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
RATE_LIMIT_BUFFER = 10  # Keep 10 requests as safety margin


# ============================================================================
# Database Initialization
# ============================================================================

def _init_db() -> None:
    """Initialize SQLite cache table for GitHub repo signals."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS github_signals (
            repo_full_name TEXT PRIMARY KEY,
            stars INTEGER,
            commit_velocity REAL,
            issue_open_rate REAL,
            fetched_at INTEGER,
            raw_commit_count INTEGER,
            raw_issue_count INTEGER,
            raw_open_issue_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _get_cached_signal(repo: str, max_age_seconds: int = 3600) -> Optional[dict]:
    """
    Retrieve cached repo signal if fresh; return None if stale or missing.
    
    Args:
        repo: Full repository name (owner/name).
        max_age_seconds: Invalidate cache older than this.
    
    Returns:
        Dict with stars, commit_velocity, issue_open_rate; None if miss.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT stars, commit_velocity, issue_open_rate, fetched_at FROM github_signals WHERE repo_full_name = ?",
        (repo,)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    stars, velocity, rate, fetched_at = row
    age = int(time.time()) - fetched_at
    if age > max_age_seconds:
        return None
    
    return {
        "stars": stars,
        "commit_velocity": velocity,
        "issue_open_rate": rate,
        "cached_age_seconds": age
    }


def _store_signal(
    repo: str,
    stars: int,
    commit_velocity: float,
    issue_open_rate: float,
    commit_count: int,
    issue_count: int,
    open_issue_count: int
) -> None:
    """Store repo signals in SQLite cache."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO github_signals
        (repo_full_name, stars, commit_velocity, issue_open_rate, fetched_at,
         raw_commit_count, raw_issue_count, raw_open_issue_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        repo, stars, commit_velocity, issue_open_rate, int(time.time()),
        commit_count, issue_count, open_issue_count
    ))
    conn.commit()
    conn.close()


# ============================================================================
# GitHub API Helpers
# ============================================================================

def _make_request(endpoint: str, params: Optional[dict] = None) -> Optional[dict]:
    """
    Execute authenticated GitHub REST API request with error handling.
    
    Args:
        endpoint: URL path (e.g., "/repos/owner/name").
        params: Query parameters.
    
    Returns:
        Parsed JSON response; None on error.
    """
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    url = GITHUB_API_BASE + endpoint
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
        if resp.status_code == 429:
            remaining = int(resp.headers.get("X-RateLimit-Remaining", 0))
            reset_ts = int(resp.headers.get("X-RateLimit-Reset", 0))
            if remaining < RATE_LIMIT_BUFFER:
                reset_dt = datetime.fromtimestamp(reset_ts)
                print(f"[GitHub] Rate limit approaching. Reset at {reset_dt}")
                return None
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"[GitHub] API error on {endpoint}: {e}")
        return None


def _get_repo_info(repo: str) -> Optional[dict]:
    """Fetch repository metadata (stars, created_at, pushed_at)."""
    data = _make_request(f"/repos/{repo}")
    if not data:
        return None
    return {
        "stars": data.get("stargazers_count", 0),
        "created_at": data.get("created_at"),
        "pushed_at": data.get("pushed_at"),
    }


def _get_commit_velocity(repo: str, days: int = 30) -> Optional[float]:
    """
    Compute commits per week over trailing days via commit search.
    
    Args:
        repo: Full repository name.
        days: Trailing window (default 30).
    
    Returns:
        Commits per week; None on error.
    """
    since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    query = f"repo:{repo} committer-date:>{since_date}"
    params = {"q": query, "per_page": 1}
    
    data = _make_request("/search/commits", params)
    if not data:
        return None
    
    total_commits = data.get("total_count", 0)
    weeks = days / 7.0
    return total_commits / weeks if weeks > 0 else 0.0


def _get_issue_open_rate(repo: str) -> Optional[tuple]:
    """
    Compute open issue ratio and raw counts.
    
    Returns:
        Tuple of (open_rate: float, total_issue_count: int, open_count: int).
    """
    # Fetch open issues
    open_data = _make_request(f"/repos/{repo}/issues", {"state": "open", "per_page": 1})
    if not open_data:
        return None
    open_count = open_data[0].get("number", 0) if open_data else 0
    
    # Fetch closed issues (approximate via search)
    query = f"repo:{repo} is:issue is:closed"
    params = {"q": query, "per_page": 1}
    closed_data = _make_request("/search/issues", params)
    closed_count = closed_data.get("total_count", 0) if closed_data else 0
    
    total = open_count + closed_count
    rate = (open_count / total) if total > 0 else 0.0
    
    return rate, total, open_count


# ============================================================================
# Public Interface
# ============================================================================

def fetch_github_signals(repo: str, use_cache: bool = True, cache_ttl_seconds: int = 3600) -> dict:
    """
    Collect GitHub repo health signals (stars, commit velocity, issue rate).
    
    Queries GitHub REST API with caching to avoid rate limits. Returns
    structured dict suitable for downstream linguist/ and
