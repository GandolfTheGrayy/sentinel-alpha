"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests GitHub repository signals (stars, commit velocity, issue open rate)
as sentiment proxies for developer ecosystem health. High commit velocity and star growth
often precede positive sentiment shifts in tech stocks; rising issue counts may signal
technical debt or user dissatisfaction. Signals feed into the Linguist for weighting
and the Judge for prediction context.

Uses the GitHub REST API (no auth required for public repos, rate-limited to 60 req/hr).
Results are cached in SQLite to minimize API calls during backtest windows.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import requests


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────────────────────

def _init_github_cache() -> str:
    """Initialize SQLite cache for GitHub signals; return path to DB file."""
    db_path = os.path.expanduser("~/.sentinel/github_health.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS github_snapshots (
            repo_owner TEXT,
            repo_name TEXT,
            timestamp INTEGER,
            stars INTEGER,
            commit_count_week INTEGER,
            open_issues INTEGER,
            PRIMARY KEY (repo_owner, repo_name, timestamp)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_repo_time 
        ON github_snapshots(repo_owner, repo_name, timestamp DESC)
    """)
    conn.commit()
    conn.close()
    return db_path


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB API CALLS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_repo_stats(owner: str, repo: str) -> Optional[Dict[str, Any]]:
    """
    Fetch current GitHub repo stats: stars, commit velocity (past 7d), open issues.
    
    Returns dict with keys: stars, open_issues, commits_last_week, timestamp.
    Returns None on API error or rate limit (will use cache instead).
    """
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    
    try:
        # Main repo data
        repo_url = f"https://api.github.com/repos/{owner}/{repo}"
        resp = requests.get(repo_url, headers=headers, timeout=10)
        resp.raise_for_status()
        repo_data = resp.json()
        
        stars = repo_data.get("stargazers_count", 0)
        open_issues = repo_data.get("open_issues_count", 0)
        
        # Commit velocity: count commits in last 7 days
        since_date = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
        commits_url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        commits_resp = requests.get(
            commits_url,
            params={"since": since_date, "per_page": 100},
            headers=headers,
            timeout=10
        )
        commits_resp.raise_for_status()
        commits_last_week = len(commits_resp.json())
        
        return {
            "stars": stars,
            "open_issues": open_issues,
            "commits_last_week": commits_last_week,
            "timestamp": int(time.time()),
        }
    except Exception as e:
        print(f"[GitHub] Error fetching {owner}/{repo}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CACHE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_snapshot(db_path: str, owner: str, repo: str, stats: Dict[str, Any]) -> None:
    """Write a snapshot to the GitHub cache DB."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        INSERT OR REPLACE INTO github_snapshots
        (repo_owner, repo_name, timestamp, stars, commit_count_week, open_issues)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (owner, repo, stats["timestamp"], stats["stars"],
          stats["commits_last_week"], stats["open_issues"]))
    conn.commit()
    conn.close()


def _get_cached_snapshot(db_path: str, owner: str, repo: str, 
                         max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Retrieve most recent cached snapshot for a repo, if fresh (< max_age_hours old).
    Returns None if no cache hit or cache expired.
    """
    cutoff = int(time.time()) - (max_age_hours * 3600)
    conn = sqlite3.connect(db_path)
    row = conn.execute("""
        SELECT stars, commit_count_week, open_issues, timestamp
        FROM github_snapshots
        WHERE repo_owner = ? AND repo_name = ? AND timestamp > ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (owner, repo, cutoff)).fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "stars": row[0],
        "commits_last_week": row[1],
        "open_issues": row[2],
        "timestamp": row[3],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def get_github_signals(owner: str, repo: str, max_cache_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Get GitHub health signals for a repo, using cache when available.
    
    Args:
        owner: GitHub repo owner (e.g., "openai").
        repo: GitHub repo name (e.g., "gpt-4").
        max_cache_age_hours: Max age of cached data before forcing refresh.
    
    Returns dict with keys:
        - stars: Total GitHub stars.
        - open_issues: Count of open GitHub issues.
        - commits_last_week: Commit count in the past 7 days.
        - timestamp: Unix timestamp of data collection.
    
    Returns None if API fails and no cache available.
    """
    db_path = _init_github_cache()
    
    # Try cache first
    cached = _get_cached_snapshot(db_path, owner, repo, max_cache_age_hours)
    if cached:
        return cached
    
    # Fetch fresh data
    fresh = fetch_repo_stats(owner, repo)
    if fresh:
        _cache_snapshot(db_path, owner, repo, fresh)
        return fresh
    
    # Fallback: return stale cache if available
    conn = sqlite3.connect(db_path)
    row = conn.execute("""
        SELECT stars, commit_count_week, open_issues, timestamp
        FROM github_snapshots
        WHERE repo_owner = ? AND repo_name = ?
        ORDER BY timestamp DESC
        LIMIT 1
    """, (owner, repo)).fetchone()
    conn.close()
    
    if row:
        return {
            "stars": row[0],
            "commits_last_week": row[1],
            "open_issues": row[2],
            "timestamp": row[3],
        }
    
    return None


def compute_velocity_trend(owner: str, repo: str) -> Optional[float]:
    """
    Compute week-over-week commit velocity trend as a percentage change.
    
    Requires at least 2
