"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests developer activity signals from public GitHub repositories,
measuring:
  - Repository star count (community adoption proxy)
  - Commit velocity (commits/week, technical momentum)
  - Issue open rate (code quality / maintenance burden)

These signals feed into the Sentinel Scout pillar to detect emerging technical
health trends that may correlate with stock price movements (e.g., rapid commits
on a company's core repo, declining maintainer engagement, growing bug backlog).

Uses the GitHub REST API (unauthenticated for public repos, or with optional
GITHUB_TOKEN env var for higher rate limits). Results are cached locally to
minimize API calls and respect rate limits.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

import requests


# ============================================================================
# Database Setup
# ============================================================================

DB_PATH = os.getenv("SENTINEL_DB_PATH", "sentinel_github_cache.db")


def _init_cache_db() -> None:
    """Initialize local SQLite cache for GitHub API responses."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS github_cache (
            repo_owner TEXT NOT NULL,
            repo_name TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value TEXT NOT NULL,
            timestamp_fetched INTEGER NOT NULL,
            PRIMARY KEY (repo_owner, repo_name, metric_type)
        )
    """)
    conn.commit()
    conn.close()


def _get_cached_value(
    owner: str, repo: str, metric: str, max_age_hours: int = 24
) -> Optional[str]:
    """
    Retrieve cached GitHub metric if fresh enough.
    
    Returns None if not cached or older than max_age_hours.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cutoff = int(time.time()) - (max_age_hours * 3600)
    cursor.execute(
        """
        SELECT value FROM github_cache
        WHERE repo_owner = ? AND repo_name = ? AND metric_type = ?
        AND timestamp_fetched > ?
        """,
        (owner, repo, metric, cutoff),
    )
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None


def _cache_value(owner: str, repo: str, metric: str, value: str) -> None:
    """Store GitHub metric in local cache."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO github_cache
        (repo_owner, repo_name, metric_type, value, timestamp_fetched)
        VALUES (?, ?, ?, ?, ?)
        """,
        (owner, repo, metric, value, int(time.time())),
    )
    conn.commit()
    conn.close()


# ============================================================================
# GitHub API Queries
# ============================================================================

def _get_github_headers() -> Dict[str, str]:
    """Return HTTP headers for GitHub API, with optional auth token."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_star_count(owner: str, repo: str, use_cache: bool = True) -> Optional[int]:
    """
    Fetch GitHub repository star count (community adoption signal).
    
    Returns star count or None on API error.
    """
    metric = "stars"
    
    if use_cache:
        cached = _get_cached_value(owner, repo, metric)
        if cached is not None:
            return int(cached)
    
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_get_github_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        stars = data.get("stargazers_count", 0)
        _cache_value(owner, repo, metric, str(stars))
        return stars
    except Exception as e:
        print(f"[GitHub] Error fetching stars for {owner}/{repo}: {e}")
        return None


def fetch_commit_velocity(
    owner: str, repo: str, weeks: int = 4, use_cache: bool = True
) -> Optional[float]:
    """
    Fetch average commits per week over the past N weeks (technical momentum).
    
    Returns commits/week or None on API error.
    """
    metric = f"commit_velocity_{weeks}w"
    
    if use_cache:
        cached = _get_cached_value(owner, repo, metric)
        if cached is not None:
            return float(cached)
    
    # GitHub API: list commits with per_page & pagination
    since = (datetime.utcnow() - timedelta(weeks=weeks)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    
    try:
        all_commits = 0
        page = 1
        while page <= 3:  # Limit to 3 pages to avoid rate-limit thrashing
            resp = requests.get(
                url,
                params={"since": since, "per_page": 100, "page": page},
                headers=_get_github_headers(),
                timeout=10,
            )
            resp.raise_for_status()
            commits = resp.json()
            if not commits:
                break
            all_commits += len(commits)
            if len(commits) < 100:
                break
            page += 1
        
        velocity = all_commits / weeks if weeks > 0 else 0
        _cache_value(owner, repo, metric, str(velocity))
        return velocity
    except Exception as e:
        print(f"[GitHub] Error fetching commit velocity for {owner}/{repo}: {e}")
        return None


def fetch_issue_open_rate(owner: str, repo: str, use_cache: bool = True) -> Optional[float]:
    """
    Fetch ratio of open to total issues (code quality / maintenance burden).
    
    Returns open_issues / total_issues or None on API error.
    """
    metric = "issue_open_rate"
    
    if use_cache:
        cached = _get_cached_value(owner, repo, metric)
        if cached is not None:
            return float(cached)
    
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_get_github_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        open_issues = data.get("open_issues_count", 0)
        # Total issues = open + closed (approximation via API)
        # GitHub doesn't directly expose closed count; we use a heuristic:
        # For better accuracy, we'd need to count closed issues separately.
        # For now, we'll fetch closed issues count via search.
        
        url_closed = f"https://api.github.com/search/issues"
        resp_closed = requests.get(
            url_closed,
            params={
                "q": f"repo:{owner}/{repo} is:issue is:closed",
                "per_page": 1,
            },
            headers=_get_github_headers(),
            timeout=10,
        )
        resp_closed.raise_for_status()
        closed_issues = resp_closed.json().get("total_count", 0)
        
        total_issues = open_issues + closed_issues
        rate = open_issues / total_issues if total_issues > 0 else 0.0
        
        _cache_value(owner, repo, metric, str(rate))
        return rate
    except Exception as e:
        print(f"[GitHub] Error fetching issue rate for {owner}/{repo}: {e}")
        return None


def get_repository_health(
    owner: str,
