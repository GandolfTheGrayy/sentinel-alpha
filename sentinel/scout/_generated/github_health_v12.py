"""
GitHub Repository Health Signal Collector for Sentinel Scout.

This module measures developer momentum and project vitality by collecting
three key signals from GitHub repositories:
  1. Star count — community adoption indicator
  2. Commit velocity (commits/week) — development momentum
  3. Issue open rate — maintenance burden vs. closure rate

Used by Scout to enrich sentiment analysis for tech companies with active
open-source footprints. Data is cached locally to avoid API rate limits.
Integrates with the Sentinel pipeline to weight predictions by repo health.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Optional, TypedDict

import requests


class GitHubHealthSignal(TypedDict):
    """Structure for a single GitHub repository health snapshot."""
    repo_url: str
    stars: int
    commit_velocity: float  # commits per week
    issue_open_rate: float  # open_issues / (open_issues + closed_issues)
    fetched_at: str  # ISO 8601 timestamp
    error: Optional[str]  # None if successful, error message otherwise


def _init_cache_db(db_path: str = "sentinel_github_cache.db") -> sqlite3.Connection:
    """Initialize SQLite cache for GitHub API responses."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS github_health (
            repo_url TEXT PRIMARY KEY,
            stars INTEGER,
            commit_velocity REAL,
            issue_open_rate REAL,
            fetched_at TEXT,
            error TEXT
        )
    """)
    conn.commit()
    return conn


def _fetch_repo_metadata(owner: str, repo: str, token: Optional[str] = None) -> dict:
    """
    Fetch repository metadata from GitHub REST API.
    
    Args:
        owner: Repository owner (username or org)
        repo: Repository name
        token: Optional GitHub API token for higher rate limits
        
    Returns:
        Dictionary with keys: stargazers_count, open_issues_count
        
    Raises:
        requests.RequestException: If API call fails
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    return {
        "stargazers_count": data.get("stargazers_count", 0),
        "open_issues_count": data.get("open_issues_count", 0),
        "pushed_at": data.get("pushed_at"),  # Last push timestamp
    }


def _fetch_commit_history(
    owner: str,
    repo: str,
    token: Optional[str] = None,
    days_back: int = 7,
) -> int:
    """
    Fetch commit count over the past N days.
    
    Args:
        owner: Repository owner
        repo: Repository name
        token: Optional GitHub API token
        days_back: Number of days to look back (default 7 for weekly velocity)
        
    Returns:
        Number of commits in the time window
        
    Raises:
        requests.RequestException: If API call fails
    """
    since_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    params = {"since": since_date, "per_page": 100}
    
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    
    # GitHub returns paginated results; count total via Link header or direct count
    commits = resp.json()
    return len(commits)


def _fetch_issue_metrics(
    owner: str,
    repo: str,
    token: Optional[str] = None,
) -> tuple[int, int]:
    """
    Fetch open and closed issue counts.
    
    Args:
        owner: Repository owner
        repo: Repository name
        token: Optional GitHub API token
        
    Returns:
        Tuple of (open_count, closed_count)
        
    Raises:
        requests.RequestException: If API call fails
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    # Fetch open issues
    open_resp = requests.get(
        url,
        headers=headers,
        params={"state": "open", "per_page": 1},
        timeout=10,
    )
    open_resp.raise_for_status()
    # Total count is in the Link header or we can use search API
    # For simplicity, use the search API which returns a count
    
    # Use search API for accurate counts
    search_url = "https://api.github.com/search/issues"
    
    open_query = f"repo:{owner}/{repo} is:issue is:open"
    open_result = requests.get(
        search_url,
        headers=headers,
        params={"q": open_query, "per_page": 1},
        timeout=10,
    )
    open_result.raise_for_status()
    open_count = open_result.json().get("total_count", 0)
    
    closed_query = f"repo:{owner}/{repo} is:issue is:closed"
    closed_result = requests.get(
        search_url,
        headers=headers,
        params={"q": closed_query, "per_page": 1},
        timeout=10,
    )
    closed_result.raise_for_status()
    closed_count = closed_result.json().get("total_count", 0)
    
    return open_count, closed_count


def collect_github_health(
    repo_url: str,
    token: Optional[str] = None,
    cache_ttl_hours: int = 24,
    db_path: str = "sentinel_github_cache.db",
) -> GitHubHealthSignal:
    """
    Collect GitHub repository health signals (stars, commit velocity, issue rate).
    
    Args:
        repo_url: Full repository URL (e.g., 'https://github.com/owner/repo')
        token: Optional GitHub API token for higher rate limits
        cache_ttl_hours: Cache validity in hours (default 24)
        db_path: Path to SQLite cache database
        
    Returns:
        GitHubHealthSignal dict with stars, commit_velocity, issue_open_rate,
        fetched_at timestamp, and optional error message if collection failed
    """
    # Parse owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        return GitHubHealthSignal(
            repo_url=repo_url,
            stars=0,
            commit_velocity=0.0,
            issue_open_rate=0.0,
            fetched_at=datetime.utcnow().isoformat(),
            error=f"Invalid repository URL: {repo_url}",
        )
    
    owner = parts[-2]
    repo = parts[-1]
    
    # Check cache first
    conn = _init_cache_db(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT stars, commit_velocity, issue_open_rate, fetched_at, error FROM github_health WHERE repo_url = ?",
        (repo_url,),
    )
    row = cursor.fetchone()
    
    if row:
        stars, velocity, issue_rate, fetched_at_str, error = row
        fetched_at = datetime.fromisoformat(fetched_at_str)
        if datetime.utcnow() - fetched_at < timedelta(hours=cache_ttl_hours):
            conn.close()
            return GitHubHealthSignal(
