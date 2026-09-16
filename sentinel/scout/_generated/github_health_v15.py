"""
GitHub repository health signal collector for Sentinel Scout.

Measures developer sentiment and project momentum via GitHub metrics:
- Star count and growth rate
- Commit velocity (commits per week)
- Issue open rate and resolution time
- Activity heatmap (commits in last 7/30 days)

Used by Sentinel to detect early signals of technical health degradation
or acceleration in open-source dependencies and company-maintained projects.
Queries via GitHub REST API (unauthenticated or token-based).
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import requests


def fetch_repo_stats(owner: str, repo: str, github_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch core GitHub repository metrics: stars, watchers, forks, open issues.
    
    Args:
        owner: GitHub username or organization name.
        repo: Repository name.
        github_token: Optional GitHub PAT for higher rate limits (recommended).
    
    Returns:
        Dict with keys: stars, watchers, forks, open_issues, created_at, pushed_at, language.
    
    Raises:
        requests.RequestException: If API call fails.
        ValueError: If repo not found (404).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code == 404:
        raise ValueError(f"Repository {owner}/{repo} not found on GitHub.")
    resp.raise_for_status()
    
    data = resp.json()
    return {
        "stars": data.get("stargazers_count", 0),
        "watchers": data.get("watchers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "created_at": data.get("created_at"),
        "pushed_at": data.get("pushed_at"),
        "language": data.get("language"),
        "description": data.get("description", ""),
    }


def fetch_commit_velocity(
    owner: str, repo: str, days: int = 7, github_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measure commit frequency: commits per week in the last N days.
    
    Args:
        owner: GitHub username or organization name.
        repo: Repository name.
        days: Lookback window (default 7 days).
        github_token: Optional GitHub PAT.
    
    Returns:
        Dict with keys: commits_7d, commits_30d, commits_90d, velocity_per_week.
    """
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"since": since, "per_page": 100}
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    all_commits = []
    page = 1
    while len(all_commits) < 1000:  # Cap to avoid excessive pagination.
        params["page"] = page
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_commits.extend(batch)
        page += 1
        time.sleep(0.1)  # Gentle rate-limit respect.
    
    commits_7d = len(all_commits)
    velocity_per_week = commits_7d if days == 7 else (commits_7d / days) * 7
    
    # Fetch 30-day and 90-day counts separately.
    since_30 = (datetime.utcnow() - timedelta(days=30)).isoformat() + "Z"
    resp_30 = requests.get(
        url, headers=headers, params={"since": since_30, "per_page": 1}, timeout=10
    )
    resp_30.raise_for_status()
    commits_30d = resp_30.json()[0].get("commit", {}).get("author", {}).get("date") if resp_30.json() else 0
    
    # Count commits in 30-day window via pagination.
    commits_30d = 0
    for p in range(1, 4):
        resp = requests.get(
            url, headers=headers, params={"since": since_30, "page": p, "per_page": 100}, timeout=10
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        commits_30d += len(batch)
        time.sleep(0.05)
    
    # Count commits in 90-day window.
    since_90 = (datetime.utcnow() - timedelta(days=90)).isoformat() + "Z"
    commits_90d = 0
    for p in range(1, 5):
        resp = requests.get(
            url, headers=headers, params={"since": since_90, "page": p, "per_page": 100}, timeout=10
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        commits_90d += len(batch)
        time.sleep(0.05)
    
    return {
        "commits_7d": commits_7d,
        "commits_30d": commits_30d,
        "commits_90d": commits_90d,
        "velocity_per_week": round(velocity_per_week, 2),
    }


def fetch_issue_health(owner: str, repo: str, github_token: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze issue health: open rate, median resolution time, closure velocity.
    
    Args:
        owner: GitHub username or organization name.
        repo: Repository name.
        github_token: Optional GitHub PAT.
    
    Returns:
        Dict with keys: open_issues, closed_issues_7d, issue_open_rate, avg_age_days.
    """
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    # Fetch recent closed issues.
    since_7d = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
    url_closed = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "closed", "since": since_7d, "per_page": 100}
    
    closed_issues = []
    for p in range(1, 3):
        params["page"] = p
        resp = requests.get(url_closed, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        closed_issues.extend(batch)
        time.sleep(0.05)
    
    closed_7d = len(closed_issues)
    
    # Fetch all open issues.
    params_open = {"state": "open", "per_page": 1}
    resp_open = requests.get(url_closed, headers=headers, params=params_open, timeout=10)
    resp_open.raise_for_status()
    open_issues = resp_open.json()[0].get("number", 0) if resp_open.json() else 0
    
    # Count total open via pagination (capped).
    open_count = 0
    for p in range(1, 5):
        resp = requests.get(
            url_closed, headers=headers, params={"state": "open", "page": p, "per_page": 100}, timeout=10
        )
        resp.raise_for_status()
        batch = resp.json()
