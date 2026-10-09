"""
GitHub repository health signal collector for Sentinel Sentiment Engine.

This module measures developer ecosystem health for companies by analyzing
their public GitHub repositories. Signals include star velocity, commit
frequency (commits/week), and issue open rate. These metrics inform the
Scout pillar's assessment of technology company momentum and community
engagement — used downstream by Linguist for sentiment synthesis and Judge
for final prediction weighting.

Integrates with the Scout data ingestion pipeline to supplement price,
SEC filing, and news sentiment with developer-facing signals.
"""

import os
import re
import time
from typing import Optional, TypedDict
from datetime import datetime, timedelta

import requests


class GitHubRepoHealth(TypedDict):
    """Schema for a single repository health snapshot."""
    repo_url: str
    stars_current: int
    stars_30d_delta: int
    commits_per_week: float
    issues_open_count: int
    issues_open_rate: float
    prs_open_count: int
    last_commit_date: Optional[str]
    snapshot_timestamp: str
    api_calls_used: int


def _get_github_headers() -> dict[str, str]:
    """Build GitHub API request headers with optional token authentication."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _parse_repo_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from GitHub URL."""
    # Normalize: strip trailing .git and trailing slashes
    url = url.rstrip("/").rstrip(".git")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+)/?$", url)
    if not match:
        raise ValueError(f"Invalid GitHub URL: {url}")
    return match.group(1), match.group(2)


def fetch_repo_stars(owner: str, repo: str) -> tuple[int, int]:
    """Fetch current star count and 30-day delta for a repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = _get_github_headers()
    
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    current_stars = data.get("stargazers_count", 0)
    
    # Estimate 30-day delta by fetching star timeline (requires GraphQL or extra REST calls)
    # For MVP, we approximate via GitHub's star history endpoint (unofficial but stable)
    stars_30d_ago = _estimate_stars_30d_ago(owner, repo)
    delta = current_stars - stars_30d_ago
    
    return current_stars, delta


def _estimate_stars_30d_ago(owner: str, repo: str) -> int:
    """Estimate star count from 30 days ago using stargazers endpoint pagination."""
    # GitHub stargazers endpoint sorts by date; we paginate to ~30d boundary
    cutoff = datetime.utcnow() - timedelta(days=30)
    headers = _get_github_headers()
    headers["Accept"] = "application/vnd.github.star+json"
    
    url = f"https://api.github.com/repos/{owner}/{repo}/stargazers"
    params = {"per_page": 100, "page": 1}
    
    earliest_count = 0
    try:
        while True:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            stars = resp.json()
            
            if not stars:
                break
            
            for star_entry in stars:
                starred_at = datetime.fromisoformat(
                    star_entry.get("starred_at", "").replace("Z", "+00:00")
                )
                if starred_at < cutoff:
                    return earliest_count
                earliest_count += 1
            
            if len(stars) < 100:
                break
            params["page"] += 1
    except Exception:
        # Fallback: assume 0 if endpoint fails
        pass
    
    return 0


def fetch_commit_velocity(owner: str, repo: str) -> float:
    """Fetch average commits per week over the last 52 weeks."""
    url = f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity"
    headers = _get_github_headers()
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            return 0.0
        
        # data is a list of 52 weekly buckets; sum total commits and divide by weeks
        total_commits = sum(week.get("total", 0) for week in data)
        weeks_with_data = len([w for w in data if w.get("total", 0) > 0])
        
        return total_commits / 52.0 if data else 0.0
    except Exception:
        return 0.0


def fetch_issue_metrics(owner: str, repo: str) -> tuple[int, float]:
    """Fetch count and open rate of issues (excluding PRs)."""
    headers = _get_github_headers()
    
    # Fetch open issues
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    params = {"state": "open", "per_page": 1}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    
    # GitHub returns total count in Link header if paginated
    link_header = resp.headers.get("Link", "")
    open_count = 0
    if "last" in link_header:
        match = re.search(r"page=(\d+)>; rel=\"last\"", link_header)
        if match:
            open_count = int(match.group(1))
    elif resp.json():
        open_count = 1
    
    # Fetch closed issues
    params["state"] = "closed"
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    
    link_header = resp.headers.get("Link", "")
    closed_count = 0
    if "last" in link_header:
        match = re.search(r"page=(\d+)>; rel=\"last\"", link_header)
        if match:
            closed_count = int(match.group(1))
    elif resp.json():
        closed_count = 1
    
    total_issues = open_count + closed_count
    open_rate = open_count / total_issues if total_issues > 0 else 0.0
    
    return open_count, open_rate


def fetch_last_commit_date(owner: str, repo: str) -> Optional[str]:
    """Fetch the most recent commit date on the default branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = _get_github_headers()
    params = {"per_page": 1}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if data and len(data) > 0:
            return data[0].get("commit", {}).get("committer", {}).get("date")
    except Exception:
        pass
    
    return None


def fetch_repo_health(repo_url: str) -> GitHubRepoHealth:
    """
    Fetch comprehensive health metrics for a GitHub repository.
    
    Args:
        repo_url: Full GitHub repository URL (e.g., https://github.com/owner/repo)
    
    Returns:
        GitHubRepoHealth dict with stars, commit velocity, issue metrics, and timestamp.
    
    Raises:
        ValueError: If URL is malformed.
        requests.HTTP
