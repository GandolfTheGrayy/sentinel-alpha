"""
GitHub Repository Health Signal Collector for Sentinel Scout.

This module collects quantitative health signals from public GitHub repositories:
- Star count (absolute popularity signal)
- Commit velocity (commits/week, developer activity intensity)
- Issue open rate (maintenance burden, code quality proxy)

These signals feed into the Scout agent's multi-source sentiment analysis,
helping predict tech-sector stock movements via developer ecosystem health.

Uses the GitHub REST API (unauthenticated, rate-limited to 60 req/hour).
For production, supply a GITHUB_TOKEN env var for 5000 req/hour.
"""

import os
import time
from datetime import datetime, timedelta
from typing import TypedDict, Optional
import requests


class GitHubHealthSignal(TypedDict):
    """GitHub health metrics for a repository."""
    repo_full_name: str
    timestamp: str
    stars: int
    commit_velocity_per_week: float
    issue_open_rate: float
    total_commits_measured: int
    measurement_window_days: int
    api_calls_used: int


def _get_github_headers() -> dict:
    """Build GitHub API request headers with optional auth token."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _fetch_repo_metadata(owner: str, repo: str) -> dict:
    """Fetch repository metadata including star count and issue counts."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = requests.get(url, headers=_get_github_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "stars": data.get("stargazers_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "total_issues": data.get("open_issues_count", 0),  # API limitation
    }


def _fetch_commits_last_n_days(
    owner: str, repo: str, days: int = 30
) -> tuple[int, int]:
    """
    Fetch commit count in last N days and total commits.
    
    Returns tuple of (commits_in_window, total_commits_in_repo).
    """
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params = {"since": since, "per_page": 1}
    
    resp = requests.get(
        url, headers=_get_github_headers(), params=params, timeout=10
    )
    resp.raise_for_status()
    
    commits_in_window = 0
    link_header = resp.headers.get("Link", "")
    
    if 'rel="last"' in link_header:
        last_url = None
        for link_part in link_header.split(","):
            if 'rel="last"' in link_part:
                last_url = link_part.split(";")[0].strip("<>")
                break
        if last_url:
            resp_last = requests.get(
                last_url, headers=_get_github_headers(), timeout=10
            )
            resp_last.raise_for_status()
            commits_in_window = resp_last.json()[0].get("commit", {}).get(
                "message", ""
            )
            if "Link" in resp_last.headers:
                try:
                    page_match = resp_last.url.split("page=")[-1]
                    commits_in_window = int(page_match.split("&")[0])
                except (IndexError, ValueError):
                    commits_in_window = len(resp_last.json())
            else:
                commits_in_window = len(resp_last.json())
    else:
        commits_in_window = len(resp.json())
    
    url_all = f"https://api.github.com/repos/{owner}/{repo}/commits"
    params_all = {"per_page": 1}
    resp_all = requests.get(
        url_all, headers=_get_github_headers(), params=params_all, timeout=10
    )
    resp_all.raise_for_status()
    total_commits = 0
    if "Link" in resp_all.headers:
        try:
            last_url = None
            for link_part in resp_all.headers["Link"].split(","):
                if 'rel="last"' in link_part:
                    last_url = link_part.split(";")[0].strip("<>")
                    break
            if last_url:
                page_match = last_url.split("page=")[-1]
                total_commits = int(page_match.split("&")[0])
        except (IndexError, ValueError):
            total_commits = 1
    else:
        total_commits = 1
    
    return commits_in_window, total_commits


def _fetch_issue_stats(owner: str, repo: str) -> tuple[int, int]:
    """
    Fetch open and closed issue counts.
    
    Returns tuple of (open_issues, total_issues_closed).
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    
    open_count = 0
    params = {"state": "open", "per_page": 1}
    resp = requests.get(
        url, headers=_get_github_headers(), params=params, timeout=10
    )
    resp.raise_for_status()
    
    if "Link" in resp.headers:
        try:
            for link_part in resp.headers["Link"].split(","):
                if 'rel="last"' in link_part:
                    last_url = link_part.split(";")[0].strip("<>")
                    page_match = last_url.split("page=")[-1]
                    open_count = int(page_match.split("&")[0])
                    break
        except (IndexError, ValueError):
            open_count = len(resp.json())
    else:
        open_count = len(resp.json())
    
    closed_count = 0
    params_closed = {"state": "closed", "per_page": 1}
    resp_closed = requests.get(
        url, headers=_get_github_headers(), params=params_closed, timeout=10
    )
    resp_closed.raise_for_status()
    
    if "Link" in resp_closed.headers:
        try:
            for link_part in resp_closed.headers["Link"].split(","):
                if 'rel="last"' in link_part:
                    last_url = link_part.split(";")[0].strip("<>")
                    page_match = last_url.split("page=")[-1]
                    closed_count = int(page_match.split("&")[0])
                    break
        except (IndexError, ValueError):
            closed_count = len(resp_closed.json())
    else:
        closed_count = len(resp_closed.json())
    
    return open_count, closed_count


def collect_github_health(
    repo_url: str, measurement_window_days: int = 30
) -> GitHubHealthSignal:
    """
    Collect health signals for a GitHub repository.
    
    Args:
        repo_url: Full repository URL (e.g., "https://github.com/owner/repo")
                  or "owner/repo" format.
        measurement_window_days: Rolling window for commit velocity (default 30).
    
    Returns:
        GitHubHealthSignal dict with stars, commit velocity, issue open rate.
    
    Raises:
        ValueError: If repo_url format is invalid.
        requests.HTTPError: If GitHub API returns error.
    """
    repo_url = repo_url.rstrip("/")
    if "github.com" in repo_url:
        parts = repo_url.split("/")
        owner, repo = parts[-2], parts[-1]
    elif "/"
