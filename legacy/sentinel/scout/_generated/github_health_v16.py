"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests developer health signals from GitHub repositories,
measuring code velocity (commits/week), community engagement (stars trend),
and issue triage health (open rate). Used by Scout to detect early warning
signs of project momentum shifts that correlate with stock sentiment moves.

Collects:
  - Star count & growth rate
  - Commit velocity (commits in last 7/30 days)
  - Issue open/closed ratio and median resolution time
  - Release frequency

Requires GITHUB_TOKEN env var (GitHub PAT with public_repo scope minimum).
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, TypedDict
import requests


class GitHubRepoHealth(TypedDict):
    """Type definition for GitHub repository health metrics."""
    repo_url: str
    stars_current: int
    stars_30d_ago: Optional[int]
    star_growth_rate: Optional[float]
    commits_7d: int
    commits_30d: int
    commit_velocity_week: float
    issues_open: int
    issues_total_30d: int
    issue_open_rate: Optional[float]
    avg_issue_resolution_days: Optional[float]
    releases_30d: int
    last_commit_date: Optional[str]
    timestamp: str


def _get_github_headers() -> dict[str, str]:
    """Return authorization headers for GitHub API with token from env."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN env var required for GitHub API access"
        )
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Sentinel-Sentiment-Engine"
    }


def _parse_repo_slug(repo_input: str) -> tuple[str, str]:
    """Parse owner/repo from URL or slug; return (owner, repo)."""
    if "github.com" in repo_input:
        parts = repo_input.rstrip("/").split("/")
        return parts[-2], parts[-1]
    elif "/" in repo_input:
        owner, repo = repo_input.split("/", 1)
        return owner, repo
    else:
        raise ValueError(
            f"Invalid repo format: {repo_input}. Use 'owner/repo' or GitHub URL"
        )


def fetch_github_repo_health(repo: str) -> GitHubRepoHealth:
    """
    Fetch comprehensive GitHub repository health metrics.
    
    Args:
        repo: Repository identifier as "owner/repo" or full GitHub URL.
    
    Returns:
        GitHubRepoHealth dict with stars, commits, issues, releases.
    
    Raises:
        requests.HTTPError: If API calls fail (rate limit, auth, 404).
        ValueError: If repo format invalid or GitHub token missing.
    """
    owner, repo_name = _parse_repo_slug(repo)
    headers = _get_github_headers()
    repo_url = f"https://api.github.com/repos/{owner}/{repo_name}"
    
    # Fetch repo metadata (stars, last push, etc.)
    repo_resp = requests.get(repo_url, headers=headers, timeout=10)
    repo_resp.raise_for_status()
    repo_data = repo_resp.json()
    
    stars_current = repo_data.get("stargazers_count", 0)
    last_push = repo_data.get("pushed_at")
    
    # Commit velocity: last 7 and 30 days.
    # GitHub's commits endpoint requires iteration; use search API for efficiency.
    now = datetime.utcnow()
    seven_days_ago = (now - timedelta(days=7)).isoformat() + "Z"
    thirty_days_ago = (now - timedelta(days=30)).isoformat() + "Z"
    
    commits_7d_query = f"repo:{owner}/{repo_name} committer-date:{seven_days_ago}..{now.isoformat()}Z"
    commits_7d_resp = requests.get(
        "https://api.github.com/search/commits",
        headers=headers,
        params={"q": commits_7d_query},
        timeout=10
    )
    commits_7d_resp.raise_for_status()
    commits_7d = commits_7d_resp.json().get("total_count", 0)
    
    commits_30d_query = f"repo:{owner}/{repo_name} committer-date:{thirty_days_ago}..{now.isoformat()}Z"
    commits_30d_resp = requests.get(
        "https://api.github.com/search/commits",
        headers=headers,
        params={"q": commits_30d_query},
        timeout=10
    )
    commits_30d_resp.raise_for_status()
    commits_30d = commits_30d_resp.json().get("total_count", 0)
    
    commit_velocity_week = commits_7d / 1.0  # Already weekly
    
    # Star growth (30 days): use starred_at via timeline. Approximate via API limit.
    # For production, consider GitHub's GraphQL API or star history service.
    stars_30d_ago: Optional[int] = None
    star_growth_rate: Optional[float] = None
    
    try:
        stars_resp = requests.get(
            f"{repo_url}/stargazers?per_page=1&sort=created",
            headers={**headers, "Accept": "application/vnd.github.v3.star+json"},
            timeout=10
        )
        stars_resp.raise_for_status()
        # Paginate to last page to infer historical count (crude approximation).
        if "Link" in stars_resp.headers:
            link_header = stars_resp.headers["Link"]
            if "last" in link_header:
                last_url = [l.split(";")[0].strip("<>") for l in link_header.split(",") if "last" in l]
                if last_url:
                    last_resp = requests.get(last_url[0], headers=headers, timeout=10)
                    last_resp.raise_for_status()
                    # Rough: total_count ≈ page * per_page. Refined by API response headers.
                    if "X-Total-Count" in last_resp.headers:
                        stars_current = int(last_resp.headers["X-Total-Count"])
    except (requests.RequestException, IndexError, ValueError, KeyError):
        pass  # Fallback: stars_30d_ago remains None.
    
    if stars_30d_ago is not None and stars_30d_ago > 0:
        star_growth_rate = (stars_current - stars_30d_ago) / stars_30d_ago
    
    # Issues: open rate and resolution time.
    issues_resp = requests.get(
        f"{repo_url}/issues",
        headers=headers,
        params={"state": "all", "per_page": 100},
        timeout=10
    )
    issues_resp.raise_for_status()
    issues_data = issues_resp.json()
    
    issues_open = 0
    issues_total_30d = 0
    total_resolution_days = 0
    closed_in_30d = 0
    
    for issue in issues_data:
        created_at = datetime.fromisoformat(issue["created_at"].replace("Z", "+00:00"))
        if (now - created_at).days <= 30:
            issues_total_30d += 1
        
        if issue["state"] == "open":
            issues_open += 1
        elif issue["state"] == "closed":
            closed_at = datetime.fromisoformat(issue["closed_at"].replace("Z", "+00:00"))
            if (now - closed_at).days <= 30:
                closed_in_30d += 1
                resolution_days = (closed_at - created_at).days
                total_resolution_days += resolution_days
    
    issue
