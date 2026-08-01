"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

Measures developer activity and project vitality through stars, commit velocity,
and issue open rate. Integrated into Scout pillar for cross-referencing tech
company sentiment with underlying developer engagement signals.

Uses GitHub REST API (unauthenticated or token-based) to fetch repo metrics.
Returns structured health scores for RAG ingestion and Judge reasoning.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional, TypedDict

import requests


class GitHubRepoHealth(TypedDict):
    """Structured health metrics for a GitHub repository."""

    owner: str
    repo: str
    stars: int
    commit_velocity: float
    issue_open_rate: float
    fetched_at: str
    raw_commits_7d: int
    raw_issues_open: int
    raw_issues_closed: int


def _get_github_token() -> Optional[str]:
    """Retrieve GitHub API token from environment, or None for unauthenticated."""
    return os.getenv("GITHUB_TOKEN")


def _github_api_request(endpoint: str, params: Optional[dict] = None) -> dict:
    """
    Make authenticated or unauthenticated request to GitHub REST API v3.

    Args:
        endpoint: URL path relative to https://api.github.com (e.g. "/repos/owner/repo")
        params: Query parameters dict

    Returns:
        Parsed JSON response as dict

    Raises:
        requests.HTTPError: If response status >= 400
    """
    base_url = "https://api.github.com"
    headers = {"Accept": "application/vnd.github.v3+json"}

    token = _get_github_token()
    if token:
        headers["Authorization"] = f"token {token}"

    url = f"{base_url}{endpoint}"
    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_repo_stars(owner: str, repo: str) -> int:
    """Fetch current star count for a repository."""
    data = _github_api_request(f"/repos/{owner}/{repo}")
    return data.get("stargazers_count", 0)


def fetch_commit_velocity(owner: str, repo: str, days: int = 7) -> float:
    """
    Fetch average commits per week over past N days.

    Args:
        owner: Repository owner
        repo: Repository name
        days: Lookback window in days (default 7)

    Returns:
        Commits per week (float)
    """
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    data = _github_api_request(
        f"/repos/{owner}/{repo}/commits",
        params={"since": since, "per_page": 100},
    )

    # data is a list of commits
    commit_count = len(data) if isinstance(data, list) else 0
    weeks = max(days / 7.0, 1.0)
    return commit_count / weeks


def fetch_issue_open_rate(owner: str, repo: str) -> float:
    """
    Fetch ratio of open issues to total (open + closed) issues.

    Args:
        owner: Repository owner
        repo: Repository name

    Returns:
        Open rate as float in [0, 1]
    """
    # Fetch open issues
    open_data = _github_api_request(
        f"/repos/{owner}/{repo}/issues",
        params={"state": "open", "per_page": 1},
    )
    open_count = open_data[0]["number"] if open_data else 0

    # Fetch closed issues (via search for more reliable count)
    search_data = _github_api_request(
        "/search/issues",
        params={"q": f"repo:{owner}/{repo} is:issue is:closed", "per_page": 1},
    )
    closed_count = search_data.get("total_count", 0)

    total = open_count + closed_count
    if total == 0:
        return 0.0

    return open_count / float(total)


def collect_repo_health(owner: str, repo: str) -> GitHubRepoHealth:
    """
    Collect all health metrics for a repository in a single call.

    Args:
        owner: Repository owner (e.g. "python")
        repo: Repository name (e.g. "cpython")

    Returns:
        GitHubRepoHealth dict with all metrics and timestamp

    Raises:
        requests.HTTPError: If any GitHub API call fails
    """
    stars = fetch_repo_stars(owner, repo)
    velocity = fetch_commit_velocity(owner, repo, days=7)

    # Fetch issue counts directly from repo endpoint for efficiency
    repo_data = _github_api_request(f"/repos/{owner}/{repo}")
    open_issues = repo_data.get("open_issues_count", 0)

    # Estimate closed count (GitHub doesn't expose directly; use search)
    search_data = _github_api_request(
        "/search/issues",
        params={"q": f"repo:{owner}/{repo} is:issue is:closed", "per_page": 1},
    )
    closed_issues = search_data.get("total_count", 0)

    total_issues = open_issues + closed_issues
    issue_rate = (
        open_issues / float(total_issues) if total_issues > 0 else 0.0
    )

    return GitHubRepoHealth(
        owner=owner,
        repo=repo,
        stars=stars,
        commit_velocity=velocity,
        issue_open_rate=issue_rate,
        fetched_at=datetime.utcnow().isoformat(),
        raw_commits_7d=int(velocity * 7),
        raw_issues_open=open_issues,
        raw_issues_closed=closed_issues,
    )


def parse_github_url(url: str) -> Optional[tuple[str, str]]:
    """
    Parse owner and repo from GitHub URL or shorthand.

    Args:
        url: GitHub URL (https://github.com/owner/repo) or shorthand (owner/repo)

    Returns:
        Tuple of (owner, repo), or None if parsing fails
    """
    # Try shorthand first
    match = re.match(r"^([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)$", url)
    if match:
        return match.group(1), match.group(2)

    # Try full URL
    match = re.search(r"github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)", url)
    if match:
        return match.group(1), match.group(2)

    return None


if __name__ == "__main__":
    import json

    # Example: fetch health for a popular repo
    result = collect_repo_health("torvalds", "linux")
    print(json.dumps(result, indent=2))
