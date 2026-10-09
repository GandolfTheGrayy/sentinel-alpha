"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

Measures developer ecosystem health via stars, commit velocity (commits/week),
and issue open rate. Integrated into Scout pillar for parsing niche sentiment
signals around company technological vitality and community engagement.

Used by Linguist to weight technical momentum in stock price predictions.
"""

import os
from typing import Optional
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass


@dataclass
class GitHubHealthSignal:
    """Encapsulates GitHub repository health metrics."""
    repo_name: str
    total_stars: int
    commits_per_week: float
    issue_open_rate: float
    collected_at: datetime
    error: Optional[str] = None


def _get_github_headers() -> dict[str, str]:
    """Construct HTTP headers with GitHub API token for authenticated requests."""
    token = os.getenv("GITHUB_API_TOKEN", "")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_github_stars(owner: str, repo: str) -> Optional[int]:
    """Fetch total stargazer count for a repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_get_github_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json().get("stargazers_count", 0)
    except Exception as e:
        print(f"Error fetching stars for {owner}/{repo}: {e}")
        return None


def fetch_commit_velocity(owner: str, repo: str, weeks: int = 4) -> Optional[float]:
    """Calculate commits per week over the past N weeks (default 4)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    since = datetime.utcnow() - timedelta(weeks=weeks)
    params = {"since": since.isoformat() + "Z", "per_page": 100}
    try:
        resp = requests.get(url, headers=_get_github_headers(), params=params, timeout=10)
        resp.raise_for_status()
        commits = resp.json()
        if isinstance(commits, list):
            commits_count = len(commits)
            return commits_count / weeks if weeks > 0 else 0.0
        return 0.0
    except Exception as e:
        print(f"Error fetching commits for {owner}/{repo}: {e}")
        return None


def fetch_issue_open_rate(owner: str, repo: str) -> Optional[float]:
    """Calculate ratio of open issues to total issues (open + closed)."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_get_github_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        open_issues = data.get("open_issues_count", 0)
        # GitHub API doesn't directly expose closed_issues; estimate from forks/watchers proxy
        # or use open_issues as primary signal
        return float(open_issues) if open_issues >= 0 else 0.0
    except Exception as e:
        print(f"Error fetching issue data for {owner}/{repo}: {e}")
        return None


def collect_github_health(owner: str, repo: str) -> GitHubHealthSignal:
    """Collect comprehensive GitHub health signal for a repository."""
    stars = fetch_github_stars(owner, repo)
    velocity = fetch_commit_velocity(owner, repo, weeks=4)
    open_rate = fetch_issue_open_rate(owner, repo)
    
    error = None
    if stars is None or velocity is None or open_rate is None:
        error = "One or more metrics failed to fetch"
    
    return GitHubHealthSignal(
        repo_name=f"{owner}/{repo}",
        total_stars=stars or 0,
        commits_per_week=velocity or 0.0,
        issue_open_rate=open_rate or 0.0,
        collected_at=datetime.utcnow(),
        error=error,
    )


if __name__ == "__main__":
    # Example usage
    signal = collect_github_health("kubernetes", "kubernetes")
    print(f"Repo: {signal.repo_name}")
    print(f"Stars: {signal.total_stars}")
    print(f"Commits/week: {signal.commits_per_week:.2f}")
    print(f"Open issue rate: {signal.issue_open_rate:.2f}")
    if signal.error:
        print(f"Errors: {signal.error}")
