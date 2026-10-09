"""
Sentinel Scout: GitHub Repository Health Signal Collector

Measures developer activity and project momentum via GitHub API:
  - Star count (community adoption signal)
  - Commit velocity (commits/week, recent 52-week window)
  - Issue open rate (% of issues currently open vs. closed)

Used by Linguist to weight sentiment signals — e.g., declining commits
+ rising open issues may indicate abandonment despite positive headlines.
Integrated into RAG context for fintech/infrastructure plays.

Requires GITHUB_TOKEN env var (fine-grained or classic PAT).
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import TypedDict, Optional
import requests


class GitHubRepoHealth(TypedDict):
    """Repository health metrics snapshot."""
    repo_name: str
    owner: str
    url: str
    stars: int
    commit_velocity_per_week: float
    issue_open_rate: float
    last_commit_date: str
    total_commits_52w: int
    open_issues: int
    closed_issues: int
    fetched_at: str


def _get_github_token() -> str:
    """Retrieve GitHub API token from environment."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN env var not set. "
            "Create a fine-grained PAT at https://github.com/settings/tokens"
        )
    return token


def _make_github_request(endpoint: str, params: Optional[dict] = None) -> dict:
    """Execute authenticated GitHub API request and return JSON."""
    token = _get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com{endpoint}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_repo_health(owner: str, repo: str) -> GitHubRepoHealth:
    """
    Fetch stars, commit velocity, and issue metrics for a GitHub repo.
    """
    # Repo metadata: stars, open/closed issues
    repo_data = _make_github_request(f"/repos/{owner}/{repo}")
    stars = repo_data["stargazers_count"]
    open_issues = repo_data["open_issues_count"]

    # Closed issues: GitHub search API
    closed_query = f"repo:{owner}/{repo} is:issue is:closed"
    closed_resp = _make_github_request(
        "/search/issues",
        {"q": closed_query, "per_page": 1}
    )
    closed_issues = closed_resp["total_count"]

    # Issue open rate
    total_issues = open_issues + closed_issues
    issue_open_rate = (
        open_issues / total_issues if total_issues > 0 else 0.0
    )

    # Commit velocity: last 52 weeks
    since_date = (datetime.utcnow() - timedelta(weeks=52)).isoformat() + "Z"
    commits_resp = _make_github_request(
        f"/repos/{owner}/{repo}/commits",
        {"since": since_date, "per_page": 1}
    )
    total_commits_52w = commits_resp.get("total_count", 0) if isinstance(
        commits_resp, dict
    ) else len(commits_resp)

    commit_velocity = total_commits_52w / 52.0

    # Last commit date
    latest_commits = _make_github_request(
        f"/repos/{owner}/{repo}/commits",
        {"per_page": 1}
    )
    last_commit_date = (
        latest_commits[0]["commit"]["committer"]["date"]
        if latest_commits
        else "unknown"
    )

    return GitHubRepoHealth(
        repo_name=repo,
        owner=owner,
        url=repo_data["html_url"],
        stars=stars,
        commit_velocity_per_week=commit_velocity,
        issue_open_rate=issue_open_rate,
        last_commit_date=last_commit_date,
        total_commits_52w=total_commits_52w,
        open_issues=open_issues,
        closed_issues=closed_issues,
        fetched_at=datetime.utcnow().isoformat(),
    )


def store_repo_health(db_path: str, health: GitHubRepoHealth) -> None:
    """Write GitHub health snapshot to SQLite for historical tracking."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create table if missing
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS github_health (
            id INTEGER PRIMARY KEY,
            repo_name TEXT NOT NULL,
            owner TEXT NOT NULL,
            url TEXT,
            stars INTEGER,
            commit_velocity_per_week REAL,
            issue_open_rate REAL,
            last_commit_date TEXT,
            total_commits_52w INTEGER,
            open_issues INTEGER,
            closed_issues INTEGER,
            fetched_at TEXT NOT NULL,
            UNIQUE(owner, repo_name, fetched_at)
        )
        """
    )

    cur.execute(
        """
        INSERT OR REPLACE INTO github_health
        (repo_name, owner, url, stars, commit_velocity_per_week,
         issue_open_rate, last_commit_date, total_commits_52w,
         open_issues, closed_issues, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            health["repo_name"],
            health["owner"],
            health["url"],
            health["stars"],
            health["commit_velocity_per_week"],
            health["issue_open_rate"],
            health["last_commit_date"],
            health["total_commits_52w"],
            health["open_issues"],
            health["closed_issues"],
            health["fetched_at"],
        ),
    )
    conn.commit()
    conn.close()


def health_signal_summary(health: GitHubRepoHealth) -> str:
    """
    Return human-readable one-liner summary of repo health for RAG context.
    """
    emoji_velocity = (
        "🚀" if health["commit_velocity_per_week"] > 5
        else "📈" if health["commit_velocity_per_week"] > 1
        else "⚠️"
    )
    emoji_issues = (
        "✅" if health["issue_open_rate"] < 0.3
        else "⚠️" if health["issue_open_rate"] < 0.6
        else "🔴"
    )

    return (
        f"{health['repo_name']}: {health['stars']} ⭐ | "
        f"{emoji_velocity} {health['commit_velocity_per_week']:.1f} commits/wk | "
        f"{emoji_issues} {health['issue_open_rate']:.1%} issues open"
    )


if __name__ == "__main__":
    # Demo: fetch health for a popular open-source project
    try:
        health = fetch_repo_health("torvalds", "linux")
        print(json.dumps(health, indent=2))
        print("\nSummary:", health_signal_summary(health))
    except Exception as e:
        print(f"Error (expected if GITHUB_TOKEN not set): {e}")
