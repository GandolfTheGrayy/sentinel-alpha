"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module measures developer activity and repository vitality as a proxy for
project momentum and institutional confidence. Metrics include:
  - Star count (absolute + velocity)
  - Commit velocity (commits/week over trailing window)
  - Issue open rate (open issues / total issues, trending)

These signals feed into the Judge's cross-asset sentiment weighting and help
detect under-the-radar technical momentum for companies with public GitHub repos
(e.g., crypto projects, open-source infrastructure firms, AI startups).

Uses the GitHub REST API (unauthenticated or token-authenticated for higher
rate limits). Caches results in ChromaDB for historical comparison and anomaly
detection by the Historian pillar.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import requests


def get_github_repo_stats(owner: str, repo: str) -> Dict[str, Any]:
    """
    Fetch star count, commit velocity, and issue metrics for a GitHub repo.
    
    Returns dict with keys: stars, commits_per_week, issue_open_rate, fetched_at.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        # Fetch repo overview (stars, etc.)
        repo_resp = requests.get(base_url, headers=headers, timeout=10)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        stars = repo_data.get("stargazers_count", 0)

        # Fetch commits from past 7 days to compute velocity.
        since = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
        commits_url = f"{base_url}/commits?since={since}&per_page=100"
        commits_resp = requests.get(commits_url, headers=headers, timeout=10)
        commits_resp.raise_for_status()
        commit_count = len(commits_resp.json())
        commits_per_week = commit_count

        # Fetch issues to compute open rate.
        issues_open_url = f"{base_url}/issues?state=open&per_page=1"
        issues_open_resp = requests.get(issues_open_url, headers=headers, timeout=10)
        issues_open_resp.raise_for_status()
        open_count = issues_open_resp.json()[0].get("number", 0) if issues_open_resp.json() else 0

        issues_closed_url = f"{base_url}/issues?state=closed&per_page=1"
        issues_closed_resp = requests.get(issues_closed_url, headers=headers, timeout=10)
        issues_closed_resp.raise_for_status()
        closed_count = issues_closed_resp.json()[0].get("number", 0) if issues_closed_resp.json() else 0

        total_issues = open_count + closed_count
        issue_open_rate = (open_count / total_issues) if total_issues > 0 else 0.0

        return {
            "owner": owner,
            "repo": repo,
            "stars": stars,
            "commits_per_week": commits_per_week,
            "issue_open_rate": issue_open_rate,
            "fetched_at": datetime.utcnow().isoformat(),
        }
    except requests.RequestException as e:
        return {
            "owner": owner,
            "repo": repo,
            "stars": None,
            "commits_per_week": None,
            "issue_open_rate": None,
            "error": str(e),
            "fetched_at": datetime.utcnow().isoformat(),
        }


def batch_github_repos(repos: list[tuple[str, str]]) -> list[Dict[str, Any]]:
    """
    Fetch health signals for multiple repos, respecting GitHub rate limits.
    
    repos: list of (owner, repo_name) tuples.
    Returns list of stats dicts, one per repo.
    """
    results = []
    for owner, repo in repos:
        stats = get_github_repo_stats(owner, repo)
        results.append(stats)
        time.sleep(0.5)
    return results


if __name__ == "__main__":
    test_repos = [
        ("openai", "gpt-4"),
        ("anthropics", "anthropic-sdk-python"),
        ("langchain-ai", "langchain"),
    ]
    stats = batch_github_repos(test_repos)
    for stat in stats:
        print(stat)
