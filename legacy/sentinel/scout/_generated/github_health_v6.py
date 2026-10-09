"""
GitHub repository health signal collector for Sentinel Sentiment Engine.

This module ingests GitHub metrics (stars, commit velocity, issue open rate)
for technology companies and projects, enabling Sentinel to detect developer
sentiment shifts and project momentum changes via the Scout pillar.

Used by: sentinel/pipeline.py → Scout → Linguist (sentiment weighting)
"""

import os
import time
from typing import Optional
import requests


def fetch_github_repo_health(owner: str, repo: str, github_token: Optional[str] = None) -> dict:
    """
    Fetch health metrics for a GitHub repository: stars, commit velocity, issue open rate.
    
    Args:
        owner: GitHub repository owner/org name.
        repo: GitHub repository name.
        github_token: Optional GitHub API token (reads from GITHUB_TOKEN env if not provided).
    
    Returns:
        dict with keys: stars (int), commits_per_week (float), issue_open_rate (float),
        error (str, if failed), timestamp (int).
    """
    token = github_token or os.getenv("GITHUB_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    base_url = "https://api.github.com"
    result = {
        "owner": owner,
        "repo": repo,
        "timestamp": int(time.time()),
        "stars": None,
        "commits_per_week": None,
        "issue_open_rate": None,
        "error": None,
    }
    
    try:
        # Fetch repo metadata (stars, default branch).
        repo_url = f"{base_url}/repos/{owner}/{repo}"
        repo_resp = requests.get(repo_url, headers=headers, timeout=10)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        result["stars"] = repo_data.get("stargazers_count", 0)
        default_branch = repo_data.get("default_branch", "main")
    except requests.RequestException as e:
        result["error"] = f"repo fetch failed: {str(e)}"
        return result
    
    try:
        # Fetch commit count for the last week.
        commits_url = f"{base_url}/repos/{owner}/{repo}/commits"
        now = int(time.time())
        one_week_ago = now - (7 * 24 * 60 * 60)
        since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(one_week_ago))
        
        commits_resp = requests.get(
            commits_url,
            params={"since": since_iso, "per_page": 100},
            headers=headers,
            timeout=10,
        )
        commits_resp.raise_for_status()
        commits_data = commits_resp.json()
        
        # GitHub returns up to 100 per page; for velocity estimate, use the count.
        commit_count = len(commits_data) if isinstance(commits_data, list) else 0
        result["commits_per_week"] = float(commit_count)
    except requests.RequestException as e:
        result["error"] = f"commits fetch failed: {str(e)}"
        return result
    
    try:
        # Fetch open and closed issues (last 30 days for rate calculation).
        issues_url = f"{base_url}/repos/{owner}/{repo}/issues"
        thirty_days_ago = now - (30 * 24 * 60 * 60)
        since_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(thirty_days_ago))
        
        open_resp = requests.get(
            issues_url,
            params={"state": "open", "since": since_iso, "per_page": 1},
            headers=headers,
            timeout=10,
        )
        open_resp.raise_for_status()
        open_link = open_resp.headers.get("Link", "")
        open_count = _parse_github_link_total(open_link) or len(open_resp.json())
        
        closed_resp = requests.get(
            issues_url,
            params={"state": "closed", "since": since_iso, "per_page": 1},
            headers=headers,
            timeout=10,
        )
        closed_resp.raise_for_status()
        closed_link = closed_resp.headers.get("Link", "")
        closed_count = _parse_github_link_total(closed_link) or len(closed_resp.json())
        
        total_issues = open_count + closed_count
        result["issue_open_rate"] = (
            (open_count / total_issues) if total_issues > 0 else 0.0
        )
    except requests.RequestException as e:
        result["error"] = f"issues fetch failed: {str(e)}"
        return result
    
    return result


def _parse_github_link_total(link_header: str) -> Optional[int]:
    """
    Extract the total count from GitHub's Link header pagination.
    
    Args:
        link_header: GitHub's Link header string.
    
    Returns:
        Total count if last page is present, None otherwise.
    """
    if not link_header:
        return None
    
    parts = link_header.split(",")
    for part in parts:
        if 'rel="last"' in part:
            # Extract page number from URL.
            import re
            match = re.search(r"[?&]page=(\d+)", part)
            if match:
                page = int(match.group(1))
                return page * 100  # Conservative estimate (100 per page).
    return None


def batch_fetch_github_health(repos: list[tuple[str, str]], github_token: Optional[str] = None) -> list[dict]:
    """
    Fetch health metrics for multiple GitHub repositories with rate-limit awareness.
    
    Args:
        repos: List of (owner, repo) tuples.
        github_token: Optional GitHub API token.
    
    Returns:
        List of health dicts, one per repo.
    """
    results = []
    for owner, repo in repos:
        result = fetch_github_repo_health(owner, repo, github_token)
        results.append(result)
        # Respect GitHub API rate limits (60 req/hr unauthenticated, 5000/hr authenticated).
        time.sleep(0.5)
    return results
</code>
