"""
GitHub Repository Health Signal Collector for Sentinel Scout.

Measures developer activity and project momentum via GitHub API:
- Star count (adoption signal)
- Commit velocity (commits/week, development cadence)
- Issue open rate (maintenance burden)

Integrates into scout pipeline to detect technical health shifts
that precede or correlate with stock price movements for tech companies.
"""

import os
import time
from typing import Optional
import requests


def get_github_repo_health(owner: str, repo: str) -> dict:
    """Fetch star count, commit velocity, and issue open rate for a GitHub repo."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    base_url = f"https://api.github.com/repos/{owner}/{repo}"

    try:
        # Fetch repo metadata (stars, etc.)
        repo_resp = requests.get(base_url, headers=headers, timeout=10)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        stars = repo_data.get("stargazers_count", 0)

        # Fetch commits from last 7 days to calculate velocity
        commits_url = f"{base_url}/commits"
        one_week_ago = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - 7 * 24 * 3600)
        )
        commits_resp = requests.get(
            commits_url,
            headers=headers,
            params={"since": one_week_ago, "per_page": 100},
            timeout=10
        )
        commits_resp.raise_for_status()
        commits_data = commits_resp.json()
        commits_per_week = len(commits_data) if isinstance(commits_data, list) else 0

        # Fetch open issues
        issues_url = f"{base_url}/issues"
        issues_resp = requests.get(
            issues_url,
            headers=headers,
            params={"state": "open", "per_page": 1},
            timeout=10
        )
        issues_resp.raise_for_status()
        # GitHub returns link header with total count; fall back to pagination
        issues_link = issues_resp.headers.get("link", "")
        open_issues = 0
        if "last" in issues_link:
            # Parse last page number from link header
            parts = issues_link.split(",")
            for part in parts:
                if 'rel="last"' in part:
                    import re
                    match = re.search(r"page=(\d+)>", part)
                    if match:
                        open_issues = int(match.group(1)) * 100 - (100 - len(issues_resp.json()))
                    break
        else:
            open_issues = len(issues_resp.json())

        # Fetch closed issues in last 7 days for open rate denominator
        closed_resp = requests.get(
            issues_url,
            headers=headers,
            params={"state": "closed", "since": one_week_ago, "per_page": 1},
            timeout=10
        )
        closed_resp.raise_for_status()
        closed_issues_week = len(closed_resp.json())

        # Calculate open rate: open / (open + closed) in last week
        total_issues_week = open_issues + closed_issues_week
        issue_open_rate = (
            open_issues / total_issues_week if total_issues_week > 0 else 0.0
        )

        return {
            "owner": owner,
            "repo": repo,
            "stars": stars,
            "commits_per_week": commits_per_week,
            "issue_open_rate": issue_open_rate,
            "open_issues_count": open_issues,
            "timestamp": time.time(),
            "status": "success"
        }

    except requests.exceptions.RequestException as e:
        return {
            "owner": owner,
            "repo": repo,
            "stars": None,
            "commits_per_week": None,
            "issue_open_rate": None,
            "open_issues_count": None,
            "timestamp": time.time(),
            "status": "error",
            "error": str(e)
        }


def compare_repo_health(
    current: dict,
    previous: Optional[dict]
) -> dict:
    """Compare current health snapshot to previous; return deltas and momentum flags."""
    if not previous or previous.get("status") != "success" or current.get("status") != "success":
        return {
            "comparison_status": "insufficient_data",
            "star_delta": None,
            "velocity_delta": None,
            "issue_rate_delta": None,
            "flags": []
        }

    star_delta = current.get("stars", 0) - previous.get("stars", 0)
    velocity_delta = current.get("commits_per_week", 0) - previous.get("commits_per_week", 0)
    issue_rate_delta = current.get("issue_open_rate", 0.0) - previous.get("issue_open_rate", 0.0)

    flags = []
    if star_delta < 0:
        flags.append("star_decline")
    if velocity_delta < -5:
        flags.append("commit_slowdown")
    if issue_rate_delta > 0.15:
        flags.append("issue_backlog_growth")

    return {
        "comparison_status": "success",
        "star_delta": star_delta,
        "velocity_delta": velocity_delta,
        "issue_rate_delta": issue_rate_delta,
        "flags": flags
    }
