"""
GitHub repository health signal collector for Sentinel Sentiment Engine.

Measures developer activity and sentiment proxies:
  - Stars: community adoption signal
  - Commit velocity (commits/week): development momentum
  - Issue open rate: maintenance burden / engineering health

Integrated into scout pillar for cross-referencing with stock movements
of companies whose primary products are tracked via GitHub (e.g., open-source
infrastructure, developer tools).

Uses GitHub REST API v3 (no authentication required for public repos, but
rate-limited to 60 req/hour; consider GITHUB_TOKEN env var for 5000 req/hour).
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional
import requests


GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)


def _github_headers() -> dict[str, str]:
    """Return HTTP headers for GitHub API requests, including auth token if available."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    return headers


def fetch_repo_stars(owner: str, repo: str) -> Optional[int]:
    """Fetch current star count for a GitHub repository."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    try:
        resp = requests.get(url, headers=_github_headers(), timeout=10)
        resp.raise_for_status()
        return resp.json().get("stargazers_count")
    except requests.RequestException as e:
        print(f"Error fetching stars for {owner}/{repo}: {e}")
        return None


def fetch_commit_velocity(owner: str, repo: str, weeks: int = 4) -> Optional[float]:
    """Fetch average commits per week over the last N weeks."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    since = (datetime.utcnow() - timedelta(weeks=weeks)).isoformat() + "Z"
    
    try:
        resp = requests.get(
            url,
            headers=_github_headers(),
            params={"since": since, "per_page": 100},
            timeout=10
        )
        resp.raise_for_status()
        commits = resp.json()
        
        # GitHub returns paginated results; commits list here is up to 100.
        # For a more complete count, we'd need to follow pagination links.
        commit_count = len(commits)
        velocity = commit_count / weeks if weeks > 0 else 0.0
        return velocity
    except requests.RequestException as e:
        print(f"Error fetching commit velocity for {owner}/{repo}: {e}")
        return None


def fetch_issue_open_rate(owner: str, repo: str) -> Optional[float]:
    """Fetch ratio of open issues to total issues (open + closed) as maintenance signal."""
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
    
    try:
        # Fetch open issues
        resp_open = requests.get(
            url,
            headers=_github_headers(),
            params={"state": "open", "per_page": 1},
            timeout=10
        )
        resp_open.raise_for_status()
        
        # Extract total count from Link header or use response length as approximation
        link_header = resp_open.headers.get("Link", "")
        open_count = 0
        if "last" in link_header:
            # Parse last page number from Link header
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link_header)
            if match:
                open_count = int(match.group(1))
        else:
            open_count = len(resp_open.json())
        
        # Fetch closed issues
        resp_closed = requests.get(
            url,
            headers=_github_headers(),
            params={"state": "closed", "per_page": 1},
            timeout=10
        )
        resp_closed.raise_for_status()
        
        link_header = resp_closed.headers.get("Link", "")
        closed_count = 0
        if "last" in link_header:
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link_header)
            if match:
                closed_count = int(match.group(1))
        else:
            closed_count = len(resp_closed.json())
        
        total = open_count + closed_count
        rate = open_count / total if total > 0 else 0.0
        return rate
    except requests.RequestException as e:
        print(f"Error fetching issue open rate for {owner}/{repo}: {e}")
        return None


def collect_repo_health(owner: str, repo: str) -> dict:
    """Collect all health signals for a given GitHub repository."""
    stars = fetch_repo_stars(owner, repo)
    velocity = fetch_commit_velocity(owner, repo, weeks=4)
    issue_rate = fetch_issue_open_rate(owner, repo)
    
    return {
        "owner": owner,
        "repo": repo,
        "timestamp": datetime.utcnow().isoformat(),
        "stars": stars,
        "commit_velocity_per_week": velocity,
        "issue_open_rate": issue_rate,
    }


if __name__ == "__main__":
    # Example: collect health for a prominent open-source project
    result = collect_repo_health("kubernetes", "kubernetes")
    print(f"GitHub Health for kubernetes/kubernetes:\n{result}")
