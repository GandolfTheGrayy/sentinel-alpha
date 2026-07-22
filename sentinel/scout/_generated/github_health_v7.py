"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module fetches developer health indicators (stars, commit velocity,
issue open rate) for a given GitHub repository. These signals feed into
Scout's niche sentiment analysis pipeline to detect project momentum and
community engagement — proxies for technology adoption and risk sentiment.

Used by: sentinel/scout/ ingestion layer to enrich tech-sector predictions.
"""

import os
from typing import Optional
import requests


def get_github_repo_health(owner: str, repo: str) -> dict:
    """
    Fetch aggregated health metrics for a GitHub repository.
    
    Args:
        owner: GitHub organization or user (e.g., "openai")
        repo: Repository name (e.g., "gpt-4")
    
    Returns:
        Dict with keys: stars, commit_velocity_per_week, issue_open_rate,
        fetched_at (ISO timestamp), error (if any).
    """
    token = os.getenv("GITHUB_API_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    try:
        # Fetch repo metadata (stars, etc.)
        repo_resp = requests.get(f"{base_url}", headers=headers, timeout=10)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        
        stars = repo_data.get("stargazers_count", 0)
        
        # Fetch commits from last 7 days to compute velocity.
        import datetime
        one_week_ago = (
            datetime.datetime.utcnow() - datetime.timedelta(days=7)
        ).isoformat() + "Z"
        commits_url = (
            f"{base_url}/commits?since={one_week_ago}"
        )
        commits_resp = requests.get(commits_url, headers=headers, timeout=10)
        commits_resp.raise_for_status()
        commits_data = commits_resp.json()
        
        # Paginated response; check Link header for more pages.
        commit_count = len(commits_data) if isinstance(commits_data, list) else 0
        commit_velocity = commit_count  # commits per week
        
        # Fetch open issues count.
        issues_url = f"{base_url}/issues?state=open&per_page=1"
        issues_resp = requests.get(issues_url, headers=headers, timeout=10)
        issues_resp.raise_for_status()
        # Parse Link header to get total count (GitHub pagination trick).
        link_header = issues_resp.headers.get("Link", "")
        open_issues = 0
        if "last" in link_header:
            # Extract page number from last link.
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link_header)
            if match:
                open_issues = int(match.group(1))
        else:
            open_issues = len(issues_resp.json()) if issues_resp.json() else 0
        
        # Compute issue open rate (issues per star, normalized).
        issue_open_rate = open_issues / max(stars, 1)
        
        return {
            "owner": owner,
            "repo": repo,
            "stars": stars,
            "commit_velocity_per_week": commit_velocity,
            "issue_open_rate": issue_open_rate,
            "open_issues_count": open_issues,
            "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
            "error": None,
        }
    
    except requests.exceptions.RequestException as e:
        return {
            "owner": owner,
            "repo": repo,
            "stars": None,
            "commit_velocity_per_week": None,
            "issue_open_rate": None,
            "open_issues_count": None,
            "fetched_at": None,
            "error": str(e),
        }


def compute_health_score(health_data: dict) -> Optional[float]:
    """
    Synthesize repo health metrics into a 0–1 sentiment score.
    
    Args:
        health_data: Output from get_github_repo_health().
    
    Returns:
        Float 0–1 (1=healthiest), or None if error.
        Heuristic: high stars + low issue rate + high velocity = high score.
    """
    if health_data.get("error"):
        return None
    
    stars = health_data.get("stars", 0) or 0
    velocity = health_data.get("commit_velocity_per_week", 0) or 0
    issue_rate = health_data.get("issue_open_rate", 0) or 0
    
    # Normalize stars (assume 0–10k is typical range for active projects).
    stars_norm = min(stars / 10000, 1.0)
    
    # Normalize velocity (assume 0–20 commits/week is typical).
    velocity_norm = min(velocity / 20, 1.0)
    
    # Issue rate: lower is better; flip and cap at 0.
    issue_norm = max(1.0 - issue_rate, 0.0)
    
    # Weighted average: 40% stars, 40% velocity, 20% issue health.
    score = 0.4 * stars_norm + 0.4 * velocity_norm + 0.2 * issue_norm
    
    return score


if __name__ == "__main__":
    # Example usage / manual test.
    health = get_github_repo_health("openai", "gpt-4")
    print("Health data:", health)
    score = compute_health_score(health)
    print("Health score:", score)
