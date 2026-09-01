"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests developer activity signals from GitHub repositories,
measuring technical momentum (stars, commit velocity, issue open rate) as
leading indicators of company/project health. Used by Scout to augment
sentiment analysis with objective developer engagement metrics.

Functions:
  - fetch_repo_stats() — Retrieve stars, commit velocity, issue rate for a repo.
  - parse_github_url() — Extract owner/repo from various GitHub URL formats.
"""

import os
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import requests


def parse_github_url(url: str) -> Optional[Dict[str, str]]:
    """
    Extract owner and repo name from a GitHub URL.

    Supports formats: https://github.com/owner/repo, git@github.com:owner/repo.git, etc.
    Returns dict with keys 'owner' and 'repo', or None if URL is invalid.
    """
    # Remove .git suffix if present
    url_clean = url.rstrip('/').rstrip('.git')
    
    # Match https://github.com/owner/repo or similar
    https_match = re.search(r'github\.com[:/]+([^/]+)/([^/\s]+?)(?:\.git)?$', url_clean)
    if https_match:
        return {'owner': https_match.group(1), 'repo': https_match.group(2)}
    
    # Match git@github.com:owner/repo.git
    ssh_match = re.search(r'git@github\.com:([^/]+)/(.+?)(?:\.git)?$', url)
    if ssh_match:
        return {'owner': ssh_match.group(1), 'repo': ssh_match.group(2)}
    
    return None


def fetch_repo_stats(
    owner: str,
    repo: str,
    token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fetch GitHub repository health metrics: stars, commit velocity, issue open rate.

    Args:
        owner: GitHub username or organization.
        repo: Repository name.
        token: Optional GitHub API token (increases rate limit from 60/hr to 5000/hr).

    Returns:
        Dict with keys:
          - stars: Integer star count.
          - commit_velocity: Float commits per week (last 4 weeks).
          - issue_open_rate: Float ratio of open to total issues (last 4 weeks).
          - last_updated: ISO timestamp of repo last push.
          - url: Full GitHub repo URL.
          - error: String error message if fetch failed.

        Returns None on catastrophic failure (network, auth, 404).
    """
    base_url = f'https://api.github.com/repos/{owner}/{repo}'
    headers = {'Accept': 'application/vnd.github.v3+json'}
    
    if token:
        headers['Authorization'] = f'token {token}'
    
    result = {
        'owner': owner,
        'repo': repo,
        'url': f'https://github.com/{owner}/{repo}',
        'stars': None,
        'commit_velocity': None,
        'issue_open_rate': None,
        'last_updated': None,
        'error': None,
    }
    
    # Fetch basic repo info (stars, last push)
    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        if resp.status_code == 404:
            result['error'] = f'Repository {owner}/{repo} not found (404).'
            return result
        resp.raise_for_status()
        data = resp.json()
        result['stars'] = data.get('stargazers_count', 0)
        result['last_updated'] = data.get('pushed_at')
    except requests.exceptions.RequestException as e:
        result['error'] = f'Failed to fetch repo info: {str(e)}'
        return result
    
    # Fetch commit activity (commits in last 4 weeks)
    try:
        commits_url = f'{base_url}/commits'
        since = (datetime.utcnow() - timedelta(weeks=4)).isoformat() + 'Z'
        params = {'since': since, 'per_page': 100}
        resp = requests.get(commits_url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        commits = resp.json()
        
        # Calculate commits per week
        commit_count = len(commits) if isinstance(commits, list) else 0
        result['commit_velocity'] = round(commit_count / 4.0, 2)
    except requests.exceptions.RequestException as e:
        result['error'] = (
            result['error'] or ''
        ) + f' Failed to fetch commits: {str(e)}'
    
    # Fetch issue stats (open vs closed in last 4 weeks)
    try:
        issues_url = f'{base_url}/issues'
        since = (datetime.utcnow() - timedelta(weeks=4)).isoformat() + 'Z'
        
        # Count open issues
        params_open = {'state': 'open', 'since': since, 'per_page': 1}
        resp_open = requests.get(issues_url, headers=headers, params=params_open, timeout=10)
        resp_open.raise_for_status()
        # GitHub returns Link header with total count if per_page=1
        link_header = resp_open.headers.get('Link', '')
        open_count = 0
        if 'last' in link_header:
            # Extract page number from last link
            last_match = re.search(r'page=(\d+)', link_header)
            if last_match:
                open_count = int(last_match.group(1))
        
        # Count closed issues in same period
        params_closed = {'state': 'closed', 'since': since, 'per_page': 1}
        resp_closed = requests.get(issues_url, headers=headers, params=params_closed, timeout=10)
        resp_closed.raise_for_status()
        link_header = resp_closed.headers.get('Link', '')
        closed_count = 0
        if 'last' in link_header:
            last_match = re.search(r'page=(\d+)', link_header)
            if last_match:
                closed_count = int(last_match.group(1))
        
        total_issues = open_count + closed_count
        if total_issues > 0:
            result['issue_open_rate'] = round(open_count / float(total_issues), 3)
        else:
            result['issue_open_rate'] = 0.0
    except requests.exceptions.RequestException as e:
        result['error'] = (
            result['error'] or ''
        ) + f' Failed to fetch issues: {str(e)}'
    
    return result


def main() -> None:
    """
    Standalone test harness: fetch and print health stats for a hardcoded repo.
    """
    token = os.environ.get('GITHUB_TOKEN')
    stats = fetch_repo_stats('anthropics', 'anthropic-sdk-python', token=token)
    if stats:
        print(f"Repository: {stats['url']}")
        print(f"  Stars: {stats['stars']}")
        print(f"  Commit Velocity (commits/week): {stats['commit_velocity']}")
        print(f"  Issue Open Rate: {stats['issue_open_rate']}")
        print(f"  Last Updated: {stats['last_updated']}")
        if stats['error']:
            print(f"  Warnings: {stats['error']}")
    else:
        print("Failed to fetch repository stats.")


if __name__ == '__main__':
    main()
