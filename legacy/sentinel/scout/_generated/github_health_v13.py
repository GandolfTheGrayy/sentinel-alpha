"""
GitHub Repository Health Signal Collector for Sentinel Sentiment Engine.

This module ingests repository signals (stars, commit velocity, issue open rate)
for developer health assessment. Used by Scout to enrich sentiment analysis with
technical momentum indicators—high velocity + rising stars often correlate with
positive sentiment drift and future stock outperformance for tech companies.

Signals feed into Historian RAG as auxiliary context vectors alongside news/SEC data.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import requests


def _get_github_token() -> str:
    """Retrieve GitHub API token from environment; raise if missing."""
    token = os.getenv("GITHUB_API_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_API_TOKEN not set. GitHub API requires authentication for higher rate limits."
        )
    return token


def fetch_repo_stars(owner: str, repo: str) -> Optional[int]:
    """Fetch current star count for a repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {_get_github_token()}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("stargazers_count")
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching stars for {owner}/{repo}: {e}")
        return None


def fetch_commit_velocity(owner: str, repo: str, weeks: int = 4) -> Optional[float]:
    """Compute commits per week over the past N weeks."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {_get_github_token()}"}
    since = (datetime.utcnow() - timedelta(weeks=weeks)).isoformat() + "Z"
    params = {"since": since, "per_page": 100}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        commit_count = len(resp.json())
        velocity = commit_count / weeks if weeks > 0 else 0.0
        return velocity
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching commit velocity for {owner}/{repo}: {e}")
        return None


def fetch_issue_open_rate(owner: str, repo: str) -> Optional[float]:
    """Compute ratio of open issues to total issues ever filed."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    headers = {"Authorization": f"token {_get_github_token()}"}
    params = {"state": "open", "per_page": 1}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        open_count = resp.json()[0].get("number", 0) if resp.json() else 0

        params["state"] = "all"
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        total_count = resp.json()[0].get("number", 1) if resp.json() else 1

        rate = open_count / total_count if total_count > 0 else 0.0
        return rate
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching issue open rate for {owner}/{repo}: {e}")
        return None


def collect_repo_health(owner: str, repo: str) -> dict:
    """Aggregate stars, commit velocity, and issue open rate into a single signal dict."""
    return {
        "owner": owner,
        "repo": repo,
        "timestamp": datetime.utcnow().isoformat(),
        "stars": fetch_repo_stars(owner, repo),
        "commit_velocity_per_week": fetch_commit_velocity(owner, repo),
        "issue_open_rate": fetch_issue_open_rate(owner, repo),
    }


def store_repo_health(db_path: str, signal: dict) -> None:
    """Persist repo health signal to SQLite for historical tracking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS github_health (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner TEXT NOT NULL,
            repo TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            stars INTEGER,
            commit_velocity_per_week REAL,
            issue_open_rate REAL
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO github_health
        (owner, repo, timestamp, stars, commit_velocity_per_week, issue_open_rate)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            signal["owner"],
            signal["repo"],
            signal["timestamp"],
            signal["stars"],
            signal["commit_velocity_per_week"],
            signal["issue_open_rate"],
        ),
    )

    conn.commit()
    conn.close()


def get_repo_health_trend(
    db_path: str, owner: str, repo: str, days: int = 30
) -> dict:
    """Retrieve aggregated health metrics over a time window."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    cursor.execute(
        """
        SELECT
            AVG(stars) as avg_stars,
            MAX(stars) as max_stars,
            AVG(commit_velocity_per_week) as avg_velocity,
            AVG(issue_open_rate) as avg_open_rate,
            COUNT(*) as sample_count
        FROM github_health
        WHERE owner = ? AND repo = ? AND timestamp >= ?
        """,
        (owner, repo, cutoff),
    )

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "owner": owner,
            "repo": repo,
            "days_lookback": days,
            "avg_stars": row[0],
            "max_stars": row[1],
            "avg_commit_velocity": row[2],
            "avg_issue_open_rate": row[3],
            "sample_count": row[4],
        }
    return {}
