"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q/10-K language against a rolling 30-day
baseline of historical filings and news sentiment to flag significant tone shifts.
Used by Judge to weight predictions when company narrative changes sharply.

Detects:
  - Sentiment polarity drift (positive → negative language increases)
  - Uncertainty markers (hedging language frequency shifts)
  - Topic emergence (new risk keywords appearing)
  - Regulatory language intensity (compliance/legal terminology spikes)

Integrates with Historian's ChromaDB to retrieve baseline embeddings and
with Linguist's certainty scorer for normalized drift scoring.
"""

import os
import json
from datetime import datetime, timedelta
from typing import dict, list, tuple, Optional
import sqlite3

import numpy as np
from anthropic import Anthropic


def _get_drift_db_path() -> str:
    """Return path to drift tracking SQLite database."""
    db_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "drift_baseline.db")


def _init_drift_db() -> None:
    """Initialize SQLite schema for baseline tracking."""
    db_path = _get_drift_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS baseline_snapshots (
            ticker TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            embedding BLOB NOT NULL,
            polarity_score REAL NOT NULL,
            uncertainty_score REAL NOT NULL,
            topic_keywords TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (ticker, snapshot_date, source)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            ticker TEXT NOT NULL,
            alert_date TEXT NOT NULL,
            drift_type TEXT NOT NULL,
            severity REAL NOT NULL,
            baseline_value REAL NOT NULL,
            current_value REAL NOT NULL,
            explanation TEXT,
            PRIMARY KEY (ticker, alert_date, drift_type)
        )
    """)
    conn.commit()
    conn.close()


def store_baseline_snapshot(
    ticker: str,
    embedding: list[float],
    polarity_score: float,
    uncertainty_score: float,
    topic_keywords: list[str],
    source: str = "filing",
) -> None:
    """Store a baseline linguistic snapshot for rolling 30-day comparison."""
    _init_drift_db()
    db_path = _get_drift_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    snapshot_date = datetime.utcnow().isoformat()
    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
    keywords_json = json.dumps(topic_keywords)
    
    c.execute("""
        INSERT OR REPLACE INTO baseline_snapshots
        (ticker, snapshot_date, embedding, polarity_score, uncertainty_score, topic_keywords, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (ticker, snapshot_date, embedding_blob, polarity_score, uncertainty_score, keywords_json, source))
    
    conn.commit()
    conn.close()


def retrieve_baseline_window(ticker: str, days: int = 30) -> list[dict]:
    """Fetch all baseline snapshots for a ticker within the past N days."""
    _init_drift_db()
    db_path = _get_drift_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    c.execute("""
        SELECT snapshot_date, embedding, polarity_score, uncertainty_score, topic_keywords, source
        FROM baseline_snapshots
        WHERE ticker = ? AND snapshot_date > ?
        ORDER BY snapshot_date DESC
    """, (ticker, cutoff))
    
    rows = c.fetchall()
    conn.close()
    
    snapshots = []
    for row in rows:
        snapshot_date, embedding_blob, polarity, uncertainty, keywords_json, source = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()
        keywords = json.loads(keywords_json)
        snapshots.append({
            "snapshot_date": snapshot_date,
            "embedding": embedding,
            "polarity_score": polarity,
            "uncertainty_score": uncertainty,
            "topic_keywords": keywords,
            "source": source,
        })
    
    return snapshots


def compute_polarity_drift(
    current_polarity: float,
    baseline_snapshots: list[dict],
) -> tuple[float, str]:
    """
    Compare current polarity against 30-day baseline median.
    
    Returns (drift_magnitude, direction) where drift_magnitude ∈ [0, 1]
    and direction ∈ ["positive", "negative", "neutral"].
    """
    if not baseline_snapshots:
        return 0.0, "neutral"
    
    baseline_polarities = [s["polarity_score"] for s in baseline_snapshots]
    baseline_median = np.median(baseline_polarities)
    baseline_std = np.std(baseline_polarities) or 0.1
    
    z_score = (current_polarity - baseline_median) / baseline_std
    drift_magnitude = min(abs(z_score) / 3.0, 1.0)
    
    direction = "positive" if z_score > 0.5 else ("negative" if z_score < -0.5 else "neutral")
    
    return drift_magnitude, direction


def compute_uncertainty_drift(
    current_uncertainty: float,
    baseline_snapshots: list[dict],
) -> tuple[float, str]:
    """
    Compare current uncertainty score against baseline.
    
    High drift indicates sudden increase/decrease in hedging language.
    Returns (drift_magnitude, direction) where direction ∈ ["increased", "decreased", "stable"].
    """
    if not baseline_snapshots:
        return 0.0, "stable"
    
    baseline_uncertainties = [s["uncertainty_score"] for s in baseline_snapshots]
    baseline_median = np.median(baseline_uncertainties)
    baseline_std = np.std(baseline_uncertainties) or 0.05
    
    z_score = (current_uncertainty - baseline_median) / baseline_std
    drift_magnitude = min(abs(z_score) / 2.5, 1.0)
    
    direction = "increased" if z_score > 0.3 else ("decreased" if z_score < -0.3 else "stable")
    
    return drift_magnitude, direction


def detect_topic_emergence(
    current_keywords: list[str],
    baseline_snapshots: list[dict],
) -> tuple[list[str], float]:
    """
    Identify new risk/regulatory keywords not in baseline window.
    
    Returns (emergent_topics, emergence_score) where emergence_score ∈ [0, 1].
    """
    if not baseline_snapshots:
        return current_keywords, 0.5
    
    baseline_all_keywords = set()
    for snapshot in baseline_snapshots:
        baseline_all_keywords.update(snapshot["topic_keywords"])
    
    emergent = [kw for kw in current_keywords if kw not in baseline_all_keywords]
    emergence_score = min(len(emergent) / max(len(current_keywords), 1), 1.0)
    
    return emergent, emergence_score


def flag_drift_alert(
    ticker: str,
    drift_type: str,
    severity: float,
    baseline_value: float,
    current_value: float,
    explanation: Optional[str] = None,
) -> None:
    """Store a drift alert for Judge post-mortem review."""
    _init_drift_db()
    db
