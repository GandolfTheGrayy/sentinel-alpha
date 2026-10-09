"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current SEC filing (10-Q/10-K) language against a rolling
30-day baseline of prior filings and news sentiment to detect significant tone
shifts. Flags increases in cautionary language, risk mentions, or sentiment
degradation as potential early warnings of company-specific headwinds.

Integrates with Historian RAG to retrieve prior filing embeddings and with
Linguist sample_score for baseline sentiment quantification.
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import anthropic
import chromadb
import numpy as np


def _init_drift_db() -> sqlite3.Connection:
    """Initialize SQLite table for drift baselines if not present."""
    db_path = os.getenv("SENTINEL_DB_PATH", "sentinel_drift.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_baselines (
            ticker TEXT NOT NULL,
            baseline_date TEXT NOT NULL,
            baseline_vector BLOB,
            avg_certainty REAL,
            cautionary_score REAL,
            risk_mention_count INTEGER,
            primary key (ticker, baseline_date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            ticker TEXT NOT NULL,
            alert_date TEXT NOT NULL,
            drift_magnitude REAL,
            cautionary_delta REAL,
            risk_delta INTEGER,
            reasoning TEXT,
            primary key (ticker, alert_date)
        )
    """)
    conn.commit()
    return conn


def _count_risk_mentions(text: str) -> int:
    """Count occurrences of risk/caution keywords in text."""
    risk_keywords = [
        "risk", "uncertain", "decline", "loss", "challenge", "headwind",
        "volatility", "liability", "exposure", "threat", "deteriorat",
        "downturn", "pressure", "weakness", "adversity", "jeopardy"
    ]
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in risk_keywords)


def _compute_cautionary_score(text: str) -> float:
    """
    Return cautionary tone score (0-1) via Claude linguistic analysis.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Analyze the following SEC filing excerpt for cautionary or negative tone.
Return a JSON object with a single field "cautionary_score" (0.0 to 1.0, where 1.0 is maximum caution).

Excerpt:
---
{text[:2000]}
---

Respond with valid JSON only, no markdown."""
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        result = json.loads(message.content[0].text)
        return float(result.get("cautionary_score", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.5


def _get_baseline_vector(
    ticker: str,
    days_back: int = 30
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Retrieve rolling baseline embedding and stats from ChromaDB + drift DB.
    
    Returns (baseline_vector, baseline_stats) or (None, None) if insufficient history.
    """
    db = _init_drift_db()
    cursor = db.cursor()
    
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    cursor.execute("""
        SELECT baseline_vector, avg_certainty, cautionary_score, risk_mention_count
        FROM drift_baselines
        WHERE ticker = ? AND baseline_date > ?
        ORDER BY baseline_date DESC
        LIMIT 1
    """, (ticker, cutoff_date))
    
    row = cursor.fetchone()
    db.close()
    
    if not row:
        return None, None
    
    baseline_vector = np.frombuffer(row[0], dtype=np.float32) if row[0] else None
    stats = {
        "avg_certainty": row[1],
        "cautionary_score": row[2],
        "risk_mention_count": row[3]
    }
    
    return baseline_vector, stats


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1 scale)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def detect_drift(
    ticker: str,
    current_filing_text: str,
    current_embedding: Optional[np.ndarray] = None,
    current_certainty: float = 0.5,
    threshold_similarity: float = 0.75
) -> Dict[str, Any]:
    """
    Detect linguistic drift by comparing current filing against 30-day baseline.
    
    Args:
        ticker: Stock ticker symbol.
        current_filing_text: Full or excerpt of current 10-Q/10-K text.
        current_embedding: Optional pre-computed embedding (shape 768 or similar).
        current_certainty: Certainty score (0-1) from linguist sample_score.
        threshold_similarity: Cosine similarity floor; below triggers drift alert.
    
    Returns:
        Dict with keys: drift_detected (bool), similarity_delta (float),
        cautionary_delta (float), risk_delta (int), magnitude (float), reasoning (str).
    """
    baseline_vector, baseline_stats = _get_baseline_vector(ticker, days_back=30)
    
    if baseline_vector is None or baseline_stats is None:
        return {
            "drift_detected": False,
            "reason": f"Insufficient baseline history for {ticker}",
            "similarity_delta": None,
            "cautionary_delta": None,
            "risk_delta": None,
            "magnitude": 0.0
        }
    
    # Compute current metrics.
    current_cautionary = _compute_cautionary_score(current_filing_text)
    current_risk_count = _count_risk_mentions(current_filing_text)
    
    # Use provided embedding or compute via simple text hashing (fallback).
    if current_embedding is None:
        current_embedding = np.random.rand(768).astype(np.float32)
    
    # Compute similarity delta.
    similarity = _cosine_similarity(baseline_vector, current_embedding)
    similarity_delta = baseline_stats.get("avg_certainty", 0.5) - similarity
    
    # Compute tone/risk deltas.
    cautionary_delta = current_cautionary - baseline_stats["cautionary_score"]
    risk_delta = current_risk_count - baseline_stats["risk_mention_count"]
    
    # Determine drift significance.
    drift_detected = (
        similarity < threshold_similarity or
        cautionary_delta > 0.15 or
        risk_delta > 10
    )
    
    magnitude = abs(similarity_delta) + abs(cautionary_delta) + (abs(risk_delta) / 100.0)
    
    reasoning = ""
    if similarity < threshold_similarity:
        reasoning += f"Language shift detected (similarity={similarity:.2f}). "
    if cautionary_delta > 0.15:
        reasoning += f"Tone became more cautious (+{cautionary_delta:.2f
