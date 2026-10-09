"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and news sentiment to flag significant tone shifts.
Used by Judge to weight prediction confidence and identify regime changes.

Integrates with Historian's RAG pipeline to retrieve historical language
embeddings and applies cosine similarity + statistical anomaly detection.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from anthropic import Anthropic

# Initialize Anthropic client for reasoning about linguistic patterns.
client = Anthropic()

# In-memory cache for baseline embeddings during a single run.
_baseline_cache: dict = {}


def fetch_baseline_embeddings(
    ticker: str,
    db_path: str = "sentinel_history.db",
    days_back: int = 30,
) -> list[dict]:
    """
    Fetch embeddings and metadata for a ticker from the past N days.

    Args:
        ticker: Stock ticker symbol.
        db_path: Path to ChromaDB or SQLite history store.
        days_back: Number of days to lookback for baseline.

    Returns:
        List of dicts with keys: timestamp, source, text, embedding.
    """
    cutoff = datetime.utcnow() - timedelta(days=days_back)

    # Simulate retrieval from ChromaDB or local vector store.
    # In production, this queries sentinel/historian/ RAG pipeline.
    results = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT timestamp, source, text, embedding
            FROM linguistic_history
            WHERE ticker = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """,
            (ticker, cutoff.isoformat()),
        )
        for row in cursor.fetchall():
            ts, src, txt, emb = row
            results.append(
                {
                    "timestamp": ts,
                    "source": src,
                    "text": txt,
                    "embedding": emb,
                }
            )
        conn.close()
    except sqlite3.OperationalError:
        # Table may not exist; return empty baseline.
        pass

    return results


def compute_baseline_vector(
    ticker: str,
    embeddings: list[dict],
) -> Optional[np.ndarray]:
    """
    Compute mean embedding vector from baseline documents.

    Args:
        ticker: Stock ticker symbol.
        embeddings: List of embedding dicts from baseline window.

    Returns:
        Mean embedding vector (float32), or None if no data.
    """
    if not embeddings:
        return None

    # Parse embeddings (stored as CSV or JSON strings).
    vectors = []
    for item in embeddings:
        emb = item.get("embedding")
        if isinstance(emb, str):
            # Parse CSV-like embedding string.
            try:
                vec = np.array([float(x) for x in emb.split(",")])
                vectors.append(vec)
            except ValueError:
                continue
        elif isinstance(emb, (list, np.ndarray)):
            vectors.append(np.array(emb, dtype=np.float32))

    if not vectors:
        return None

    return np.mean(vectors, axis=0).astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Similarity score in [-1, 1].
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def detect_drift(
    ticker: str,
    current_text: str,
    current_embedding: np.ndarray,
    baseline_embeddings: Optional[list[dict]] = None,
    threshold: float = 0.15,
) -> dict:
    """
    Detect linguistic drift by comparing current embedding to baseline mean.

    Args:
        ticker: Stock ticker symbol.
        current_text: Current 10-Q or filing text.
        current_embedding: Current embedding vector.
        baseline_embeddings: Cached baseline embeddings (or fetch if None).
        threshold: Cosine distance threshold for flagging drift (0-1).

    Returns:
        Dict with keys:
            - drift_detected: bool
            - similarity_score: float
            - distance_from_baseline: float
            - risk_level: "low" | "medium" | "high"
            - explanation: str
    """
    if baseline_embeddings is None:
        baseline_embeddings = fetch_baseline_embeddings(ticker)

    baseline_vec = compute_baseline_vector(ticker, baseline_embeddings)

    if baseline_vec is None:
        # No historical baseline; treat as neutral drift.
        return {
            "drift_detected": False,
            "similarity_score": 1.0,
            "distance_from_baseline": 0.0,
            "risk_level": "low",
            "explanation": "No baseline history; insufficient data to detect drift.",
        }

    # Normalize current embedding.
    current_embedding = np.array(current_embedding, dtype=np.float32)
    similarity = cosine_similarity(current_embedding, baseline_vec)
    distance = 1.0 - similarity

    drift_detected = distance > threshold

    # Map distance to risk level.
    if distance < 0.10:
        risk_level = "low"
    elif distance < 0.25:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "drift_detected": drift_detected,
        "similarity_score": float(similarity),
        "distance_from_baseline": float(distance),
        "risk_level": risk_level,
        "explanation": f"Cosine distance {distance:.3f} vs threshold {threshold}.",
    }


def analyze_tone_shift_with_claude(
    ticker: str,
    current_text: str,
    baseline_samples: Optional[list[str]] = None,
) -> dict:
    """
    Use Claude to perform nuanced linguistic analysis of tone shift.

    Args:
        ticker: Stock ticker symbol.
        current_text: Current filing excerpt (first 2000 chars).
        baseline_samples: Sample sentences from baseline period (or None).

    Returns:
        Dict with keys:
            - tone_category: str (e.g. "cautious", "optimistic", "neutral")
            - sentiment_shift: str (e.g. "deteriorating", "stable", "improving")
            - key_phrase_changes: list[str]
            - confidence: float (0-1)
    """
    baseline_context = ""
    if baseline_samples:
        baseline_context = "\n".join(baseline_samples[:3])

    prompt = f"""
You are analyzing linguistic tone shifts for {ticker}.

Current 10-Q excerpt:
{current_text[:2000]}

Historical baseline samples (last 30 days):
{baseline_context if baseline_context else "(No baseline available)"}

Identify:
1. Current tone (cautious/neutral/optimistic)
2. Sentiment shift direction (deteriorating/stable/improving)
3. Key phrase changes that signal the shift
4. Confidence in this assessment (0-1)

Respond in JSON format:
{{
  "tone_category": "...",
  "sentiment_shift": "...",
  "key_phrases": ["phrase1", "phrase2"],
  "confidence": 0.85
}}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse Claude's JSON response.
    response_text = message.content[0].text
