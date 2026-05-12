"""
Linguistic Drift Detector — Sentinel Linguist pillar.

Compares a company's current 10-Q/8-K language against a rolling 30-day baseline
and flags significant tone shifts via embedding similarity and keyword frequency analysis.
Feeds anomaly scores into Judge for prediction confidence calibration.

Uses Gemini embeddings (fast, high-volume) for similarity; Claude for nuanced
drift interpretation when confidence threshold is crossed.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from anthropic import Anthropic
from google.generativeai import Client, embed_content
from google.generativeai.types import EmbedContentRequest

# Initialize clients from env
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_client = Client(api_key=GEMINI_API_KEY)

# SQLite DB for baseline corpus
DRIFT_DB = "sentinel_drift.db"


def init_drift_db() -> None:
    """Initialize SQLite schema for baseline text corpus and embeddings."""
    conn = sqlite3.connect(DRIFT_DB)
    c = conn.cursor()
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS text_baseline (
        id INTEGER PRIMARY KEY,
        ticker TEXT NOT NULL,
        source TEXT NOT NULL,
        ingested_date TEXT NOT NULL,
        text_snippet TEXT NOT NULL,
        embedding BLOB NOT NULL,
        doc_type TEXT
    )
    """
    )
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS drift_scores (
        id INTEGER PRIMARY KEY,
        ticker TEXT NOT NULL,
        comparison_date TEXT NOT NULL,
        baseline_window_start TEXT NOT NULL,
        embedding_similarity REAL NOT NULL,
        keyword_divergence REAL NOT NULL,
        tone_shift_flag INTEGER NOT NULL,
        final_drift_score REAL NOT NULL,
        interpretation TEXT
    )
    """
    )
    conn.commit()
    conn.close()


def embed_text_gemini(text: str) -> np.ndarray:
    """Embed text snippet via Gemini; return normalized vector."""
    request = EmbedContentRequest(
        model="models/embedding-001",
        content=text,
    )
    response = embed_content(request)
    embedding = np.array(response.embedding, dtype=np.float32)
    # Normalize to unit vector
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def ingest_baseline_text(
    ticker: str, source: str, text_snippet: str, doc_type: str = "news"
) -> None:
    """
    Ingest a text snippet into the 30-day baseline corpus with embedding.
    """
    embedding = embed_text_gemini(text_snippet)
    embedding_bytes = embedding.tobytes()

    conn = sqlite3.connect(DRIFT_DB)
    c = conn.cursor()
    c.execute(
        """
    INSERT INTO text_baseline
    (ticker, source, ingested_date, text_snippet, embedding, doc_type)
    VALUES (?, ?, ?, ?, ?, ?)
    """,
        (
            ticker,
            source,
            datetime.utcnow().isoformat(),
            text_snippet,
            embedding_bytes,
            doc_type,
        ),
    )
    conn.commit()
    conn.close()


def get_baseline_embeddings(ticker: str, days_back: int = 30) -> list[np.ndarray]:
    """
    Retrieve all embeddings for a ticker from the past N days.
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()

    conn = sqlite3.connect(DRIFT_DB)
    c = conn.cursor()
    c.execute(
        """
    SELECT embedding FROM text_baseline
    WHERE ticker = ? AND ingested_date > ?
    """,
        (ticker, cutoff_date),
    )
    rows = c.fetchall()
    conn.close()

    embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in rows]
    return embeddings


def compute_embedding_similarity(
    current_embedding: np.ndarray, baseline_embeddings: list[np.ndarray]
) -> float:
    """
    Compute cosine similarity between current and mean baseline embedding.
    Returns scalar in [0, 1]; lower = more drift.
    """
    if not baseline_embeddings:
        return 0.5  # Neutral if no baseline

    mean_baseline = np.mean(baseline_embeddings, axis=0)
    mean_baseline_norm = np.linalg.norm(mean_baseline)
    if mean_baseline_norm > 0:
        mean_baseline = mean_baseline / mean_baseline_norm

    similarity = np.dot(current_embedding, mean_baseline)
    # Clamp to [0, 1] and convert so that lower similarity = higher drift
    similarity = max(0.0, min(1.0, similarity))
    return similarity


def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """
    Extract top N keywords (simple: split, filter stopwords, count freq).
    """
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "was",
        "are",
        "be",
        "have",
        "has",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "which",
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
    }

    words = text.lower().split()
    filtered = [w.strip(".,!?;:") for w in words if w.lower() not in stopwords]
    freq = pd.Series(filtered).value_counts().head(top_n)
    return freq.index.tolist()


def compute_keyword_divergence(
    current_keywords: list[str], baseline_keywords: list[str]
) -> float:
    """
    Compute keyword drift: (1 - Jaccard overlap) between current and baseline top keywords.
    Returns scalar in [0, 1]; higher = more drift.
    """
    current_set = set(current_keywords)
    baseline_set = set(baseline_keywords)

    if not baseline_set:
        return 0.5

    intersection = len(current_set & baseline_set)
    union = len(current_set | baseline_set)

    if union == 0:
        return 0.5

    jaccard = intersection / union
    divergence = 1.0 - jaccard
    return divergence


def detect_drift(
    ticker: str, current_text: str, threshold: float = 0.35
) -> dict:
    """
    Compare current 10-Q/8-K text against 30-day baseline.
    Returns dict with embedding_similarity, keyword_divergence, drift_flag, and final_score.
    """
    init_drift_db()

    # Embed current text
    current_embedding = embed_text_gemini(current_text)
    baseline_embeddings = get_baseline_embeddings(ticker, days_back=30)

    # Compute embedding similarity (high = similar to baseline, low = drift)
    embedding_sim = compute_embedding_similarity(current_embedding, baseline_embeddings)
    embedding_drift = 1.0 - embedding_sim

    # Compute keyword divergence
    current_keywords = extract_keywords(current_text, top_n=20)
    baseline
