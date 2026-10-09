"""
Linguistic Drift Detector for Sentinel.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and sentiment signals. Flags significant tone shifts
(e.g., increasing risk language, declining confidence markers) that may precede
market moves. Integrates with ChromaDB for historical embedding lookups and
Claude for nuanced linguistic analysis.

Role in Sentinel:
  - Ingests recent 10-Q text via Scout (sentinel/scout/sec_filings.py).
  - Embeds filings into ChromaDB vector DB alongside historical baseline.
  - Compares current embedding + keyword drift against 30-day rolling window.
  - Scores tone shift magnitude (0–1) and flags anomalies.
  - Output feeds into Judge for prediction calibration.
"""

import os
import re
from typing import Optional
from datetime import datetime, timedelta
import sqlite3

import chromadb
import numpy as np
from anthropic import Anthropic


# ──────────────────────────────────────────────────────────────────────────────
# Configuration & Constants
# ──────────────────────────────────────────────────────────────────────────────

RISK_KEYWORDS = [
    "risk", "uncertain", "volatile", "decline", "loss", "impair", "exposure",
    "stress", "default", "bankruptcy", "litigation", "breach", "fraud",
    "weakness", "deteriorat", "downside", "headwind", "challenge"
]

CONFIDENCE_KEYWORDS = [
    "strong", "robust", "solid", "growth", "expand", "opportunity", "upside",
    "momentum", "confident", "positive", "improvement", "achieve", "exceed",
    "leverage", "synerg", "strategic", "excellence"
]

BASELINE_DAYS = 30
DRIFT_THRESHOLD = 0.25  # Tone shift score above this flags anomaly


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB & History Management
# ──────────────────────────────────────────────────────────────────────────────

def init_drift_db(db_path: str = ".sentinel_drift.db") -> sqlite3.Connection:
    """Initialize SQLite store for filing metadata and drift scores."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            filing_type TEXT,
            embedding_id TEXT,
            risk_score REAL,
            confidence_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_scores (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            current_date TEXT NOT NULL,
            baseline_date TEXT NOT NULL,
            tone_drift REAL,
            risk_delta REAL,
            confidence_delta REAL,
            anomaly_flag INTEGER,
            analysis_summary TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def get_baseline_window(
    conn: sqlite3.Connection,
    ticker: str,
    days: int = BASELINE_DAYS
) -> list[dict]:
    """Retrieve filings from rolling window (default 30 days ago to now)."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, ticker, filing_date, risk_score, confidence_score, embedding_id
        FROM filings
        WHERE ticker = ? AND filing_date > ?
        ORDER BY filing_date DESC
    """, (ticker, cutoff))
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "ticker": r[1],
            "filing_date": r[2],
            "risk_score": r[3],
            "confidence_score": r[4],
            "embedding_id": r[5],
        }
        for r in rows
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Keyword & Tone Analysis
# ──────────────────────────────────────────────────────────────────────────────

def extract_tone_metrics(text: str) -> dict:
    """
    Count risk vs. confidence keywords in filing text.
    Returns: {"risk_count": int, "confidence_count": int, "risk_density": float, "confidence_density": float}
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())

    risk_count = sum(
        len(re.findall(r'\b' + kw + r'\w*\b', text_lower))
        for kw in RISK_KEYWORDS
    )
    confidence_count = sum(
        len(re.findall(r'\b' + kw + r'\w*\b', text_lower))
        for kw in CONFIDENCE_KEYWORDS
    )

    risk_density = risk_count / max(word_count, 1)
    confidence_density = confidence_count / max(word_count, 1)

    return {
        "risk_count": risk_count,
        "confidence_count": confidence_count,
        "risk_density": risk_density,
        "confidence_density": confidence_density,
    }


def compute_tone_score(metrics: dict) -> float:
    """
    Compute normalized tone score: 0 = all risk, 1 = all confidence.
    Score = confidence_density / (risk_density + confidence_density + ε)
    """
    risk_density = metrics["risk_density"]
    confidence_density = metrics["confidence_density"]
    total = risk_density + confidence_density + 1e-6
    return confidence_density / total


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB Integration
# ──────────────────────────────────────────────────────────────────────────────

def embed_filing(
    client: chromadb.HttpClient,
    collection_name: str,
    ticker: str,
    filing_date: str,
    text: str,
    embedding_id: str,
) -> None:
    """
    Embed filing text into ChromaDB collection.
    Uses Gemini embeddings (via ChromaDB default).
    """
    collection = client.get_or_create_collection(name=collection_name)
    collection.add(
        ids=[embedding_id],
        documents=[text],
        metadatas=[{
            "ticker": ticker,
            "filing_date": filing_date,
        }]
    )


def query_similar_filings(
    client: chromadb.HttpClient,
    collection_name: str,
    query_text: str,
    ticker: str,
    n_results: int = 5,
) -> list[dict]:
    """
    Query ChromaDB for semantically similar filings within same ticker.
    Returns: list of {"id": str, "distance": float, "metadata": dict}
    """
    collection = client.get_or_create_collection(name=collection_name)
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"ticker": ticker},
    )
    return [
        {
            "id": results["ids"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Drift Detection & Scoring
# ──────────────────────────────────────────────────────────────────────────────

def detect_drift(
    current_text: str,
    baseline_metrics_list: list[dict],
) -> dict:
    """
    Compare current filing metrics against rolling baseline.
    Returns: {
        "current_tone_
