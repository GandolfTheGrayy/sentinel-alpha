"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and detects significant tone shifts. Uses embedding
distance, keyword frequency analysis, and uncertainty/hedging language trends
to flag meaningful linguistic divergence that may precede market moves.

Integrated into the Linguist pillar for real-time tone anomaly detection.
"""

import os
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import numpy as np
import anthropic
import chromadb


def normalize_text(text: str) -> str:
    """Normalize filing text for comparison: lowercase, remove extra whitespace."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.strip()


def extract_hedging_language(text: str) -> dict[str, int]:
    """Extract counts of hedging/uncertainty keywords from text."""
    hedging_patterns = {
        'may': r'\bmay\b',
        'might': r'\bmight\b',
        'could': r'\bcould\b',
        'uncertain': r'\buncertain(ty)?\b',
        'risk': r'\brisk(s)?\b',
        'challenge': r'\bchallenge(s)?\b',
        'volatile': r'\bvolatile(ity)?\b',
        'difficult': r'\bdifficult(y)?\b',
        'pressure': r'\bpressure(s)?\b',
        'concern': r'\bconcern(s)?\b',
    }
    
    normalized = normalize_text(text)
    counts = {}
    for key, pattern in hedging_patterns.items():
        counts[key] = len(re.findall(pattern, normalized))
    return counts


def extract_positive_language(text: str) -> dict[str, int]:
    """Extract counts of positive/confidence keywords from text."""
    positive_patterns = {
        'growth': r'\bgrowth\b',
        'increase': r'\bincrease(d|s)?\b',
        'strong': r'\bstrong(ly)?\b',
        'improve': r'\bimprov(e|ed|ing|ement)\b',
        'opportunity': r'\bopportunity(ies)?\b',
        'expand': r'\bexpand(ing|ed|s)?\b',
        'success': r'\bsuccess(ful|fully)?\b',
        'revenue': r'\brevenue\b',
        'profit': r'\bprofit(s)?\b',
        'margin': r'\bmargin(s)?\b',
    }
    
    normalized = normalize_text(text)
    counts = {}
    for key, pattern in positive_patterns.items():
        counts[key] = len(re.findall(pattern, normalized))
    return counts


def compute_linguistic_distance(current_text: str, baseline_texts: list[str]) -> float:
    """
    Compute embedding-based distance between current text and baseline average.
    Returns distance metric (0 = identical, >1 = significant drift).
    """
    if not baseline_texts:
        return 0.0
    
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="drift_embeddings")
    
    # Normalize texts
    current_normalized = normalize_text(current_text)
    baseline_normalized = [normalize_text(t) for t in baseline_texts]
    
    # Add baseline documents to ChromaDB
    collection.delete_collection()
    collection = client.get_or_create_collection(name="drift_embeddings")
    
    for idx, text in enumerate(baseline_normalized):
        collection.add(
            documents=[text],
            ids=[f"baseline_{idx}"]
        )
    
    # Query with current text to get distance
    try:
        results = collection.query(
            query_texts=[current_normalized],
            n_results=len(baseline_normalized)
        )
        
        # ChromaDB returns distances; average them
        if results['distances'] and results['distances'][0]:
            avg_distance = float(np.mean(results['distances'][0]))
            return avg_distance
    except Exception:
        pass
    
    return 0.0


def analyze_tone_shift(
    current_text: str,
    baseline_texts: list[str]
) -> dict[str, float]:
    """
    Analyze tone shift between current and baseline texts.
    Returns scores for hedging, positivity, and overall drift.
    """
    current_hedge = extract_hedging_language(current_text)
    current_positive = extract_positive_language(current_text)
    
    baseline_hedge_list = [extract_hedging_language(t) for t in baseline_texts]
    baseline_positive_list = [extract_positive_language(t) for t in baseline_texts]
    
    # Compute averages
    avg_hedge = {}
    avg_positive = {}
    
    if baseline_hedge_list:
        for key in current_hedge.keys():
            avg_hedge[key] = np.mean([h[key] for h in baseline_hedge_list])
        for key in current_positive.keys():
            avg_positive[key] = np.mean([p[key] for p in baseline_positive_list])
    
    # Compute shifts (current vs. baseline)
    hedge_shift = {}
    positive_shift = {}
    
    for key in current_hedge.keys():
        baseline_val = avg_hedge.get(key, 0)
        current_val = current_hedge[key]
        hedge_shift[key] = (current_val - baseline_val) / (baseline_val + 1)
    
    for key in current_positive.keys():
        baseline_val = avg_positive.get(key, 0)
        current_val = current_positive[key]
        positive_shift[key] = (current_val - baseline_val) / (baseline_val + 1)
    
    # Aggregate metrics
    avg_hedge_shift = np.mean(list(hedge_shift.values())) if hedge_shift else 0.0
    avg_positive_shift = np.mean(list(positive_shift.values())) if positive_shift else 0.0
    
    # Embedding distance
    emb_distance = compute_linguistic_distance(current_text, baseline_texts)
    
    return {
        'hedging_shift': float(avg_hedge_shift),
        'positivity_shift': float(avg_positive_shift),
        'embedding_distance': emb_distance,
        'tone_direction': 'more_cautious' if avg_hedge_shift > 0.1 else
                         'more_confident' if avg_positive_shift > 0.1 else
                         'neutral'
    }


def fetch_baseline_filings(
    ticker: str,
    days_back: int = 30,
    db_path: str = 'sentinel.db'
) -> list[str]:
    """
    Fetch 10-Q filings from rolling window (default 30 days) for baseline.
    Returns list of filing texts.
    """
    baseline_texts = []
    
    if not os.path.exists(db_path):
        return baseline_texts
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_str = cutoff_date.isoformat()
    
    try:
        cursor.execute('''
            SELECT filing_text
            FROM sec_filings
            WHERE ticker = ? AND form_type IN ('10-Q', '10-K')
              AND filing_date >= ?
            ORDER BY filing_date DESC
        ''', (ticker, cutoff_str))
        
        rows = cursor.fetchall()
        baseline_texts = [row[0] for row in rows if row[0]]
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()
    
    return baseline_texts


def
