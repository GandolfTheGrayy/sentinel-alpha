"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q/10-K language against a rolling 30-day
baseline of prior filings and news sentiment to flag significant tone shifts.
Uses Claude for semantic drift analysis and stores baseline embeddings in
ChromaDB for historical comparison.

Integrates with:
  - sentinel/scout/sec_filings.py (current filing text)
  - sentinel/scout/news.py (recent headlines)
  - sentinel/historian/rag_query.py (baseline embedding retrieval)
  - sentinel/linguist/sample_score.py (certainty scoring)
"""

import os
import json
from datetime import datetime, timedelta
from typing import TypedDict, Optional, Dict, List, Tuple
import sqlite3

import anthropic
import chromadb
from chromadb.config import Settings


class DriftSignal(TypedDict):
    """Structured drift detection result."""
    ticker: str
    company_name: str
    analysis_date: str
    drift_detected: bool
    drift_magnitude: float
    baseline_tone: str
    current_tone: str
    key_shifts: List[str]
    confidence: float
    raw_analysis: str


def _get_chroma_client() -> chromadb.HttpClient:
    """
    Initialize or retrieve ChromaDB client for baseline corpus.
    
    Returns:
        Configured ChromaDB HTTP client pointing to local/remote instance.
    """
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./data/chroma_drift",
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def _get_drift_db() -> sqlite3.Connection:
    """
    Initialize SQLite connection for drift detection history.
    
    Returns:
        SQLite connection with schema for storing drift signals.
    """
    db_path = "./data/drift_history.db"
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            company_name TEXT,
            analysis_date TEXT NOT NULL,
            drift_detected BOOLEAN,
            drift_magnitude REAL,
            baseline_tone TEXT,
            current_tone TEXT,
            key_shifts TEXT,
            confidence REAL,
            raw_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn


def retrieve_baseline_corpus(ticker: str, days: int = 30) -> List[Dict]:
    """
    Retrieve embedding-based baseline corpus from ChromaDB for a ticker over N days.
    
    Args:
        ticker: Stock ticker symbol.
        days: Lookback window in days (default 30).
    
    Returns:
        List of metadata dicts with text, date, source for baseline documents.
    """
    client = _get_chroma_client()
    try:
        collection = client.get_collection(name=f"baseline_{ticker.lower()}")
    except Exception:
        return []

    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    results = collection.get(
        where={"date": {"$gte": cutoff_date}},
        limit=50,
    )

    docs = []
    if results and "metadatas" in results:
        for meta in results["metadatas"]:
            docs.append({
                "text": meta.get("text", ""),
                "date": meta.get("date", ""),
                "source": meta.get("source", "unknown"),
            })
    return docs


def store_filing_baseline(
    ticker: str,
    company_name: str,
    filing_text: str,
    filing_date: str,
    form_type: str,
) -> bool:
    """
    Store current filing as part of rolling baseline in ChromaDB.
    
    Args:
        ticker: Stock ticker.
        company_name: Full company name.
        filing_text: Full text of 10-Q/10-K.
        filing_date: ISO format date string.
        form_type: "10-Q", "10-K", etc.
    
    Returns:
        True if stored successfully, False otherwise.
    """
    client = _get_chroma_client()
    collection_name = f"baseline_{ticker.lower()}"

    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)

    chunk_size = 2000
    chunks = [
        filing_text[i : i + chunk_size]
        for i in range(0, len(filing_text), chunk_size)
    ]

    for idx, chunk in enumerate(chunks):
        doc_id = f"{ticker}_{form_type}_{filing_date}_{idx}"
        collection.add(
            ids=[doc_id],
            documents=[chunk],
            metadatas=[{
                "ticker": ticker,
                "company_name": company_name,
                "date": filing_date,
                "form_type": form_type,
                "source": "sec_filing",
                "text": chunk[:500],
            }],
        )

    return True


def analyze_drift_with_claude(
    ticker: str,
    company_name: str,
    current_filing_text: str,
    baseline_corpus: List[Dict],
    current_date: str,
) -> DriftSignal:
    """
    Use Claude to perform nuanced semantic drift analysis comparing current filing against baseline.
    
    Args:
        ticker: Stock ticker.
        company_name: Full company name.
        current_filing_text: Full text of most recent 10-Q/10-K.
        baseline_corpus: List of prior filing/news texts from rolling window.
        current_date: ISO format date of analysis.
    
    Returns:
        DriftSignal with detected shifts, magnitude, and confidence.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    baseline_summary = "\n---\n".join([
        f"[{doc.get('source', 'unknown')} | {doc.get('date', '?')}]\n{doc.get('text', '')[:1000]}"
        for doc in baseline_corpus[:5]
    ])

    prompt = f"""You are a financial linguistic analyst for {company_name} ({ticker}).

TASK: Detect significant tone/sentiment shifts in current 10-Q language vs. historical baseline.

BASELINE (prior 30 days of filings & news):
{baseline_summary}

CURRENT FILING (excerpt):
{current_filing_text[:3000]}

Analyze for:
1. Tone shift (optimism → pessimism or vice versa)
2. Risk language increase/decrease
3. Cautionary language patterns
4. Regulatory/legal language intensity
5. Forward guidance changes

Output JSON:
{{
  "drift_detected": bool,
  "drift_magnitude": float (0.0-1.0, where 0.5+ is significant),
  "baseline_tone": "str (optimistic/neutral/pessimistic)",
  "current_tone": "str (optimistic/neutral/pessimistic)",
  "key_shifts": ["shift1", "shift2", ...],
  "confidence": float (0.0-1.0),
  "summary": "str (1-2 sentences)"
}}"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    try:
        response_text = message.content[0].text
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_
