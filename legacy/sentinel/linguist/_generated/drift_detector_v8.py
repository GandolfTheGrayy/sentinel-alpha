"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and news headlines. Flags significant tone shifts
(e.g., increased caution, risk language, or optimism) that may precede
market moves. Used by the Linguist pillar to enrich certainty scoring.

Relies on ChromaDB for baseline corpus storage and Claude for semantic drift analysis.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import chromadb
from anthropic import Anthropic

client_llm = Anthropic()
client_chroma = chromadb.Client()


DRIFT_DB_PATH = "sentinel_drift.db"
BASELINE_COLLECTION_NAME = "linguistic_baseline_30d"


def init_drift_db() -> None:
    """Initialize SQLite table for drift detection metadata."""
    conn = sqlite3.connect(DRIFT_DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            filing_type TEXT,
            avg_sentiment_score REAL,
            risk_language_count INTEGER,
            optimism_language_count INTEGER,
            drift_magnitude REAL,
            drift_signals TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def store_baseline_embeddings(
    ticker: str, filing_text: str, filing_date: str, filing_type: str
) -> None:
    """Store filing text in ChromaDB baseline corpus for drift comparison."""
    collection = client_chroma.get_or_create_collection(
        name=BASELINE_COLLECTION_NAME,
        metadata={"description": "30-day rolling baseline of SEC filings and news"},
    )
    doc_id = f"{ticker}_{filing_type}_{filing_date}"
    collection.add(
        ids=[doc_id],
        documents=[filing_text],
        metadatas=[
            {
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
            }
        ],
    )


def retrieve_baseline_corpus(ticker: str, days: int = 30) -> list[dict]:
    """Retrieve embedding-based baseline documents for a ticker from the past N days."""
    collection = client_chroma.get_or_create_collection(
        name=BASELINE_COLLECTION_NAME
    )
    cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
    try:
        results = collection.get(
            where={"ticker": ticker, "filing_date": {"$gte": cutoff_date}}
        )
        return results.get("documents", [])
    except Exception:
        return []


def analyze_drift_with_claude(
    ticker: str,
    current_filing_text: str,
    current_filing_date: str,
    baseline_corpus: list[str],
) -> dict:
    """
    Use Claude to compare current filing tone against baseline and flag drift signals.
    Returns a dict with drift_magnitude (0-1), drift_signals (list of strings), and reasoning.
    """
    baseline_summary = (
        "\n---\n".join(baseline_corpus[:3])
        if baseline_corpus
        else "No baseline documents available."
    )

    prompt = f"""You are a financial linguist analyzing tone shifts in corporate filings.

Ticker: {ticker}
Current Filing Date: {current_filing_date}

CURRENT 10-Q EXCERPT (first 1500 chars):
{current_filing_text[:1500]}

BASELINE CORPUS (prior 30 days):
{baseline_summary}

Analyze the current filing for these drift signals:
1. Risk Language Increase: More "risk", "uncertain", "litigation", "contingent", "loss"
2. Caution Shift: Increase in hedging ("may", "could", "subject to")
3. Optimism Shift: More "growth", "expansion", "opportunity", "strong"
4. Operational Tone: Shift in language about supply chain, workforce, operations
5. Regulatory Mentions: New or increased references to compliance, investigations, penalties

Return a JSON object:
{
  "drift_magnitude": <float 0-1, where 0=no drift, 1=extreme drift>,
  "drift_signals": [<list of detected signals>],
  "risk_language_count": <int estimate>,
  "optimism_language_count": <int estimate>,
  "reasoning": "<explanation of detected shifts>"
}
"""

    response = client_llm.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        result = json.loads(response.content[0].text)
    except (json.JSONDecodeError, IndexError):
        result = {
            "drift_magnitude": 0.0,
            "drift_signals": [],
            "risk_language_count": 0,
            "optimism_language_count": 0,
            "reasoning": "Failed to parse Claude response.",
        }

    return result


def detect_drift(
    ticker: str,
    current_filing_text: str,
    current_filing_date: str,
    filing_type: str = "10-Q",
) -> dict:
    """
    Main entry point: detect linguistic drift for a ticker's current filing.
    Compares against 30-day rolling baseline.
    Returns a dict with drift_magnitude, signals, and stored DB record ID.
    """
    init_drift_db()

    baseline_corpus = retrieve_baseline_corpus(ticker, days=30)

    drift_analysis = analyze_drift_with_claude(
        ticker, current_filing_text, current_filing_date, baseline_corpus
    )

    conn = sqlite3.connect(DRIFT_DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO drift_snapshots
        (ticker, snapshot_date, filing_type, avg_sentiment_score, risk_language_count,
         optimism_language_count, drift_magnitude, drift_signals)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            ticker,
            current_filing_date,
            filing_type,
            0.5,
            drift_analysis.get("risk_language_count", 0),
            drift_analysis.get("optimism_language_count", 0),
            drift_analysis.get("drift_magnitude", 0.0),
            json.dumps(drift_analysis.get("drift_signals", [])),
        ),
    )
    conn.commit()
    record_id = c.lastrowid
    conn.close()

    store_baseline_embeddings(ticker, current_filing_text, current_filing_date, filing_type)

    return {
        "ticker": ticker,
        "drift_magnitude": drift_analysis.get("drift_magnitude", 0.0),
        "drift_signals": drift_analysis.get("drift_signals", []),
        "risk_language_count": drift_analysis.get("risk_language_count", 0),
        "optimism_language_count": drift_analysis.get("optimism_language_count", 0),
        "reasoning": drift_analysis.get("reasoning", ""),
        "db_record_id": record_id,
    }


def get_drift_history(ticker: str, limit: int = 10) -> list[dict]:
    """Retrieve recent drift detection records for a ticker from SQLite."""
    conn = sqlite3.connect(DRIFT_DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT * FROM drift_snapshots
        WHERE ticker = ?
        ORDER BY snapshot_date DESC
        LIMIT ?
    """,
        (ticker, limit),
    )
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def
