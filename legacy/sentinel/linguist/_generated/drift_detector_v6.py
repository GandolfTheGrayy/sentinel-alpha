"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and sentiment snapshots. Flags significant tone shifts
(e.g., risk language increases, confidence decreases) that may precede market moves.

Integrates with:
  - sentinel/historian/ (retrieve historical filings via RAG)
  - Claude (nuanced tone analysis and drift scoring)
  - sentinel/scout/sec_filings.py (fetch current 10-Q text)
"""

import os
import json
from typing import TypedDict, Optional
from datetime import datetime, timedelta
import sqlite3

import anthropic


class DriftSignal(TypedDict):
    """Structured output of linguistic drift analysis."""
    ticker: str
    filing_date: str
    baseline_start: str
    baseline_end: str
    risk_language_delta: float
    confidence_delta: float
    tone_shift_score: float
    key_shifts: list[str]
    severity: str
    recommendation: str


def _init_drift_db(db_path: str = "sentinel_drift.db") -> None:
    """Initialize SQLite table for storing linguistic baselines."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS linguistic_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            baseline_type TEXT NOT NULL,
            risk_score REAL,
            confidence_score REAL,
            tone_vector TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def _store_baseline(
    ticker: str,
    filing_date: str,
    risk_score: float,
    confidence_score: float,
    tone_vector: dict,
    db_path: str = "sentinel_drift.db",
) -> None:
    """Store linguistic baseline snapshot for future drift comparison."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO linguistic_baselines
        (ticker, filing_date, baseline_type, risk_score, confidence_score, tone_vector)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        ticker,
        filing_date,
        "10-q",
        risk_score,
        confidence_score,
        json.dumps(tone_vector),
    ))
    conn.commit()
    conn.close()


def _fetch_baseline_window(
    ticker: str,
    days_back: int = 30,
    db_path: str = "sentinel_drift.db",
) -> dict:
    """
    Retrieve rolling baseline (risk/confidence/tone) for the past N days.
    Returns aggregated scores or empty dict if no historical data.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    c.execute("""
        SELECT AVG(risk_score), AVG(confidence_score), tone_vector
        FROM linguistic_baselines
        WHERE ticker = ? AND created_at > ?
    """, (ticker, cutoff_date))
    row = c.fetchone()
    conn.close()

    if not row or row[0] is None:
        return {}

    avg_risk, avg_confidence = row[0], row[1]
    tone_vectors = []
    if row[2]:
        try:
            tone_vectors.append(json.loads(row[2]))
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "avg_risk_score": avg_risk,
        "avg_confidence_score": avg_confidence,
        "tone_vectors": tone_vectors,
    }


def analyze_linguistic_drift(
    ticker: str,
    current_filing_text: str,
    current_filing_date: str,
    baseline_days: int = 30,
) -> DriftSignal:
    """
    Compare current 10-Q language against rolling 30-day baseline.
    Returns drift signal with tone shifts and severity.
    """
    _init_drift_db()

    baseline = _fetch_baseline_window(ticker, days_back=baseline_days)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    if not baseline:
        baseline_context = (
            "No historical baseline available. This is the first filing in the window."
        )
        baseline_risk = 0.5
        baseline_confidence = 0.5
    else:
        baseline_risk = baseline.get("avg_risk_score", 0.5)
        baseline_confidence = baseline.get("avg_confidence_score", 0.5)
        baseline_context = (
            f"Baseline (past {baseline_days} days): "
            f"Risk score {baseline_risk:.2f}, Confidence {baseline_confidence:.2f}."
        )

    prompt = f"""You are a financial linguist analyzing SEC 10-Q filings for sentiment drift.

Ticker: {ticker}
Current Filing Date: {current_filing_date}
{baseline_context}

Current 10-Q Excerpt (first 2000 chars):
{current_filing_text[:2000]}

Analyze the following:
1. **Risk Language Delta**: Count of risk/warning words (bankruptcy, default, loss, uncertainty, etc.) 
   in current vs. baseline. Return as a float between -1.0 (less risky) and +1.0 (more risky).
2. **Confidence Delta**: Measure of certainty language (strong, expect, will, confident, etc.) 
   vs. hedge language (may, could, might). Return float -1.0 (less confident) to +1.0 (more confident).
3. **Key Shifts**: List 2-3 most notable tone changes (e.g., "increased litigation risk mentions", 
   "reduced guidance optimism").
4. **Severity**: "LOW", "MEDIUM", or "HIGH" drift detected.
5. **Recommendation**: Brief trading signal (e.g., "Watch for downside", "Neutral drift").

Respond in JSON format:
{{
  "risk_language_delta": <float>,
  "confidence_delta": <float>,
  "key_shifts": [<string>, ...],
  "severity": "<LOW|MEDIUM|HIGH>",
  "recommendation": "<string>"
}}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    try:
        analysis = json.loads(response_text)
    except json.JSONDecodeError:
        analysis = {
            "risk_language_delta": 0.0,
            "confidence_delta": 0.0,
            "key_shifts": ["Parse error"],
            "severity": "LOW",
            "recommendation": "Unable to parse LLM response",
        }

    current_risk = baseline_risk + analysis.get("risk_language_delta", 0.0)
    current_confidence = baseline_confidence + analysis.get("confidence_delta", 0.0)

    _store_baseline(
        ticker,
        current_filing_date,
        risk_score=current_risk,
        confidence_score=current_confidence,
        tone_vector={
            "risk": current_risk,
            "confidence": current_confidence,
            "key_shifts": analysis.get("key_shifts", []),
        },
    )

    baseline_start = (
        datetime.utcnow() - timedelta(days=baseline_days)
    ).date().isoformat()
    baseline_end = datetime.utcnow().date().isoformat()

    return DriftSignal(
        ticker=ticker,
        filing_date=current_filing_date,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        risk_language_delta=analysis.get("risk_language_
