"""
Linguistic Drift Detector — Sentinel Linguist Pillar

Detects significant tone and language shifts in company communications (10-Q filings,
press releases, earnings calls) by comparing current sentiment/vocabulary against a
rolling 30-day baseline. Flags anomalies that may signal management concerns, strategic
pivots, or financial stress without explicit disclosure.

Feeds into Judge's confidence scoring and anomaly alerts.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, TypedDict
from dataclasses import dataclass, asdict

import numpy as np
from anthropic import Anthropic


@dataclass
class DriftSignal:
    """Single drift detection result."""
    ticker: str
    document_type: str
    analysis_date: str
    baseline_tone: float
    current_tone: float
    tone_shift_pct: float
    vocabulary_divergence: float
    risk_keywords_spike: bool
    caution_keywords_spike: bool
    confidence: float
    summary: str
    flagged: bool


class DriftDatabase(TypedDict):
    """Schema for drift_history table."""
    ticker: str
    document_type: str
    recorded_date: str
    tone_score: float
    vocabulary_hash: str
    risk_keyword_count: int
    caution_keyword_count: int
    raw_text: str


def _init_drift_db(db_path: str) -> None:
    """Initialize drift history database if missing."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS drift_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            document_type TEXT NOT NULL,
            recorded_date TEXT NOT NULL,
            tone_score REAL,
            vocabulary_hash TEXT,
            risk_keyword_count INTEGER DEFAULT 0,
            caution_keyword_count INTEGER DEFAULT 0,
            raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE INDEX IF NOT EXISTS idx_drift_ticker_date
        ON drift_history(ticker, recorded_date DESC)
    """)
    conn.commit()
    conn.close()


def _count_keyword_occurrences(text: str, keywords: list[str]) -> int:
    """Count occurrences of keywords in text (case-insensitive)."""
    text_lower = text.lower()
    count = 0
    for kw in keywords:
        count += text_lower.count(kw.lower())
    return count


def _simple_vocabulary_hash(text: str) -> str:
    """Generate a simple hash of unique words in text for drift comparison."""
    words = text.lower().split()
    words = [w.strip('.,;:!?') for w in words if len(w) > 3]
    unique_words = sorted(set(words))
    return str(hash(tuple(unique_words[:100])))


def analyze_linguistic_drift(
    ticker: str,
    document_type: str,
    current_text: str,
    db_path: str = "sentinel.db",
    baseline_days: int = 30,
    anthropic_client: Optional[Anthropic] = None,
) -> DriftSignal:
    """
    Analyze linguistic drift in company documents vs. rolling 30-day baseline.

    Args:
        ticker: Stock ticker symbol.
        document_type: Type of document (e.g., "10-Q", "press_release", "earnings_call").
        current_text: Full text of current document to analyze.
        db_path: Path to drift history SQLite database.
        baseline_days: Rolling window for baseline (default 30 days).
        anthropic_client: Optional Claude client; if None, creates new connection.

    Returns:
        DriftSignal with tone shift, vocabulary divergence, and flagging decision.
    """
    _init_drift_db(db_path)

    risk_keywords = [
        "impairment", "writedown", "restructuring", "liquidation",
        "bankruptcy", "covenant", "default", "litigation", "investigation",
        "material weakness", "going concern", "restatement", "fraud",
    ]
    caution_keywords = [
        "challenging", "uncertain", "headwinds", "pressure", "decline",
        "weakness", "risk", "volatility", "market conditions", "cautious",
        "conservative", "difficult", "adverse", "downside",
    ]

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    cutoff_date = (datetime.now() - timedelta(days=baseline_days)).isoformat()
    c.execute(
        """
        SELECT tone_score, risk_keyword_count, caution_keyword_count
        FROM drift_history
        WHERE ticker = ? AND document_type = ? AND recorded_date > ?
        ORDER BY recorded_date DESC
        """,
        (ticker, document_type, cutoff_date),
    )
    baseline_rows = c.fetchall()

    if baseline_rows:
        baseline_tone = np.mean([row[0] for row in baseline_rows if row[0] is not None])
        baseline_risk_count = np.mean([row[1] for row in baseline_rows])
        baseline_caution_count = np.mean([row[2] for row in baseline_rows])
    else:
        baseline_tone = 0.5
        baseline_risk_count = 0
        baseline_caution_count = 0

    current_risk_count = _count_keyword_occurrences(current_text, risk_keywords)
    current_caution_count = _count_keyword_occurrences(current_text, caution_keywords)

    current_tone = (
        0.5 - (current_risk_count * 0.05) - (current_caution_count * 0.02)
    )
    current_tone = max(0.0, min(1.0, current_tone))

    tone_shift_pct = (
        ((current_tone - baseline_tone) / max(0.01, baseline_tone)) * 100
        if baseline_tone != 0
        else 0
    )

    vocab_hash_current = _simple_vocabulary_hash(current_text)
    c.execute(
        """
        SELECT vocabulary_hash FROM drift_history
        WHERE ticker = ? AND document_type = ?
        ORDER BY recorded_date DESC LIMIT 1
        """,
        (ticker, document_type),
    )
    last_vocab = c.fetchone()
    vocabulary_divergence = 0.0 if not last_vocab or last_vocab[0] == vocab_hash_current else 0.3

    risk_spike = current_risk_count > baseline_risk_count * 1.5
    caution_spike = current_caution_count > baseline_caution_count * 1.5

    c.execute(
        """
        INSERT INTO drift_history
        (ticker, document_type, recorded_date, tone_score, vocabulary_hash,
         risk_keyword_count, caution_keyword_count, raw_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticker,
            document_type,
            datetime.now().isoformat(),
            current_tone,
            vocab_hash_current,
            current_risk_count,
            current_caution_count,
            current_text[:5000],
        ),
    )
    conn.commit()
    conn.close()

    if anthropic_client is None:
        anthropic_client = Anthropic()

    flagged = abs(tone_shift_pct) > 15 or risk_spike or caution_spike
    confidence = min(0.95, 0.5 + abs(tone_shift_pct) / 100 + (0.2 if risk_spike else 0))

    prompt = f"""
    Analyze this linguistic drift signal for {ticker} ({document_type}):
    
    Baseline tone: {baseline_tone:.2f}
    Current tone: {current_tone:.2f}
    Tone shift: {tone_shift_pct:.1f}%
    Risk keywords spike: {risk
