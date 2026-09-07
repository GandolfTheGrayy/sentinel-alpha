"""
Linguistic Drift Detector — Sentinel Linguist pillar.

Compares a company's current 10-Q/8-K language against a rolling 30-day baseline
of prior filings and detects significant tone shifts (pessimism, risk escalation,
confidence collapse). Uses Claude via RAG context to score drift magnitude and
flag anomalies for the Judge's confidence weighting.

Workflow:
  1. Historian retrieves last 30 days of indexed SEC filings for a ticker.
  2. Drift detector computes baseline tone fingerprint (uncertainty, risk, sentiment).
  3. Current filing is analyzed; differences are quantified.
  4. Claude reasons about magnitude and business implications.
  5. Judge uses drift_score to modulate prediction confidence.
"""

import os
import json
from typing import Optional
from dataclasses import dataclass

import anthropic


@dataclass
class ToneFingerprint:
    """Linguistic fingerprint of a filing."""
    ticker: str
    filing_date: str
    doc_type: str  # "10-Q", "8-K", etc.
    uncertainty_score: float  # 0–1, higher = more hedging/uncertainty
    risk_mentions: int  # count of risk-adjacent terms
    sentiment_polarity: float  # -1 (negative) to +1 (positive)
    confidence_markers: int  # count of high-confidence assertions
    md_a_length: int  # Management Discussion & Analysis word count


@dataclass
class DriftReport:
    """Result of drift detection analysis."""
    ticker: str
    current_date: str
    baseline_period_days: int
    baseline_fingerprint: ToneFingerprint
    current_fingerprint: ToneFingerprint
    uncertainty_delta: float  # increase in hedging language
    risk_escalation: int  # delta in risk mentions
    sentiment_shift: float  # polarity swing
    confidence_collapse: float  # 0–1, likelihood of sudden loss of conviction
    drift_score: float  # 0–1, overall magnitude of linguistic shift
    claude_rationale: str  # human-readable explanation


def extract_tone_fingerprint(
    ticker: str,
    filing_date: str,
    doc_type: str,
    full_text: str,
) -> ToneFingerprint:
    """
    Parse filing text and extract tone markers (uncertainty, risk, sentiment).
    """
    # Uncertainty markers: hedging language
    uncertainty_phrases = [
        "may", "might", "could", "uncertain", "unclear", "estimate",
        "subject to", "if", "could be", "appears to", "likely", "expected",
    ]
    uncertainty_count = sum(
        full_text.lower().count(phrase) for phrase in uncertainty_phrases
    )
    text_word_count = len(full_text.split())
    uncertainty_score = min(
        1.0, uncertainty_count / max(text_word_count / 100, 1)
    )

    # Risk markers
    risk_phrases = [
        "risk", "exposure", "loss", "decline", "adverse", "negative",
        "competition", "regulatory", "liability", "impair",
    ]
    risk_mentions = sum(
        full_text.lower().count(phrase) for phrase in risk_phrases
    )

    # Sentiment polarity (simplified)
    positive_phrases = ["growth", "increase", "strong", "improved", "success"]
    negative_phrases = ["decline", "loss", "weak", "challenge", "risk"]
    pos_count = sum(full_text.lower().count(p) for p in positive_phrases)
    neg_count = sum(full_text.lower().count(p) for p in negative_phrases)
    total_sentiment = pos_count + neg_count
    sentiment_polarity = (
        (pos_count - neg_count) / total_sentiment if total_sentiment > 0 else 0.0
    )

    # Confidence markers: strong assertions
    confidence_phrases = ["definitely", "clearly", "undoubtedly", "proven", "confirmed"]
    confidence_markers = sum(
        full_text.lower().count(p) for p in confidence_phrases
    )

    # Extract Management Discussion & Analysis length (approx)
    md_a_start = full_text.lower().find("item 7")
    if md_a_start == -1:
        md_a_start = full_text.lower().find("management")
    md_a_length = (
        len(full_text[md_a_start:].split())
        if md_a_start >= 0
        else len(full_text.split())
    )

    return ToneFingerprint(
        ticker=ticker,
        filing_date=filing_date,
        doc_type=doc_type,
        uncertainty_score=uncertainty_score,
        risk_mentions=risk_mentions,
        sentiment_polarity=sentiment_polarity,
        confidence_markers=confidence_markers,
        md_a_length=md_a_length,
    )


def compute_baseline_fingerprint(
    ticker: str,
    historical_filings: list[dict],
) -> Optional[ToneFingerprint]:
    """
    Average tone fingerprints from historical filings (30-day window).
    Returns None if insufficient data.
    """
    if not historical_filings:
        return None

    avg_uncertainty = sum(f["uncertainty_score"] for f in historical_filings) / len(
        historical_filings
    )
    avg_risk = sum(f["risk_mentions"] for f in historical_filings) / len(
        historical_filings
    )
    avg_sentiment = sum(f["sentiment_polarity"] for f in historical_filings) / len(
        historical_filings
    )
    avg_confidence = sum(f["confidence_markers"] for f in historical_filings) / len(
        historical_filings
    )
    avg_md_a_len = sum(f["md_a_length"] for f in historical_filings) / len(
        historical_filings
    )

    return ToneFingerprint(
        ticker=ticker,
        filing_date=historical_filings[0]["filing_date"],
        doc_type="BASELINE",
        uncertainty_score=avg_uncertainty,
        risk_mentions=int(avg_risk),
        sentiment_polarity=avg_sentiment,
        confidence_markers=int(avg_confidence),
        md_a_length=int(avg_md_a_len),
    )


def detect_drift(
    baseline: ToneFingerprint,
    current: ToneFingerprint,
    rag_context: str = "",
) -> DriftReport:
    """
    Compare baseline and current fingerprints; use Claude to reason about drift magnitude.
    """
    uncertainty_delta = current.uncertainty_score - baseline.uncertainty_score
    risk_escalation = current.risk_mentions - baseline.risk_mentions
    sentiment_shift = current.sentiment_polarity - baseline.sentiment_polarity
    confidence_collapse = max(
        0.0, (baseline.confidence_markers - current.confidence_markers) / max(baseline.confidence_markers, 1)
    )

    # Simple aggregate drift score
    drift_score = (
        abs(uncertainty_delta) * 0.4
        + abs(risk_escalation) / max(baseline.risk_mentions, 1) * 0.3
        + abs(sentiment_shift) * 0.2
        + confidence_collapse * 0.1
    )
    drift_score = min(1.0, drift_score)

    # Use Claude for nuanced reasoning
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""
Given a company's 10-Q/8-K language patterns, assess the magnitude and business significance of this tone shift.

BASELINE FINGERPRINT (30-day average):
- Uncertainty Score: {baseline.uncertainty_score:.3f}
- Risk Mentions: {baseline.risk_mentions}
- Sentiment Polarity: {baseline.sentiment_polarity:.3f}
- Confidence Markers: {baseline.confidence_markers}

CURRENT FILING:
- Uncertainty Score: {current.uncertainty_score:.3f}
- Risk Mentions: {current.risk_mentions}
- Sentiment
