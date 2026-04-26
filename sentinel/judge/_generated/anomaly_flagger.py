"""
Sentinel Judge — Anomaly Flagging System

This module is responsible for detecting when actual market moves deviate
significantly from what the Sentinel engine predicted. Specifically, it flags
cases where the absolute actual market move exceeds 2x the absolute predicted
residual, which indicates that the engine's model missed something material —
a black swan event, a surprise earnings beat/miss, regulatory shock, or a
failure in the upstream sentiment pipeline.

Flagged anomalies are encapsulated in AnomalyAlert dataclasses and persisted
to a SQLite database for the Judge agent's post-mortem pipeline. Claude is
used to produce a brief natural-language diagnosis of each anomaly, drawing on
the surrounding signal context so that heuristic refinements can be logged and
fed back into the Historian and Linguist agents.

Fits into the broader Sentinel architecture as the early-warning layer of the
Judge agent: anomalies it surfaces inform the heuristic update logger and the
post-mortem report generator.
"""

import os
import sqlite3
import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional
import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ANOMALY_MULTIPLIER_THRESHOLD: float = 2.0  # flag when |actual| > N * |predicted|
DEFAULT_DB_PATH: str = "sentinel_judge.db"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """A single prediction produced by Sentinel for one ticker on one date."""

    ticker: str
    prediction_date: str          # ISO-8601 date, e.g. "2025-07-14"
    predicted_residual: float     # expected % move (signed), e.g. +0.023 = +2.3 %
    actual_move: float            # realised % move (signed)
    confidence_score: float       # 0.0–1.0 from Historian weighting
    signal_context: dict          # raw signals that drove the prediction
    model_version: str = "0.1.0"


@dataclass
class AnomalyAlert:
    """
    Represents a detected anomaly where the actual market move exceeded
    2x the predicted residual in absolute terms.
    """

    alert_id: str                          # unique identifier
    ticker: str
    prediction_date: str                   # ISO-8601 date
    predicted_residual: float              # original prediction
    actual_move: float                     # what actually happened
    anomaly_ratio: float                   # |actual| / |predicted|
    confidence_score: float                # confidence at prediction time
    signal_context: dict                   # upstream signals
    severity: str                          # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    llm_diagnosis: Optional[str] = None    # Claude's natural-language analysis
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_version: str = "0.1.0"


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_severity(anomaly_ratio: float) -> str:
    """Return a severity label based on how far the anomaly ratio exceeds the threshold."""
    if anomaly_ratio < 3.0:
        return "LOW"
    elif anomaly_ratio < 5.0:
        return "MEDIUM"
    elif anomaly_ratio < 10.0:
        return "HIGH"
    else:
        return "CRITICAL"


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def is_anomalous(
    predicted_residual: float,
    actual_move: float,
    threshold_multiplier: float = ANOMALY_MULTIPLIER_THRESHOLD,
) -> tuple[bool, float]:
    """
    Return (True, ratio) if |actual_move| exceeds threshold_multiplier * |predicted_residual|.

    Edge case: if |predicted_residual| is effectively zero we use a floor of
    0.001 (0.1 %) to avoid division-by-zero while still flagging large moves.
    """
    predicted_abs = max(abs(predicted_residual), 0.001)
    actual_abs = abs(actual_move)
    ratio = actual_abs / predicted_abs
    return ratio >= threshold_multiplier, ratio


def generate_alert_id(ticker: str, prediction_date: str) -> str:
    """Produce a deterministic, human-readable alert identifier."""
    sanitised_date = prediction_date.replace("-", "")
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    return f"ALERT-{ticker.upper()}-{sanitised_date}-{ts}"


# ---------------------------------------------------------------------------
# LLM diagnosis (Claude — reasoning only)
# ---------------------------------------------------------------------------

def _build_diagnosis_prompt(record: PredictionRecord, anomaly_ratio: float) -> str:
    """Construct the Claude prompt for anomaly diagnosis."""
    context_str = json.dumps(record.signal_context, indent=2)
    return (
        f"You are the Judge agent of the Sentinel financial intelligence system.\n\n"
        f"An anomaly has been detected for ticker **{record.ticker}** on "
        f"{record.prediction_date}.\n\n"
        f"**Prediction summary**\n"
        f"- Predicted residual (expected % move): {record.predicted_residual:+.4f} "
        f"({record.predicted_residual * 100:+.2f}%)\n"
        f"- Actual market move:                   {record.actual_move:+.4f} "
        f"({record.actual_move * 100:+.2f}%)\n"
        f"- Anomaly ratio (|actual| / |predicted|): {anomaly_ratio:.2f}x\n"
        f"- Model confidence at prediction time:    {record.confidence_score:.3f}\n\n"
        f"**Upstream signal context**\n"
        f"```json\n{context_str}\n```\n\n"
        f"Please provide a concise diagnosis (3–5 sentences) covering:\n"
        f"1. The most likely cause(s) of the prediction error.\n"
        f"2. Which upstream signals may have been misleading or absent.\n"
        f"3. A specific recommendation for heuristic refinement to reduce "
        f"similar errors in future.\n\n"
        f"Be direct and technical. Do not repeat the numbers already stated above."
    )


def request_llm_diagnosis(
    record: PredictionRecord,
    anomaly_ratio: float,
    api_key: Optional[str] = None,
) -> str:
    """
    Call Claude (claude-sonnet-4-6) to produce a natural-language anomaly diagnosis.
    Returns an empty string on failure so the alert is still persisted.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping LLM diagnosis.")
        return ""

    client = anthropic.Anthropic(api_key=key)
    prompt = _build_diagnosis_prompt(record, anomaly_ratio)

    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except anthropic.APIError as exc:
        logger.error("Claude API error during anomaly diagnosis: %s", exc)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error requesting LLM diagnosis: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Alert construction
# ---------------------------------------------------------------------------

def build_anomaly_alert(
    record: PredictionRecord,
    anomaly_ratio: float,
    use_llm: bool = True,
    api_key: Optional[str] = None,
) -> AnomalyAlert:
    """
    Construct a fully-populated A
