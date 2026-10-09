"""
Sentinel Judge — Predicted Residual vs. Actual Market Move Comparator.

This module is the core calibration engine of the Judge agent. After each
trading session, the Judge ingests the Linguist/Historian-generated predictions
and compares them against real market outcomes sourced by the Scout price
fetcher. It computes:

  - Directional accuracy  : did we call up/down correctly?
  - Magnitude error       : how far off was the predicted % move?
  - Residual distribution : systematic bias detection across a batch
  - CalibrationResult     : a structured dataclass consumed by the heuristic
                            update logger and the anomaly flagging system.

Claude (claude-sonnet-4-6) is invoked for qualitative post-mortem synthesis
when anomalous residuals are detected — never Gemini, which handles only
high-volume extraction tasks in the Scout layer.

Typical call chain:
    Scout → live prices → compare_prediction() → CalibrationResult
    batch_calibrate()   → BatchCalibrationReport
    Judge post-mortem   → Claude synthesis (see judge/postmortem.py)
"""

import os
import math
import sqlite3
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import anthropic
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """
    A single prediction emitted by the Linguist / Historian pipeline.

    Attributes:
        ticker          : stock ticker symbol (e.g. "AAPL")
        session_date    : ISO-8601 date the prediction was made for
        predicted_move  : expected price change as a decimal fraction
                          (e.g.  0.023 means +2.3 %, -0.015 means -1.5 %)
        confidence      : model confidence in [0, 1]
        signal_sources  : list of signal tags that contributed
                          (e.g. ["reddit_wsb", "sec_8k", "linguistic_drift"])
        prediction_id   : unique identifier for this prediction
    """
    ticker: str
    session_date: str                   # "YYYY-MM-DD"
    predicted_move: float               # fractional, e.g. 0.023
    confidence: float                   # [0.0, 1.0]
    signal_sources: list[str] = field(default_factory=list)
    prediction_id: str = ""


@dataclass
class CalibrationResult:
    """
    Result of comparing one prediction against the realised market move.

    Attributes:
        prediction_id       : echoes PredictionRecord.prediction_id
        ticker              : stock ticker
        session_date        : trading session date
        predicted_move      : fractional predicted price change
        actual_move         : fractional actual price change (from Scout)
        directional_correct : True if sign(predicted) == sign(actual)
        magnitude_error     : |predicted_move - actual_move|  (absolute error)
        residual            : predicted_move - actual_move     (signed error)
        confidence          : original confidence score
        confidence_penalty  : penalty when high confidence + wrong direction
        anomaly             : True when |residual| > anomaly_threshold
        anomaly_reason      : human-readable explanation of anomaly flag
        evaluated_at        : UTC timestamp of evaluation
    """
    prediction_id: str
    ticker: str
    session_date: str
    predicted_move: float
    actual_move: float
    directional_correct: bool
    magnitude_error: float
    residual: float
    confidence: float
    confidence_penalty: float
    anomaly: bool
    anomaly_reason: str
    evaluated_at: str


@dataclass
class BatchCalibrationReport:
    """
    Aggregate statistics across a batch of CalibrationResults.

    Attributes:
        report_date           : date label for this batch
        n_predictions         : total predictions evaluated
        directional_accuracy  : fraction correctly called (0–1)
        mean_absolute_error   : average |residual| across batch
        mean_residual         : average signed residual (bias detector)
        rmse                  : root-mean-squared residual
        n_anomalies           : count of flagged anomalies
        anomaly_tickers       : tickers that triggered anomalies
        confidence_calibration: Pearson correlation of confidence vs accuracy
        results               : individual CalibrationResult list
        llm_synthesis         : optional Claude narrative on the batch
    """
    report_date: str
    n_predictions: int
    directional_accuracy: float
    mean_absolute_error: float
    mean_residual: float
    rmse: float
    n_anomalies: int
    anomaly_tickers: list[str]
    confidence_calibration: float
    results: list[CalibrationResult]
    llm_synthesis: str = ""


# ---------------------------------------------------------------------------
# Core comparator
# ---------------------------------------------------------------------------

_DEFAULT_ANOMALY_THRESHOLD = 0.05   # 5 % absolute residual triggers anomaly
_DEFAULT_CONFIDENCE_PENALTY_FACTOR = 2.0  # multiplier when high-conf + wrong


def compare_prediction(
    prediction: PredictionRecord,
    actual_move: float,
    anomaly_threshold: float = _DEFAULT_ANOMALY_THRESHOLD,
    confidence_penalty_factor: float = _DEFAULT_CONFIDENCE_PENALTY_FACTOR,
) -> CalibrationResult:
    """
    Compare one prediction to its realised market move and return a CalibrationResult.

    The function is pure (no I/O) and may be called in tight loops.

    Args:
        prediction              : the prediction to evaluate
        actual_move             : realised fractional price change from Scout
        anomaly_threshold       : |residual| above which the result is flagged
        confidence_penalty_factor: scales penalty for high-confidence errors

    Returns:
        CalibrationResult with all computed metrics populated.
    """
    residual = prediction.predicted_move - actual_move
    magnitude_error = abs(residual)

    # Directional accuracy: both positive, both negative, or both exactly zero
    pred_sign = _sign(prediction.predicted_move)
    actual_sign = _sign(actual_move)
    directional_correct = pred_sign == actual_sign

    # Confidence penalty: high-confidence wrong calls are penalised more
    confidence_penalty = 0.0
    if not directional_correct:
        confidence_penalty = prediction.confidence * magnitude_error * confidence_penalty_factor

    # Anomaly detection
    anomaly = magnitude_error > anomaly_threshold
    anomaly_reason = ""
    if anomaly:
        anomaly_reason = (
            f"Residual {residual:+.4f} exceeds threshold ±{anomaly_threshold:.4f}. "
            f"Predicted {prediction.predicted_move:+.4f}, actual {actual_move:+.4f}."
        )
        if not directional_correct:
            anomaly_reason += " Direction also incorrect."

    evaluated_at = datetime.now(timezone.utc).isoformat()

    return CalibrationResult(
        prediction_id=prediction.prediction_id,
        ticker=prediction.ticker,
        session_date=prediction.session_date,
        predicted_move=prediction.predicted_move,
        actual_move=actual_move,
        directional_correct=directional_correct,
        magnitude_error=magnitude_error,
        residual=residual,
        confidence=prediction.confidence,
        confidence_penalty=confidence_penalty,
        anomaly=anomaly,
        anomaly_reason=anomaly_reason,
        evaluated_at=evaluated_at,
    )


def batch_calibrate(
    predictions: list[PredictionRecord],
    actuals: dict[str, float],
    report_date: str = "",
    anomaly_threshold: float = _DEFAULT_ANOMALY_THRESHOLD,
    confidence_penalty_factor: float = _DEFAULT_CONFIDENCE_PENALTY_FACTOR,
    synthesise_with_llm: bool = False,
) -> BatchCalibrationReport:
    """
    Evaluate a list of predictions against a ticker→actual_move map and return a BatchCal
