"""
Sentinel Calibrator — Post-prediction market move validator.

This module compares predicted price movements (from judge/predictor.py)
against actual realized market moves. It calculates directional accuracy
(did we call the trend right?), magnitude error (by how much?), and
confidence-weighted scores. Results feed into judge/postmortem.py for
heuristic refinement and anomaly detection.

Part of Sentinel's feedback loop: predict → realize → calibrate → improve.
"""

from dataclasses import dataclass
from typing import Optional
import sqlite3
from datetime import datetime, timedelta


@dataclass
class CalibrationResult:
    """Encodes prediction vs. actual comparison for a single ticker."""
    ticker: str
    prediction_date: str
    prediction_direction: str  # 'UP', 'DOWN', 'HOLD'
    prediction_confidence: float  # 0.0–1.0
    predicted_move_pct: float  # e.g., +2.5 or -1.3
    actual_move_pct: float  # realized move from prediction_date to settlement_date
    settlement_date: str
    directional_hit: bool  # True if predicted direction matched actual move sign
    magnitude_error_pct: float  # abs(predicted - actual)
    confidence_weighted_accuracy: float  # directional_hit * prediction_confidence
    notes: Optional[str] = None


def validate_prediction_against_actual(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_confidence: float,
    predicted_move_pct: float,
    actual_move_pct: float,
    settlement_date: str,
) -> CalibrationResult:
    """
    Compare a single prediction to realized market move; return scored result.
    """
    # Normalize direction strings.
    pred_dir = predicted_direction.upper().strip()
    if pred_dir not in ("UP", "DOWN", "HOLD"):
        pred_dir = "HOLD"

    # Compute directional accuracy: did sign match?
    # UP → actual_move_pct > 0.5% (noise margin)
    # DOWN → actual_move_pct < -0.5%
    # HOLD → -0.5% ≤ actual_move_pct ≤ 0.5%
    directional_hit = False
    if pred_dir == "UP" and actual_move_pct > 0.5:
        directional_hit = True
    elif pred_dir == "DOWN" and actual_move_pct < -0.5:
        directional_hit = True
    elif pred_dir == "HOLD" and -0.5 <= actual_move_pct <= 0.5:
        directional_hit = True

    # Magnitude error: how far off in percentage points?
    magnitude_error = abs(predicted_move_pct - actual_move_pct)

    # Confidence-weighted accuracy: did we call it right, and were we confident?
    confidence_weighted = (1.0 if directional_hit else 0.0) * max(
        0.0, min(1.0, predicted_confidence)
    )

    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        prediction_direction=pred_dir,
        prediction_confidence=max(0.0, min(1.0, predicted_confidence)),
        predicted_move_pct=predicted_move_pct,
        actual_move_pct=actual_move_pct,
        settlement_date=settlement_date,
        directional_hit=directional_hit,
        magnitude_error_pct=magnitude_error,
        confidence_weighted_accuracy=confidence_weighted,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate a batch of predictions; each dict must have keys:
      ticker, prediction_date, predicted_direction, predicted_confidence,
      predicted_move_pct, actual_move_pct, settlement_date.
    """
    results = []
    for pred in predictions:
        result = validate_prediction_against_actual(
            ticker=pred.get("ticker", "UNKNOWN"),
            prediction_date=pred.get("prediction_date", ""),
            predicted_direction=pred.get("predicted_direction", "HOLD"),
            predicted_confidence=float(pred.get("predicted_confidence", 0.5)),
            predicted_move_pct=float(pred.get("predicted_move_pct", 0.0)),
            actual_move_pct=float(pred.get("actual_move_pct", 0.0)),
            settlement_date=pred.get("settlement_date", ""),
        )
        results.append(result)
    return results


def compute_aggregate_metrics(
    calibration_results: list[CalibrationResult],
) -> dict:
    """
    Summarize calibration across a batch: directional accuracy rate, avg magnitude error, etc.
    """
    if not calibration_results:
        return {
            "total_predictions": 0,
            "directional_accuracy_pct": 0.0,
            "avg_magnitude_error_pct": 0.0,
            "avg_confidence_weighted_accuracy": 0.0,
            "high_confidence_hits": 0,
        }

    total = len(calibration_results)
    directional_hits = sum(1 for r in calibration_results if r.directional_hit)
    avg_magnitude = sum(r.magnitude_error_pct for r in calibration_results) / total
    avg_cw_accuracy = (
        sum(r.confidence_weighted_accuracy for r in calibration_results) / total
    )
    high_conf_hits = sum(
        1
        for r in calibration_results
        if r.directional_hit and r.prediction_confidence >= 0.7
    )

    return {
        "total_predictions": total,
        "directional_accuracy_pct": 100.0 * directional_hits / total,
        "avg_magnitude_error_pct": avg_magnitude,
        "avg_confidence_weighted_accuracy": avg_cw_accuracy,
        "high_confidence_hits": high_conf_hits,
    }


def persist_calibration_result(
    db_path: str,
    result: CalibrationResult,
) -> None:
    """
    Write a single calibration result to SQLite for audit trail & retrospective.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if missing.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            prediction_direction TEXT,
            prediction_confidence REAL,
            predicted_move_pct REAL,
            actual_move_pct REAL,
            settlement_date TEXT,
            directional_hit INTEGER,
            magnitude_error_pct REAL,
            confidence_weighted_accuracy REAL,
            notes TEXT,
            recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO calibration_log (
            ticker, prediction_date, prediction_direction, prediction_confidence,
            predicted_move_pct, actual_move_pct, settlement_date, directional_hit,
            magnitude_error_pct, confidence_weighted_accuracy, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.ticker,
            result.prediction_date,
            result.prediction_direction,
            result.prediction_confidence,
            result.predicted_move_pct,
            result.actual_move_pct,
            result.settlement_date,
            1 if result.directional_hit else 0,
            result.magnitude_error_pct,
            result.confidence_weighted_accuracy,
            result.notes,
        ),
    )

    conn.commit()
    conn.close()


def load_recent_calibrations(
    db_path: str,
    days_back: int = 30,
)
