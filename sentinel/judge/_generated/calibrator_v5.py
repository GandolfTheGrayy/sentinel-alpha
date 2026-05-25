"""
Sentinel Calibrator — Post-prediction market move analyzer.

This module compares predicted price movements (from Judge) against actual
market outcomes, computing directional accuracy, magnitude error, and
confidence-weighted performance metrics. Results feed into daily post-mortems
and heuristic refinement loops.

Part of sentinel/judge/ pillar: transforms raw predictions + actuals into
CalibrationResult objects for trend analysis and anomaly detection.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import sqlite3
from datetime import datetime


@dataclass
class CalibrationResult:
    """
    Post-prediction market move comparison result.

    Attributes:
        ticker: Stock symbol analyzed.
        prediction_date: Date prediction was made.
        predicted_direction: 'UP', 'DOWN', or 'NEUTRAL' from predictor.
        predicted_magnitude: Float percentage move predicted (e.g., 2.5 for +2.5%).
        actual_direction: 'UP', 'DOWN', or 'NEUTRAL' observed in market.
        actual_magnitude: Float percentage move observed (e.g., -1.2 for -1.2%).
        directional_correct: True if predicted_direction matches actual_direction.
        magnitude_error: Absolute difference in percentage points.
        confidence_score: Predictor's confidence [0.0, 1.0] from Judge.
        confidence_weighted_error: magnitude_error * (1 - confidence_score).
        holding_period_hours: Hours between prediction and close; default 24.
    """
    ticker: str
    prediction_date: str
    predicted_direction: str
    predicted_magnitude: float
    actual_direction: str
    actual_magnitude: float
    directional_correct: bool
    magnitude_error: float
    confidence_score: float
    confidence_weighted_error: float
    holding_period_hours: int = 24


def infer_direction(magnitude: float) -> str:
    """Infer direction label from signed magnitude percentage."""
    if magnitude > 0.05:
        return "UP"
    elif magnitude < -0.05:
        return "DOWN"
    return "NEUTRAL"


def calculate_calibration(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_magnitude: float,
    actual_magnitude: float,
    confidence_score: float,
    holding_period_hours: int = 24,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move; return CalibrationResult.

    Args:
        ticker: Stock symbol (e.g., 'NVDA').
        prediction_date: ISO date string of prediction (e.g., '2025-01-15').
        predicted_direction: 'UP', 'DOWN', or 'NEUTRAL'.
        predicted_magnitude: Predicted % move as float (e.g., 2.5).
        actual_magnitude: Observed % move as float (e.g., -1.2).
        confidence_score: Predictor confidence in [0.0, 1.0].
        holding_period_hours: Hours held; default 24 (next market close).

    Returns:
        CalibrationResult with directional accuracy, magnitude error, and
        confidence-weighted metrics.
    """
    actual_direction = infer_direction(actual_magnitude)
    directional_correct = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    confidence_weighted_error = magnitude_error * (1.0 - confidence_score)

    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        directional_correct=directional_correct,
        magnitude_error=magnitude_error,
        confidence_score=confidence_score,
        confidence_weighted_error=confidence_weighted_error,
        holding_period_hours=holding_period_hours,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate multiple prediction-vs-actual pairs in bulk.

    Args:
        predictions: List of dicts, each with keys:
            {ticker, prediction_date, predicted_direction, predicted_magnitude,
             actual_magnitude, confidence_score, holding_period_hours?}

    Returns:
        List of CalibrationResult objects, one per prediction.
    """
    results = []
    for pred in predictions:
        result = calculate_calibration(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_direction=pred["predicted_direction"],
            predicted_magnitude=pred["predicted_magnitude"],
            actual_magnitude=pred["actual_magnitude"],
            confidence_score=pred["confidence_score"],
            holding_period_hours=pred.get("holding_period_hours", 24),
        )
        results.append(result)
    return results


def store_calibration(
    db_path: str,
    result: CalibrationResult,
) -> None:
    """
    Persist CalibrationResult to SQLite for post-mortem analysis.

    Args:
        db_path: Path to Sentinel SQLite database.
        result: CalibrationResult to store.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_magnitude REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            actual_magnitude REAL NOT NULL,
            directional_correct INTEGER NOT NULL,
            magnitude_error REAL NOT NULL,
            confidence_score REAL NOT NULL,
            confidence_weighted_error REAL NOT NULL,
            holding_period_hours INTEGER NOT NULL,
            stored_at TEXT NOT NULL,
            UNIQUE(ticker, prediction_date)
        )
        """
    )

    cursor.execute(
        """
        INSERT OR REPLACE INTO calibration_results
        (ticker, prediction_date, predicted_direction, predicted_magnitude,
         actual_direction, actual_magnitude, directional_correct,
         magnitude_error, confidence_score, confidence_weighted_error,
         holding_period_hours, stored_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.ticker,
            result.prediction_date,
            result.predicted_direction,
            result.predicted_magnitude,
            result.actual_direction,
            result.actual_magnitude,
            int(result.directional_correct),
            result.magnitude_error,
            result.confidence_score,
            result.confidence_weighted_error,
            result.holding_period_hours,
            datetime.utcnow().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def load_calibration(
    db_path: str,
    ticker: Optional[str] = None,
    limit: int = 100,
) -> list[CalibrationResult]:
    """
    Load stored CalibrationResult rows from SQLite, optionally filtered by ticker.

    Args:
        db_path: Path to Sentinel SQLite database.
        ticker: If provided, filter to this ticker only.
        limit: Max rows to return; default 100.

    Returns:
        List of CalibrationResult objects reconstructed from database.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if ticker:
        cursor.execute(
            """
            SELECT * FROM calibration_results
            WHERE ticker = ?
            ORDER BY prediction_date DESC
            LIMIT ?
            """,
            (ticker, limit),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM calibration_results
            ORDER BY prediction_date DESC
            LIMIT ?
            """,
            (limit,),
        )

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append(
