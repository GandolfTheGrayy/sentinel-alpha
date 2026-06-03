"""Calibrator — Post-prediction market move analyzer for Sentinel.

This module compares predicted price movements against actual market outcomes,
calculating directional accuracy, magnitude error, and confidence calibration.
Used by the Judge pillar to refine heuristics and flag anomalies in the daily
post-mortem workflow.
"""

from dataclasses import dataclass
from typing import Optional
import sqlite3
from datetime import datetime, timedelta


@dataclass
class CalibrationResult:
    """Container for a single prediction vs. actual outcome comparison."""
    
    ticker: str
    prediction_date: str
    predicted_direction: str  # "UP", "DOWN", or "NEUTRAL"
    predicted_magnitude: float  # percentage change predicted
    predicted_confidence: float  # 0.0 to 1.0
    actual_direction: str  # "UP", "DOWN", or "NEUTRAL"
    actual_magnitude: float  # percentage change observed
    directional_hit: bool  # True if predicted_direction matches actual_direction
    magnitude_error: float  # absolute difference in percentage points
    confidence_calibration_error: float  # |predicted_confidence - directional_hit as 0/1|
    notes: Optional[str] = None


def compare_prediction_to_actual(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_magnitude: float,
    predicted_confidence: float,
    actual_price_start: float,
    actual_price_end: float,
) -> CalibrationResult:
    """Compare a single prediction to observed market outcome.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        prediction_date: ISO date string of prediction.
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        predicted_magnitude: Predicted percentage change.
        predicted_confidence: Confidence score 0.0–1.0.
        actual_price_start: Opening or reference price on prediction date.
        actual_price_end: Closing price after holding period.
    
    Returns:
        CalibrationResult with directional and magnitude comparisons.
    """
    if actual_price_start <= 0:
        raise ValueError(f"Invalid start price for {ticker}: {actual_price_start}")
    
    actual_magnitude = ((actual_price_end - actual_price_start) / actual_price_start) * 100
    
    if actual_magnitude > 0.5:
        actual_direction = "UP"
    elif actual_magnitude < -0.5:
        actual_direction = "DOWN"
    else:
        actual_direction = "NEUTRAL"
    
    directional_hit = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    confidence_calibration_error = abs(predicted_confidence - (1.0 if directional_hit else 0.0))
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        predicted_confidence=predicted_confidence,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        confidence_calibration_error=confidence_calibration_error,
    )


def batch_calibrate(results: list[CalibrationResult]) -> dict:
    """Aggregate calibration metrics across multiple predictions.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Dict with keys: directional_accuracy, mean_magnitude_error,
        mean_confidence_calibration_error, count.
    """
    if not results:
        return {
            "directional_accuracy": 0.0,
            "mean_magnitude_error": 0.0,
            "mean_confidence_calibration_error": 0.0,
            "count": 0,
        }
    
    directional_hits = sum(1 for r in results if r.directional_hit)
    directional_accuracy = directional_hits / len(results)
    mean_magnitude_error = sum(r.magnitude_error for r in results) / len(results)
    mean_confidence_calibration_error = sum(
        r.confidence_calibration_error for r in results
    ) / len(results)
    
    return {
        "directional_accuracy": directional_accuracy,
        "mean_magnitude_error": mean_magnitude_error,
        "mean_confidence_calibration_error": mean_confidence_calibration_error,
        "count": len(results),
    }


def persist_calibration(result: CalibrationResult, db_path: str) -> None:
    """Store a CalibrationResult in SQLite for retrospective analysis.
    
    Args:
        result: CalibrationResult to persist.
        db_path: Path to SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            predicted_magnitude REAL,
            predicted_confidence REAL,
            actual_direction TEXT,
            actual_magnitude REAL,
            directional_hit INTEGER,
            magnitude_error REAL,
            confidence_calibration_error REAL,
            notes TEXT,
            logged_at TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        INSERT INTO calibration_log (
            ticker, prediction_date, predicted_direction, predicted_magnitude,
            predicted_confidence, actual_direction, actual_magnitude,
            directional_hit, magnitude_error, confidence_calibration_error,
            notes, logged_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.ticker,
        result.prediction_date,
        result.predicted_direction,
        result.predicted_magnitude,
        result.predicted_confidence,
        result.actual_direction,
        result.actual_magnitude,
        int(result.directional_hit),
        result.magnitude_error,
        result.confidence_calibration_error,
        result.notes,
        datetime.utcnow().isoformat(),
    ))
    
    conn.commit()
    conn.close()


def load_recent_calibrations(db_path: str, days: int = 7) -> list[CalibrationResult]:
    """Load calibration results from the past N days.
    
    Args:
        db_path: Path to SQLite database file.
        days: Number of past days to retrieve.
    
    Returns:
        List of CalibrationResult objects.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    cursor.execute("""
        SELECT
            ticker, prediction_date, predicted_direction, predicted_magnitude,
            predicted_confidence, actual_direction, actual_magnitude,
            directional_hit, magnitude_error, confidence_calibration_error, notes
        FROM calibration_log
        WHERE logged_at >= ?
        ORDER BY logged_at DESC
    """, (cutoff_date,))
    
    results = []
    for row in cursor.fetchall():
        results.append(CalibrationResult(
            ticker=row[0],
            prediction_date=row[1],
            predicted_direction=row[2],
            predicted_magnitude=row[3],
            predicted_confidence=row[4],
            actual_direction=row[5],
            actual_magnitude=row[6],
            directional_hit=bool(row[7]),
            magnitude_error=row[8],
            confidence_calibration_error=row[9],
            notes=row[10],
        ))
    
    conn.close()
    return results
