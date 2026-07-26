"""
Calibrator — Post-prediction accuracy and residual analysis for Sentinel.

This module compares predicted vs. actual market moves, computing directional
accuracy, magnitude error, and confidence-weighted residuals. Used by Judge
to refine heuristics after market close and flag anomalies in prediction drift.

Output: CalibrationResult objects ingested by the daily post-mortem renderer
and weekly retrospective generator.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
from datetime import datetime


@dataclass
class CalibrationResult:
    """Holds comparative accuracy metrics for a single prediction vs. reality."""
    
    ticker: str
    prediction_date: str
    predicted_direction: str  # "UP", "DOWN", "HOLD"
    predicted_magnitude: float  # % change predicted
    actual_direction: str  # "UP", "DOWN", "HOLD"
    actual_magnitude: float  # % change observed
    confidence_score: float  # 0.0–1.0, from Linguist
    
    # Computed metrics
    directional_hit: bool  # True if predicted_direction == actual_direction
    magnitude_error: float  # |predicted_magnitude - actual_magnitude|
    signed_error: float  # predicted_magnitude - actual_magnitude (signed residual)
    confidence_weighted_error: float  # magnitude_error * (1 - confidence_score)
    
    # Flags
    anomaly_flag: bool  # True if error > 2 std devs or direction flip despite high confidence
    notes: Optional[str] = None


def calibrate_prediction(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_magnitude: float,
    confidence_score: float,
    actual_direction: str,
    actual_magnitude: float,
    historical_errors: Optional[list[float]] = None,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move and compute residual metrics.
    
    Args:
        ticker: Stock symbol.
        prediction_date: ISO date string of prediction.
        predicted_direction: "UP", "DOWN", or "HOLD".
        predicted_magnitude: Predicted % change (e.g., 2.5 for +2.5%).
        confidence_score: Certainty from Linguist (0.0–1.0).
        actual_direction: Observed direction.
        actual_magnitude: Observed % change.
        historical_errors: List of past magnitude errors for anomaly detection.
    
    Returns:
        CalibrationResult with directional accuracy, magnitude error, and flags.
    """
    
    # Directional accuracy
    directional_hit = predicted_direction == actual_direction
    
    # Magnitude metrics
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    signed_error = predicted_magnitude - actual_magnitude
    confidence_weighted_error = magnitude_error * (1.0 - confidence_score)
    
    # Anomaly detection: flag if error > 2σ or high-confidence wrong direction
    anomaly_flag = False
    notes = None
    
    if historical_errors and len(historical_errors) > 1:
        mean_error = np.mean(historical_errors)
        std_error = np.std(historical_errors)
        if std_error > 0 and magnitude_error > mean_error + 2 * std_error:
            anomaly_flag = True
            notes = f"Error {magnitude_error:.2f}% exceeds mean+2σ ({mean_error:.2f} + {2*std_error:.2f})"
    
    if confidence_score >= 0.75 and not directional_hit:
        anomaly_flag = True
        if not notes:
            notes = f"High confidence ({confidence_score:.2f}) but wrong direction"
        else:
            notes += f"; high confidence direction miss"
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        confidence_score=confidence_score,
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        signed_error=signed_error,
        confidence_weighted_error=confidence_weighted_error,
        anomaly_flag=anomaly_flag,
        notes=notes,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate a batch of predictions against actuals.
    
    Args:
        predictions: List of dicts with keys: ticker, prediction_date,
                     predicted_direction, predicted_magnitude, confidence_score,
                     actual_direction, actual_magnitude, historical_errors (opt).
    
    Returns:
        List of CalibrationResult objects.
    """
    results = []
    for pred in predictions:
        result = calibrate_prediction(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_direction=pred["predicted_direction"],
            predicted_magnitude=pred["predicted_magnitude"],
            confidence_score=pred["confidence_score"],
            actual_direction=pred["actual_direction"],
            actual_magnitude=pred["actual_magnitude"],
            historical_errors=pred.get("historical_errors"),
        )
        results.append(result)
    return results


def compute_portfolio_stats(
    calibrations: list[CalibrationResult],
) -> dict:
    """
    Aggregate calibration metrics across a portfolio of predictions.
    
    Args:
        calibrations: List of CalibrationResult objects.
    
    Returns:
        Dict with keys: hit_rate, mean_magnitude_error, mean_signed_error,
                        mean_confidence_weighted_error, anomaly_count, total_count.
    """
    if not calibrations:
        return {
            "hit_rate": 0.0,
            "mean_magnitude_error": 0.0,
            "mean_signed_error": 0.0,
            "mean_confidence_weighted_error": 0.0,
            "anomaly_count": 0,
            "total_count": 0,
        }
    
    total = len(calibrations)
    hits = sum(1 for c in calibrations if c.directional_hit)
    anomalies = sum(1 for c in calibrations if c.anomaly_flag)
    
    magnitude_errors = [c.magnitude_error for c in calibrations]
    signed_errors = [c.signed_error for c in calibrations]
    weighted_errors = [c.confidence_weighted_error for c in calibrations]
    
    return {
        "hit_rate": hits / total if total > 0 else 0.0,
        "mean_magnitude_error": float(np.mean(magnitude_errors)),
        "mean_signed_error": float(np.mean(signed_errors)),
        "mean_confidence_weighted_error": float(np.mean(weighted_errors)),
        "anomaly_count": anomalies,
        "total_count": total,
    }


def filter_by_anomaly(
    calibrations: list[CalibrationResult],
    flag_only: bool = True,
) -> list[CalibrationResult]:
    """
    Filter calibration results for anomalies.
    
    Args:
        calibrations: List of CalibrationResult objects.
        flag_only: If True, return only flagged anomalies; else return non-anomalies.
    
    Returns:
        Filtered list.
    """
    if flag_only:
        return [c for c in calibrations if c.anomaly_flag]
    else:
        return [c for c in calibrations if not c.anomaly_flag]
