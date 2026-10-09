"""
Calibrator — Sentinel's prediction accuracy analyzer.

This module compares predicted price movements against actual market outcomes,
calculating directional accuracy, magnitude error, and confidence-weighted
calibration metrics. Results feed into Judge's post-mortem heuristic refinement.

Part of sentinel/judge/ — consumes PredictionResult from predictor.py and
actual market deltas to produce CalibrationResult for anomaly detection and
model tuning.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Encapsulates prediction accuracy and magnitude metrics."""
    
    ticker: str
    prediction_date: str
    predicted_direction: int  # -1 (down), 0 (hold), +1 (up)
    predicted_magnitude: float  # Percentage move, e.g., 2.5 for +2.5%
    predicted_confidence: float  # [0.0, 1.0]
    actual_direction: int  # -1, 0, +1
    actual_magnitude: float  # Percentage move
    
    directional_correct: bool
    magnitude_error: float  # Absolute percentage point difference
    confidence_weighted_error: float  # magnitude_error * (1 - predicted_confidence)
    
    signal_strength: str  # "strong", "weak", "contradicted"
    anomaly_flagged: bool


def compare_prediction_to_actual(
    ticker: str,
    prediction_date: str,
    predicted_direction: int,
    predicted_magnitude: float,
    predicted_confidence: float,
    actual_open_price: float,
    actual_close_price: float,
    actual_high_price: float,
    actual_low_price: float,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move and return calibration metrics.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        prediction_date: ISO date string when prediction was made.
        predicted_direction: -1 (down), 0 (hold), or +1 (up).
        predicted_magnitude: Expected move magnitude as percentage (e.g., 2.5).
        predicted_confidence: Confidence score [0.0, 1.0].
        actual_open_price: Opening price on prediction day.
        actual_close_price: Closing price on prediction day.
        actual_high_price: Intraday high.
        actual_low_price: Intraday low.
    
    Returns:
        CalibrationResult with directional accuracy, magnitude error, and flags.
    """
    # Calculate actual magnitude and direction
    if actual_open_price <= 0:
        raise ValueError(f"Invalid open price for {ticker}: {actual_open_price}")
    
    actual_magnitude = ((actual_close_price - actual_open_price) / actual_open_price) * 100
    
    # Determine actual direction
    if actual_close_price > actual_open_price:
        actual_direction = 1
    elif actual_close_price < actual_open_price:
        actual_direction = -1
    else:
        actual_direction = 0
    
    # Check directional correctness
    directional_correct = (predicted_direction == actual_direction) or (
        predicted_direction == 0  # "hold" prediction always counts as correct if small move
        and abs(actual_magnitude) < 0.5
    )
    
    # Calculate magnitude error (absolute percentage point difference)
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    # Confidence-weighted error: penalize high-confidence wrong calls
    confidence_weighted_error = magnitude_error * (1.0 - predicted_confidence)
    
    # Determine signal strength
    if abs(actual_magnitude) < 0.5:
        signal_strength = "weak"
    elif directional_correct and magnitude_error < 1.0:
        signal_strength = "strong"
    else:
        signal_strength = "contradicted"
    
    # Flag anomalies: high confidence + wrong direction OR huge magnitude miss
    anomaly_flagged = (
        (predicted_confidence > 0.75 and not directional_correct)
        or magnitude_error > 5.0
    )
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        predicted_confidence=predicted_confidence,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        directional_correct=directional_correct,
        magnitude_error=magnitude_error,
        confidence_weighted_error=confidence_weighted_error,
        signal_strength=signal_strength,
        anomaly_flagged=anomaly_flagged,
    )


def batch_calibrate(
    predictions_and_actuals: list[tuple[str, str, int, float, float, float, float, float, float]],
) -> list[CalibrationResult]:
    """
    Calibrate a batch of predictions against actual outcomes.
    
    Args:
        predictions_and_actuals: List of tuples (ticker, pred_date, pred_dir,
            pred_mag, pred_conf, actual_open, actual_close, actual_high, actual_low).
    
    Returns:
        List of CalibrationResult objects.
    """
    results = []
    for (ticker, pred_date, pred_dir, pred_mag, pred_conf,
         actual_open, actual_close, actual_high, actual_low) in predictions_and_actuals:
        result = compare_prediction_to_actual(
            ticker=ticker,
            prediction_date=pred_date,
            predicted_direction=pred_dir,
            predicted_magnitude=pred_mag,
            predicted_confidence=pred_conf,
            actual_open_price=actual_open,
            actual_close_price=actual_close,
            actual_high_price=actual_high,
            actual_low_price=actual_low,
        )
        results.append(result)
    return results


def compute_aggregate_metrics(
    calibration_results: list[CalibrationResult],
) -> dict[str, float]:
    """
    Compute aggregate accuracy metrics across multiple predictions.
    
    Args:
        calibration_results: List of CalibrationResult objects.
    
    Returns:
        Dictionary with keys: directional_accuracy, avg_magnitude_error,
        avg_confidence_weighted_error, anomaly_rate.
    """
    if not calibration_results:
        return {
            "directional_accuracy": 0.0,
            "avg_magnitude_error": 0.0,
            "avg_confidence_weighted_error": 0.0,
            "anomaly_rate": 0.0,
            "sample_count": 0,
        }
    
    n = len(calibration_results)
    directional_hits = sum(1 for r in calibration_results if r.directional_correct)
    total_magnitude_error = sum(r.magnitude_error for r in calibration_results)
    total_cw_error = sum(r.confidence_weighted_error for r in calibration_results)
    anomaly_count = sum(1 for r in calibration_results if r.anomaly_flagged)
    
    return {
        "directional_accuracy": directional_hits / n if n > 0 else 0.0,
        "avg_magnitude_error": total_magnitude_error / n if n > 0 else 0.0,
        "avg_confidence_weighted_error": total_cw_error / n if n > 0 else 0.0,
        "anomaly_rate": anomaly_count / n if n > 0 else 0.0,
        "sample_count": n,
    }


def filter_by_signal_strength(
    calibration_results: list[CalibrationResult],
    strength: str,
) -> list[CalibrationResult]:
    """
    Filter calibration results by signal strength ("strong", "weak", "contradicted").
    
    Args:
        calibration_results: List of CalibrationResult objects.
        strength: Target signal strength.
    
    Returns:
        Filtered list.
