"""
Sentinel Calibrator — Predicted vs. Actual Market Move Comparator.

This module compares predicted price movements against realized market moves,
calculating directional accuracy, magnitude error, and confidence-weighted metrics.
Output is a CalibrationResult that feeds into Judge post-mortems and heuristic
refinement loops.

Used by Judge's daily post-mortem to measure prediction quality and identify
systematic biases in the Linguist/Historian pipeline.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """
    Container for a single prediction vs. actual market move comparison.
    
    Fields:
      ticker: Stock symbol (e.g., "AAPL").
      predicted_direction: "UP", "DOWN", or "NEUTRAL" from predictor.
      predicted_magnitude: Float, predicted percentage move (e.g., 2.5 for +2.5%).
      actual_direction: "UP", "DOWN", or "NEUTRAL" from price data.
      actual_magnitude: Float, realized percentage move (e.g., 1.8).
      confidence: Float 0–1, confidence score from Linguist.
      directional_hit: Bool, True if predicted and actual directions match.
      magnitude_error: Float, absolute error between predicted and actual (pct pts).
      confidence_weighted_error: Float, magnitude_error scaled by (1 - confidence).
      prediction_timestamp: ISO string or description of when prediction was made.
      realization_timestamp: ISO string of when price was sampled.
    """
    ticker: str
    predicted_direction: str
    predicted_magnitude: float
    actual_direction: str
    actual_magnitude: float
    confidence: float
    directional_hit: bool
    magnitude_error: float
    confidence_weighted_error: float
    prediction_timestamp: str
    realization_timestamp: str


def compute_direction(magnitude: float) -> str:
    """
    Convert a percentage move into a direction label.
    
    Args:
      magnitude: Percentage move; negative is DOWN, positive is UP, ~0 is NEUTRAL.
    
    Returns:
      "UP", "DOWN", or "NEUTRAL" (threshold: ±0.1%).
    """
    threshold = 0.1
    if magnitude > threshold:
        return "UP"
    elif magnitude < -threshold:
        return "DOWN"
    else:
        return "NEUTRAL"


def calibrate(
    ticker: str,
    predicted_magnitude: float,
    confidence: float,
    actual_magnitude: float,
    prediction_timestamp: str,
    realization_timestamp: str,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move and return a CalibrationResult.
    
    Args:
      ticker: Stock symbol.
      predicted_magnitude: Predicted percentage move.
      confidence: Linguist confidence score (0–1).
      actual_magnitude: Realized percentage move.
      prediction_timestamp: ISO string or label for prediction time.
      realization_timestamp: ISO string or label for realization time.
    
    Returns:
      CalibrationResult with directional accuracy, magnitude error, and metrics.
    """
    predicted_direction = compute_direction(predicted_magnitude)
    actual_direction = compute_direction(actual_magnitude)
    
    directional_hit = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    confidence_weighted_error = magnitude_error * (1.0 - confidence)
    
    return CalibrationResult(
        ticker=ticker,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        confidence=confidence,
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        confidence_weighted_error=confidence_weighted_error,
        prediction_timestamp=prediction_timestamp,
        realization_timestamp=realization_timestamp,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate multiple predictions in bulk.
    
    Args:
      predictions: List of dicts with keys:
        ticker, predicted_magnitude, confidence, actual_magnitude,
        prediction_timestamp, realization_timestamp.
    
    Returns:
      List of CalibrationResult objects.
    """
    results = []
    for pred in predictions:
        result = calibrate(
            ticker=pred["ticker"],
            predicted_magnitude=pred["predicted_magnitude"],
            confidence=pred["confidence"],
            actual_magnitude=pred["actual_magnitude"],
            prediction_timestamp=pred["prediction_timestamp"],
            realization_timestamp=pred["realization_timestamp"],
        )
        results.append(result)
    return results


def directional_accuracy(results: list[CalibrationResult]) -> float:
    """
    Calculate the fraction of predictions with correct direction.
    
    Args:
      results: List of CalibrationResult objects.
    
    Returns:
      Float between 0 and 1; 1.0 is perfect directional accuracy.
    """
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.directional_hit)
    return hits / len(results)


def mean_magnitude_error(results: list[CalibrationResult]) -> float:
    """
    Calculate the mean absolute percentage-point error across all predictions.
    
    Args:
      results: List of CalibrationResult objects.
    
    Returns:
      Mean magnitude error in percentage points.
    """
    if not results:
        return 0.0
    return sum(r.magnitude_error for r in results) / len(results)


def mean_confidence_weighted_error(
    results: list[CalibrationResult],
) -> float:
    """
    Calculate the mean magnitude error weighted by (1 - confidence).
    
    High-confidence misses are penalized more; low-confidence guesses are forgiven.
    
    Args:
      results: List of CalibrationResult objects.
    
    Returns:
      Mean confidence-weighted error.
    """
    if not results:
        return 0.0
    return sum(r.confidence_weighted_error for r in results) / len(results)


def root_mean_square_error(results: list[CalibrationResult]) -> float:
    """
    Calculate RMSE of magnitude errors.
    
    Args:
      results: List of CalibrationResult objects.
    
    Returns:
      RMSE in percentage points.
    """
    if not results:
        return 0.0
    squared_sum = sum(r.magnitude_error ** 2 for r in results)
    return math.sqrt(squared_sum / len(results))


def calibration_summary(results: list[CalibrationResult]) -> dict:
    """
    Generate a summary report of calibration metrics.
    
    Args:
      results: List of CalibrationResult objects.
    
    Returns:
      Dict with keys: directional_accuracy, mean_magnitude_error,
      mean_confidence_weighted_error, rmse, total_predictions.
    """
    return {
        "directional_accuracy": directional_accuracy(results),
        "mean_magnitude_error": mean_magnitude_error(results),
        "mean_confidence_weighted_error": mean_confidence_weighted_error(results),
        "rmse": root_mean_square_error(results),
        "total_predictions": len(results),
    }
