"""
Sentinel Judge — Calibration Comparator Module.

Compares predicted market moves (from Linguist sentiment scoring) against
actual market moves (from Scout price data) to compute directional accuracy,
magnitude error, and confidence-weighted residuals. Outputs CalibrationResult
for post-mortem analysis and heuristic refinement feedback loops.

Part of the Judge Agent that validates Sentinel's prediction quality daily.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """
    Holds comparison metrics between predicted and actual market moves.
    
    Attributes:
        ticker: Stock symbol being evaluated.
        prediction_date: ISO date string when prediction was made.
        predicted_direction: 1 (up), -1 (down), 0 (neutral).
        predicted_magnitude: Float percentage move predicted (e.g., 2.5 for +2.5%).
        predicted_confidence: Float in [0, 1] — how certain the Linguist was.
        actual_direction: 1 (up), -1 (down), 0 (neutral) from actual price data.
        actual_magnitude: Float percentage move that actually occurred.
        directional_accuracy: Boolean — did predicted_direction match actual_direction?
        magnitude_error: Float absolute difference between predicted and actual magnitude.
        confidence_weighted_error: Float — magnitude_error scaled by (1 - predicted_confidence).
        residual_pct: Float — signed error: (predicted_magnitude - actual_magnitude) / abs(actual_magnitude + 0.001).
        calibration_score: Float in [0, 1] — composite quality metric for this prediction.
    """
    ticker: str
    prediction_date: str
    predicted_direction: int
    predicted_magnitude: float
    predicted_confidence: float
    actual_direction: int
    actual_magnitude: float
    directional_accuracy: bool
    magnitude_error: float
    confidence_weighted_error: float
    residual_pct: float
    calibration_score: float


def compare_predicted_vs_actual(
    ticker: str,
    prediction_date: str,
    predicted_direction: int,
    predicted_magnitude: float,
    predicted_confidence: float,
    actual_direction: int,
    actual_magnitude: float,
) -> CalibrationResult:
    """
    Compare a single prediction against actual market move and return calibration metrics.
    
    Args:
        ticker: Stock symbol (e.g. "AAPL").
        prediction_date: ISO date string (e.g. "2025-01-15").
        predicted_direction: -1 (down), 0 (neutral), 1 (up).
        predicted_magnitude: Predicted percentage move (positive or negative).
        predicted_confidence: Confidence score in [0, 1].
        actual_direction: -1 (down), 0 (neutral), 1 (up) from actual data.
        actual_magnitude: Actual percentage move that occurred.
    
    Returns:
        CalibrationResult with all comparison metrics populated.
    """
    # Validate inputs.
    if not isinstance(predicted_direction, int) or predicted_direction not in [-1, 0, 1]:
        raise ValueError(f"predicted_direction must be -1, 0, or 1; got {predicted_direction}")
    if not isinstance(actual_direction, int) or actual_direction not in [-1, 0, 1]:
        raise ValueError(f"actual_direction must be -1, 0, or 1; got {actual_direction}")
    if not 0 <= predicted_confidence <= 1:
        raise ValueError(f"predicted_confidence must be in [0, 1]; got {predicted_confidence}")
    
    # Directional accuracy: did the sign match?
    directional_accuracy = (predicted_direction == actual_direction)
    
    # Magnitude error: absolute difference between predicted and actual.
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    # Confidence-weighted error: penalizes high-confidence misses more.
    confidence_weighted_error = magnitude_error * (1.0 - predicted_confidence)
    
    # Residual percentage: signed relative error.
    # Use small epsilon (0.001) to avoid division by zero on tiny moves.
    denom = abs(actual_magnitude) + 0.001
    residual_pct = (predicted_magnitude - actual_magnitude) / denom
    
    # Calibration score: composite metric in [0, 1].
    # Higher score = better prediction.
    # Components:
    #   1. Directional hit bonus: +0.5 if direction matches.
    #   2. Magnitude accuracy: decay based on magnitude_error.
    #   3. Confidence calibration: bonus if high confidence + low error, penalty if high confidence + high error.
    
    directional_bonus = 0.5 if directional_accuracy else 0.0
    
    # Magnitude accuracy: gaussian-like decay. Full 0.3 points if error < 0.5%, decays to ~0 at error > 5%.
    magnitude_score = 0.3 * math.exp(-0.5 * (magnitude_error ** 2) / (1.0 ** 2))
    
    # Confidence calibration: if high confidence, we want low error; if low confidence, more tolerance.
    # Award up to 0.2 points for well-calibrated confidence.
    confidence_calibration = 0.2 * (
        predicted_confidence * math.exp(-magnitude_error / 2.0)
    )
    
    calibration_score = min(1.0, directional_bonus + magnitude_score + confidence_calibration)
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        predicted_confidence=predicted_confidence,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        directional_accuracy=directional_accuracy,
        magnitude_error=magnitude_error,
        confidence_weighted_error=confidence_weighted_error,
        residual_pct=residual_pct,
        calibration_score=calibration_score,
    )


def batch_compare(predictions: list[dict]) -> list[CalibrationResult]:
    """
    Compare multiple predictions in batch and return list of CalibrationResults.
    
    Args:
        predictions: List of dicts with keys:
            ticker, prediction_date, predicted_direction, predicted_magnitude,
            predicted_confidence, actual_direction, actual_magnitude.
    
    Returns:
        List of CalibrationResult objects, one per input prediction.
    """
    results = []
    for pred in predictions:
        result = compare_predicted_vs_actual(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_direction=pred["predicted_direction"],
            predicted_magnitude=pred["predicted_magnitude"],
            predicted_confidence=pred["predicted_confidence"],
            actual_direction=pred["actual_direction"],
            actual_magnitude=pred["actual_magnitude"],
        )
        results.append(result)
    return results


def aggregate_calibration(results: list[CalibrationResult]) -> dict:
    """
    Aggregate calibration metrics across multiple predictions for summary reporting.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Dict with keys: directional_accuracy_pct, mean_magnitude_error,
        mean_confidence_weighted_error, mean_calibration_score, sample_size.
    """
    if not results:
        return {
            "directional_accuracy_pct": 0.0,
            "mean_magnitude_error": 0.0,
            "mean_confidence_weighted_error": 0.0,
            "mean_calibration_score": 0.0,
            "sample_size": 0,
        }
    
    n = len(results)
    directional_hits = sum(1 for r in results if r.directional_accuracy)
    directional_accuracy_pct = 100.0 * directional_hits / n if n > 0 else 0.0
    
    mean_magnitude_error = sum(r.magnitude_error for r in results
