"""
Calibrator — Predicted Residual vs. Actual Market Move comparator.

Part of the Sentinel Judge pillar. After predictions are made and market
close data arrives, this module compares predicted direction/magnitude against
actual price movement, computing directional accuracy and magnitude error.
Results feed into the daily post-mortem and heuristic refinement pipeline.

Exports CalibrationResult dataclass and compare() function for use by
sentinel/judge/postmortem.py and the daily build harness.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Container for a single predicted-vs-actual comparison."""
    ticker: str
    prediction_date: str
    predicted_direction: str  # "UP", "DOWN", or "NEUTRAL"
    predicted_magnitude: float  # e.g., 2.5 for +2.5%
    actual_magnitude: float  # observed % change at market close
    directional_hit: bool  # True if predicted_direction matched actual sign
    magnitude_error: float  # |predicted_magnitude - actual_magnitude|
    confidence_score: float  # model's stated confidence [0, 1]
    notes: str = ""


def compare(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_magnitude: float,
    actual_magnitude: float,
    confidence_score: float,
    notes: str = "",
) -> CalibrationResult:
    """
    Compare a single prediction against realized market move.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        prediction_date: ISO date string of prediction.
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        predicted_magnitude: Predicted % change (can be negative).
        actual_magnitude: Realized % change at market close.
        confidence_score: Model confidence in [0, 1].
        notes: Optional commentary on discrepancy.
    
    Returns:
        CalibrationResult with directional_hit and magnitude_error computed.
    """
    # Determine if prediction direction matched sign of actual move.
    actual_direction = _magnitude_to_direction(actual_magnitude)
    directional_hit = (predicted_direction == actual_direction) if predicted_direction != "NEUTRAL" else True
    
    # Compute magnitude error.
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_magnitude=actual_magnitude,
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        confidence_score=confidence_score,
        notes=notes,
    )


def batch_compare(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Compare multiple predictions against realized moves in bulk.
    
    Args:
        predictions: List of dicts with keys:
            ticker, prediction_date, predicted_direction, predicted_magnitude,
            actual_magnitude, confidence_score, (optional) notes.
    
    Returns:
        List of CalibrationResult objects.
    """
    results = []
    for pred in predictions:
        result = compare(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_direction=pred["predicted_direction"],
            predicted_magnitude=pred["predicted_magnitude"],
            actual_magnitude=pred["actual_magnitude"],
            confidence_score=pred["confidence_score"],
            notes=pred.get("notes", ""),
        )
        results.append(result)
    return results


def accuracy_summary(results: list[CalibrationResult]) -> dict:
    """
    Aggregate directional accuracy and magnitude error across multiple results.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Dict with keys: directional_accuracy (%), mean_magnitude_error,
        median_magnitude_error, count, high_confidence_accuracy (%).
    """
    if not results:
        return {
            "directional_accuracy": 0.0,
            "mean_magnitude_error": 0.0,
            "median_magnitude_error": 0.0,
            "count": 0,
            "high_confidence_accuracy": 0.0,
        }
    
    directional_hits = sum(1 for r in results if r.directional_hit)
    directional_accuracy = (directional_hits / len(results)) * 100
    
    magnitude_errors = [r.magnitude_error for r in results]
    mean_error = sum(magnitude_errors) / len(magnitude_errors)
    median_error = sorted(magnitude_errors)[len(magnitude_errors) // 2]
    
    # High confidence: confidence_score >= 0.7
    high_conf_results = [r for r in results if r.confidence_score >= 0.7]
    high_conf_accuracy = 0.0
    if high_conf_results:
        high_conf_hits = sum(1 for r in high_conf_results if r.directional_hit)
        high_conf_accuracy = (high_conf_hits / len(high_conf_results)) * 100
    
    return {
        "directional_accuracy": round(directional_accuracy, 2),
        "mean_magnitude_error": round(mean_error, 3),
        "median_magnitude_error": round(median_error, 3),
        "count": len(results),
        "high_confidence_accuracy": round(high_conf_accuracy, 2),
    }


def _magnitude_to_direction(magnitude: float) -> str:
    """
    Convert a magnitude (% change) to direction string.
    
    Args:
        magnitude: Realized % change.
    
    Returns:
        "UP" if magnitude > 0.1%, "DOWN" if magnitude < -0.1%, else "NEUTRAL".
    """
    if magnitude > 0.1:
        return "UP"
    elif magnitude < -0.1:
        return "DOWN"
    else:
        return "NEUTRAL"
