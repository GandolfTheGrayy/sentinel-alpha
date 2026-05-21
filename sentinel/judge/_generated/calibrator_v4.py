"""
Calibrator — Predicted vs. Actual Market Move Comparator

Part of the Judge pillar's post-mortem workflow. Compares predicted price
movements (direction & magnitude) against realized market moves, computing
directional accuracy, magnitude error, and confidence-weighted calibration
scores. Returns structured CalibrationResult for daily refinement loops.

Used by: sentinel/judge/postmortem.py, sentinel/judge/resolver.py
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Holds directional and magnitude metrics for a single prediction."""
    
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    actual_direction: str     # "UP", "DOWN", "NEUTRAL"
    predicted_magnitude: float  # % change predicted
    actual_magnitude: float     # % change realized
    confidence_score: float    # 0.0–1.0 from Linguist
    
    directional_correct: bool  # True if predicted_direction matches actual_direction
    magnitude_error: float      # |predicted_magnitude - actual_magnitude|
    magnitude_error_pct: float  # magnitude_error / |actual_magnitude| (or 0 if near-zero actual)
    
    calibration_score: float   # confidence-weighted accuracy penalty
    timestamp: str              # ISO 8601 timestamp of prediction
    notes: str                  # Optional commentary


def normalize_direction(move_pct: float, threshold: float = 0.5) -> str:
    """
    Classify raw % change into discrete direction.
    
    threshold: % move required to classify as UP/DOWN; below = NEUTRAL
    Returns: "UP", "DOWN", or "NEUTRAL"
    """
    if move_pct > threshold:
        return "UP"
    elif move_pct < -threshold:
        return "DOWN"
    else:
        return "NEUTRAL"


def compute_calibration_result(
    ticker: str,
    predicted_magnitude: float,
    actual_magnitude: float,
    confidence_score: float,
    direction_threshold: float = 0.5,
    timestamp: str = "",
    notes: str = ""
) -> CalibrationResult:
    """
    Compare predicted vs. actual move; compute directional accuracy and magnitude error.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        predicted_magnitude: Predicted % price change (e.g., 2.5 for +2.5%).
        actual_magnitude: Realized % price change.
        confidence_score: Model confidence 0.0–1.0 from Linguist module.
        direction_threshold: % move threshold for UP/DOWN classification.
        timestamp: ISO 8601 string of prediction time.
        notes: Optional metadata or reasoning.
    
    Returns:
        CalibrationResult with directional match, magnitude error, and weighted score.
    """
    predicted_direction = normalize_direction(predicted_magnitude, direction_threshold)
    actual_direction = normalize_direction(actual_magnitude, direction_threshold)
    
    directional_correct = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    # Compute magnitude error as % of actual move (handle near-zero case).
    if abs(actual_magnitude) > 0.01:
        magnitude_error_pct = (magnitude_error / abs(actual_magnitude)) * 100.0
    else:
        # If actual move is tiny, penalize only if we predicted large move.
        magnitude_error_pct = magnitude_error * 100.0
    
    # Calibration score: penalize confident wrong predictions, reward confident right ones.
    if directional_correct:
        # Reward: confidence boost if direction correct.
        calibration_score = confidence_score * (1.0 - min(magnitude_error_pct / 100.0, 1.0))
    else:
        # Penalty: confidence discount if direction wrong.
        calibration_score = -confidence_score * (1.0 + magnitude_error_pct / 100.0)
    
    # Clamp calibration score to [-1.0, 1.0].
    calibration_score = max(-1.0, min(1.0, calibration_score))
    
    return CalibrationResult(
        ticker=ticker,
        predicted_direction=predicted_direction,
        actual_direction=actual_direction,
        predicted_magnitude=predicted_magnitude,
        actual_magnitude=actual_magnitude,
        confidence_score=confidence_score,
        directional_correct=directional_correct,
        magnitude_error=magnitude_error,
        magnitude_error_pct=magnitude_error_pct,
        calibration_score=calibration_score,
        timestamp=timestamp,
        notes=notes
    )


def compute_batch_calibration(
    predictions: list[dict],
    actuals: list[dict],
    direction_threshold: float = 0.5
) -> tuple[list[CalibrationResult], dict]:
    """
    Batch-compute calibration for multiple ticker predictions.
    
    Args:
        predictions: List of dicts with keys: ticker, magnitude, confidence, timestamp.
        actuals: List of dicts with keys: ticker, magnitude (realized move).
        direction_threshold: % threshold for UP/DOWN classification.
    
    Returns:
        (results, summary) where results is list of CalibrationResult,
        summary is dict with aggregate metrics (accuracy, avg_error, etc.).
    """
    results = []
    actual_by_ticker = {a["ticker"]: a["magnitude"] for a in actuals}
    
    for pred in predictions:
        ticker = pred["ticker"]
        if ticker not in actual_by_ticker:
            continue  # Skip if no actual data.
        
        result = compute_calibration_result(
            ticker=ticker,
            predicted_magnitude=pred["magnitude"],
            actual_magnitude=actual_by_ticker[ticker],
            confidence_score=pred.get("confidence", 0.5),
            direction_threshold=direction_threshold,
            timestamp=pred.get("timestamp", ""),
            notes=pred.get("notes", "")
        )
        results.append(result)
    
    # Aggregate metrics.
    if not results:
        summary = {
            "total_predictions": 0,
            "directional_accuracy": 0.0,
            "avg_magnitude_error": 0.0,
            "avg_magnitude_error_pct": 0.0,
            "avg_calibration_score": 0.0,
            "bullish_accuracy": 0.0,
            "bearish_accuracy": 0.0
        }
        return results, summary
    
    correct_count = sum(1 for r in results if r.directional_correct)
    bullish_pred = [r for r in results if r.predicted_direction == "UP"]
    bearish_pred = [r for r in results if r.predicted_direction == "DOWN"]
    
    bullish_accuracy = (
        sum(1 for r in bullish_pred if r.directional_correct) / len(bullish_pred)
        if bullish_pred else 0.0
    )
    bearish_accuracy = (
        sum(1 for r in bearish_pred if r.directional_correct) / len(bearish_pred)
        if bearish_pred else 0.0
    )
    
    summary = {
        "total_predictions": len(results),
        "directional_accuracy": correct_count / len(results),
        "avg_magnitude_error": sum(r.magnitude_error for r in results) / len(results),
        "avg_magnitude_error_pct": sum(r.magnitude_error_pct for r in results) / len(results),
        "avg_calibration_score": sum(r.calibration_score for r in results) / len(results),
        "bullish_accuracy": bullish_accuracy,
        "bearish_accuracy": bearish_accuracy
    }
    
    return results, summary


def calibration_result_to_dict(result: CalibrationResult) -> dict:
    """Convert CalibrationResult dataclass to JSON-serializable dict."""
    return {
        "ticker": result.ticker,
        "predicted_direction
