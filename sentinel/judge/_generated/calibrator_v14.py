"""
Sentinel Calibrator — Post-prediction analysis module.

Compares predicted price movements (direction & magnitude) against actual market
outcomes. Produces CalibrationResult objects used by Judge's daily post-mortem
to refine heuristics and flag anomalies.

This module is called by sentinel/judge/resolver.py after market close to assess
prediction accuracy across the portfolio.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Result of comparing predicted vs. actual market movement for a single ticker."""
    
    ticker: str
    predicted_direction: str  # "UP", "DOWN", or "NEUTRAL"
    predicted_magnitude: float  # percentage change, e.g., 2.5 for +2.5%
    actual_direction: str  # "UP", "DOWN", or "NEUTRAL"
    actual_magnitude: float  # percentage change, e.g., -1.2 for -1.2%
    confidence_score: float  # 0.0 to 1.0, from original prediction
    directional_correct: bool  # True if predicted_direction matches actual_direction
    magnitude_error: float  # absolute percentage point difference
    magnitude_error_pct: float  # magnitude_error / abs(actual_magnitude) * 100, capped at 1000%
    signal_quality: str  # "STRONG", "WEAK", "NOISE", "REVERSED"
    notes: Optional[str] = None


def compare_prediction_to_actual(
    ticker: str,
    predicted_direction: str,
    predicted_magnitude: float,
    confidence_score: float,
    actual_open: float,
    actual_close: float,
) -> CalibrationResult:
    """
    Compare a single predicted movement against actual market outcome.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        predicted_direction: One of "UP", "DOWN", or "NEUTRAL".
        predicted_magnitude: Predicted percentage change (e.g., 2.5 for +2.5%).
        confidence_score: Confidence in prediction, 0.0 to 1.0.
        actual_open: Opening price on prediction day.
        actual_close: Closing price on prediction day.
    
    Returns:
        CalibrationResult with directional accuracy, magnitude error, and signal quality.
    """
    
    # Calculate actual movement
    if actual_open <= 0:
        raise ValueError(f"Invalid actual_open for {ticker}: {actual_open}")
    
    actual_magnitude = ((actual_close - actual_open) / actual_open) * 100
    
    # Determine actual direction
    if actual_magnitude > 0.1:
        actual_direction = "UP"
    elif actual_magnitude < -0.1:
        actual_direction = "DOWN"
    else:
        actual_direction = "NEUTRAL"
    
    # Check directional correctness
    directional_correct = predicted_direction == actual_direction
    
    # Calculate magnitude error (absolute difference in percentage points)
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    # Calculate magnitude error as percentage of actual (avoid division by zero)
    if abs(actual_magnitude) < 0.01:
        magnitude_error_pct = 0.0 if magnitude_error < 0.01 else 1000.0
    else:
        magnitude_error_pct = min((magnitude_error / abs(actual_magnitude)) * 100, 1000.0)
    
    # Determine signal quality
    signal_quality = _classify_signal_quality(
        directional_correct,
        confidence_score,
        magnitude_error,
        abs(actual_magnitude),
    )
    
    # Build notes
    notes = None
    if not directional_correct and confidence_score > 0.7:
        notes = f"HIGH-CONFIDENCE MISS: predicted {predicted_direction} but got {actual_direction}"
    elif directional_correct and confidence_score < 0.5:
        notes = f"LOW-CONFIDENCE HIT: predicted {predicted_direction} with {confidence_score:.1%} confidence"
    elif magnitude_error > 5.0:
        notes = f"Direction correct but magnitude off by {magnitude_error:.2f}pp"
    
    return CalibrationResult(
        ticker=ticker,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        confidence_score=confidence_score,
        directional_correct=directional_correct,
        magnitude_error=magnitude_error,
        magnitude_error_pct=magnitude_error_pct,
        signal_quality=signal_quality,
        notes=notes,
    )


def _classify_signal_quality(
    directional_correct: bool,
    confidence_score: float,
    magnitude_error: float,
    abs_actual_magnitude: float,
) -> str:
    """
    Classify signal quality into STRONG, WEAK, NOISE, or REVERSED.
    
    Args:
        directional_correct: Whether direction prediction was correct.
        confidence_score: Confidence in original prediction (0.0–1.0).
        magnitude_error: Absolute error in percentage points.
        abs_actual_magnitude: Absolute value of actual movement.
    
    Returns:
        Signal quality label.
    """
    
    if directional_correct:
        if confidence_score > 0.75 and magnitude_error < 2.0:
            return "STRONG"
        elif confidence_score > 0.6 or magnitude_error < 3.0:
            return "WEAK"
        else:
            return "NOISE"
    else:
        # Direction was wrong
        if abs_actual_magnitude > 3.0 and confidence_score > 0.7:
            return "REVERSED"  # High-confidence miss on significant move
        else:
            return "NOISE"


def batch_calibrate(
    predictions: list[dict],
) -> tuple[list[CalibrationResult], dict]:
    """
    Calibrate a batch of predictions against actual market outcomes.
    
    Args:
        predictions: List of dicts with keys:
            - ticker, predicted_direction, predicted_magnitude, confidence_score,
              actual_open, actual_close.
    
    Returns:
        Tuple of (CalibrationResult list, summary stats dict).
    """
    
    results = []
    for pred in predictions:
        result = compare_prediction_to_actual(
            ticker=pred["ticker"],
            predicted_direction=pred["predicted_direction"],
            predicted_magnitude=pred["predicted_magnitude"],
            confidence_score=pred["confidence_score"],
            actual_open=pred["actual_open"],
            actual_close=pred["actual_close"],
        )
        results.append(result)
    
    # Compute summary statistics
    if not results:
        return results, {}
    
    directional_acc = sum(1 for r in results if r.directional_correct) / len(results)
    avg_magnitude_error = sum(r.magnitude_error for r in results) / len(results)
    avg_magnitude_error_pct = sum(r.magnitude_error_pct for r in results) / len(results)
    
    strong_count = sum(1 for r in results if r.signal_quality == "STRONG")
    weak_count = sum(1 for r in results if r.signal_quality == "WEAK")
    noise_count = sum(1 for r in results if r.signal_quality == "NOISE")
    reversed_count = sum(1 for r in results if r.signal_quality == "REVERSED")
    
    avg_confidence = sum(r.confidence_score for r in results) / len(results)
    high_conf_hits = sum(
        1 for r in results if r.directional_correct and r.confidence_score > 0.7
    )
    
    summary = {
        "total_predictions": len(results),
        "directional_accuracy": round(directional_acc, 4),
        "avg_magnitude_error_pp": round(avg_magnitude_error, 2),
        "avg_magnitude_error_pct": round(avg_magnitude_error
