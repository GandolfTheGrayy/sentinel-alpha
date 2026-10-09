"""
Calibrator — Post-prediction residual analysis for Sentinel.

This module compares predicted price movements against actual market moves,
calculating directional accuracy (did we call the right direction?) and
magnitude error (how far off was our price target?). Results feed into
the Judge's daily post-mortem and heuristic refinement loop.

Part of sentinel/judge/ — the validation and learning pillar.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class CalibrationResult:
    """Encodes comparison between predicted and actual market moves."""
    
    ticker: str
    prediction_date: datetime
    predicted_direction: str  # "UP", "DOWN", or "FLAT"
    predicted_price_target: float
    actual_price_open: float
    actual_price_close: float
    actual_direction: str  # "UP", "DOWN", or "FLAT"
    
    directional_accuracy: bool  # Did we call the right direction?
    magnitude_error_pct: float  # |predicted - actual| / actual * 100
    residual_abs: float  # |predicted_target - actual_close|
    residual_pct: float  # residual_abs / actual_close * 100
    
    confidence_score: Optional[float] = None  # Prediction confidence (0–1)
    notes: str = ""  # Free-form calibration notes


def compare_prediction_to_actual(
    ticker: str,
    prediction_date: datetime,
    predicted_direction: str,
    predicted_price_target: float,
    actual_price_open: float,
    actual_price_close: float,
    confidence_score: Optional[float] = None,
    notes: str = "",
) -> CalibrationResult:
    """
    Compare a single prediction against realized market move; return CalibrationResult.
    
    Args:
        ticker: Stock symbol (e.g., "AAPL").
        prediction_date: When the prediction was made.
        predicted_direction: "UP", "DOWN", or "FLAT".
        predicted_price_target: Target closing price.
        actual_price_open: Opening price on prediction day.
        actual_price_close: Closing price on prediction day.
        confidence_score: Optional (0–1) confidence in the prediction.
        notes: Optional free-form calibration notes.
    
    Returns:
        CalibrationResult with directional accuracy, magnitude error, residuals.
    """
    
    # Infer actual direction from open → close
    if actual_price_close > actual_price_open * 1.001:  # 0.1% threshold for FLAT
        actual_direction = "UP"
    elif actual_price_close < actual_price_open * 0.999:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"
    
    # Directional accuracy: did predicted direction match actual?
    directional_accuracy = predicted_direction == actual_direction
    
    # Magnitude error: how far off was our target vs. actual move?
    magnitude_error_pct = abs(predicted_price_target - actual_price_close) / actual_price_close * 100
    
    # Absolute residual (in dollars)
    residual_abs = abs(predicted_price_target - actual_price_close)
    
    # Percentage residual
    residual_pct = (residual_abs / actual_price_close) * 100
    
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_price_target=predicted_price_target,
        actual_price_open=actual_price_open,
        actual_price_close=actual_price_close,
        actual_direction=actual_direction,
        directional_accuracy=directional_accuracy,
        magnitude_error_pct=magnitude_error_pct,
        residual_abs=residual_abs,
        residual_pct=residual_pct,
        confidence_score=confidence_score,
        notes=notes,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate a batch of predictions; each dict must have keys:
    ticker, prediction_date, predicted_direction, predicted_price_target,
    actual_price_open, actual_price_close, [confidence_score], [notes].
    
    Returns a list of CalibrationResult objects.
    """
    results = []
    for pred in predictions:
        result = compare_prediction_to_actual(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_direction=pred["predicted_direction"],
            predicted_price_target=pred["predicted_price_target"],
            actual_price_open=pred["actual_price_open"],
            actual_price_close=pred["actual_price_close"],
            confidence_score=pred.get("confidence_score"),
            notes=pred.get("notes", ""),
        )
        results.append(result)
    return results


def accuracy_summary(results: list[CalibrationResult]) -> dict:
    """
    Aggregate calibration results into a summary dict: hit rate, avg error, counts.
    """
    if not results:
        return {
            "total_predictions": 0,
            "directional_hits": 0,
            "hit_rate_pct": 0.0,
            "avg_magnitude_error_pct": 0.0,
            "avg_residual_pct": 0.0,
            "up_predictions": 0,
            "down_predictions": 0,
            "flat_predictions": 0,
        }
    
    directional_hits = sum(1 for r in results if r.directional_accuracy)
    total = len(results)
    hit_rate = (directional_hits / total * 100) if total > 0 else 0.0
    
    avg_magnitude_error = sum(r.magnitude_error_pct for r in results) / total
    avg_residual_pct = sum(r.residual_pct for r in results) / total
    
    up_count = sum(1 for r in results if r.predicted_direction == "UP")
    down_count = sum(1 for r in results if r.predicted_direction == "DOWN")
    flat_count = sum(1 for r in results if r.predicted_direction == "FLAT")
    
    return {
        "total_predictions": total,
        "directional_hits": directional_hits,
        "hit_rate_pct": round(hit_rate, 2),
        "avg_magnitude_error_pct": round(avg_magnitude_error, 2),
        "avg_residual_pct": round(avg_residual_pct, 2),
        "up_predictions": up_count,
        "down_predictions": down_count,
        "flat_predictions": flat_count,
    }
