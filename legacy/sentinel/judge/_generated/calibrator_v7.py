"""
Calibrator module for Sentinel Sentiment Engine.

Compares predicted price movements against actual market outcomes,
computing directional accuracy, magnitude error, and confidence-weighted
residuals. Produces CalibrationResult objects for post-mortem analysis
and heuristic refinement in the Judge pillar.

Used by judge/postmortem.py to assess prediction quality and flag
anomalies for daily retrospectives.
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Encapsulates comparison between predicted and actual market moves."""
    
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    predicted_magnitude: float  # percent change, e.g., 2.5
    predicted_confidence: float  # 0.0 to 1.0
    actual_direction: str  # "UP", "DOWN", "NEUTRAL"
    actual_magnitude: float  # percent change, e.g., 1.8
    directional_hit: bool  # True if predicted_direction matches actual_direction
    magnitude_error: float  # absolute difference in magnitude
    confidence_weighted_error: float  # magnitude_error * (1 - predicted_confidence)
    anomaly_flag: bool  # True if error exceeds 2-sigma threshold
    notes: str  # Human-readable summary


def direction_from_magnitude(magnitude: float, threshold: float = 0.5) -> str:
    """
    Convert a percent magnitude to directional label.
    
    Args:
        magnitude: Signed percent change (positive = up, negative = down).
        threshold: Magnitude below which is classified as NEUTRAL.
    
    Returns:
        "UP", "DOWN", or "NEUTRAL".
    """
    if abs(magnitude) < threshold:
        return "NEUTRAL"
    return "UP" if magnitude > 0 else "DOWN"


def compare_predictions(
    ticker: str,
    predicted_magnitude: float,
    predicted_confidence: float,
    actual_magnitude: float,
    direction_threshold: float = 0.5,
    anomaly_sigma: float = 2.0,
    historical_stdev: Optional[float] = None,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market moves and compute calibration metrics.
    
    Args:
        ticker: Stock ticker symbol.
        predicted_magnitude: Predicted percent change (e.g., 2.5 for +2.5%).
        predicted_confidence: Confidence score 0.0–1.0.
        actual_magnitude: Actual observed percent change.
        direction_threshold: Magnitude below which is NEUTRAL (default 0.5%).
        anomaly_sigma: Std-dev multiplier for anomaly detection (default 2.0).
        historical_stdev: Optional historical error std-dev for anomaly context.
    
    Returns:
        CalibrationResult with directional accuracy, magnitude error, and flags.
    """
    predicted_direction = direction_from_magnitude(predicted_magnitude, direction_threshold)
    actual_direction = direction_from_magnitude(actual_magnitude, direction_threshold)
    
    directional_hit = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    confidence_weighted_error = magnitude_error * (1.0 - predicted_confidence)
    
    # Anomaly detection: flag if error is unusually large
    anomaly_flag = False
    if historical_stdev is not None and historical_stdev > 0:
        z_score = magnitude_error / historical_stdev
        anomaly_flag = z_score > anomaly_sigma
    elif magnitude_error > 5.0:
        # Fallback: flag if raw error exceeds 5%
        anomaly_flag = True
    
    # Construct human-readable summary
    direction_match = "✓" if directional_hit else "✗"
    notes = (
        f"{direction_match} Predicted {predicted_direction} ({predicted_magnitude:+.2f}%) "
        f"| Actual {actual_direction} ({actual_magnitude:+.2f}%) "
        f"| Error {magnitude_error:.2f}% | Conf {predicted_confidence:.2f}"
    )
    
    return CalibrationResult(
        ticker=ticker,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        predicted_confidence=predicted_confidence,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        confidence_weighted_error=confidence_weighted_error,
        anomaly_flag=anomaly_flag,
        notes=notes,
    )


def batch_calibrate(
    predictions: list[dict],
    actuals: list[dict],
    direction_threshold: float = 0.5,
    anomaly_sigma: float = 2.0,
) -> tuple[list[CalibrationResult], dict]:
    """
    Compare multiple predictions against actuals, returning results and aggregate stats.
    
    Args:
        predictions: List of dicts with keys ticker, magnitude, confidence.
        actuals: List of dicts with keys ticker, magnitude.
        direction_threshold: Magnitude threshold for NEUTRAL classification.
        anomaly_sigma: Std-dev multiplier for anomaly detection.
    
    Returns:
        Tuple of (list of CalibrationResult, dict of aggregate stats).
    """
    results = []
    
    # Build actuals lookup by ticker
    actuals_map = {a["ticker"]: a["magnitude"] for a in actuals}
    
    # Compute historical stdev of errors for anomaly detection
    errors_for_stdev = []
    for pred in predictions:
        ticker = pred["ticker"]
        if ticker in actuals_map:
            err = abs(pred["magnitude"] - actuals_map[ticker])
            errors_for_stdev.append(err)
    
    historical_stdev = None
    if errors_for_stdev and len(errors_for_stdev) > 1:
        mean_error = sum(errors_for_stdev) / len(errors_for_stdev)
        variance = sum((e - mean_error) ** 2 for e in errors_for_stdev) / len(errors_for_stdev)
        historical_stdev = math.sqrt(variance)
    
    # Calibrate each prediction
    for pred in predictions:
        ticker = pred["ticker"]
        if ticker not in actuals_map:
            continue
        
        result = compare_predictions(
            ticker=ticker,
            predicted_magnitude=pred["magnitude"],
            predicted_confidence=pred.get("confidence", 0.5),
            actual_magnitude=actuals_map[ticker],
            direction_threshold=direction_threshold,
            anomaly_sigma=anomaly_sigma,
            historical_stdev=historical_stdev,
        )
        results.append(result)
    
    # Compute aggregate statistics
    hits = sum(1 for r in results if r.directional_hit)
    total = len(results)
    directional_accuracy = hits / total if total > 0 else 0.0
    
    avg_magnitude_error = (
        sum(r.magnitude_error for r in results) / total if total > 0 else 0.0
    )
    
    avg_confidence_weighted = (
        sum(r.confidence_weighted_error for r in results) / total if total > 0 else 0.0
    )
    
    anomalies = sum(1 for r in results if r.anomaly_flag)
    
    stats = {
        "total_predictions": total,
        "directional_hits": hits,
        "directional_accuracy": directional_accuracy,
        "avg_magnitude_error_pct": avg_magnitude_error,
        "avg_confidence_weighted_error": avg_confidence_weighted,
        "anomalies_detected": anomalies,
        "historical_error_stdev": historical_stdev,
    }
    
    return results, stats
