"""Calibrator module for Sentinel Sentiment Engine.

This module compares predicted price movements against actual market outcomes,
calculating directional accuracy, magnitude error, and confidence-weighted
performance metrics. Results feed into the Judge's daily post-mortem and
heuristic refinement loop.

Core responsibility:
  - Accept prediction (direction, confidence, magnitude) and actual market move.
  - Compute hit/miss, directional accuracy, RMSE, and calibration curves.
  - Return CalibrationResult for aggregation across portfolio and time windows.
"""

from dataclasses import dataclass, field
from typing import Optional
import sqlite3
from datetime import datetime


@dataclass
class PredictionOutcome:
    """Single prediction + actual outcome for a ticker."""

    ticker: str
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    predicted_magnitude: float  # % change, e.g., 2.5
    predicted_confidence: float  # 0.0–1.0
    actual_direction: str  # "UP", "DOWN", "NEUTRAL"
    actual_magnitude: float  # % change (signed), e.g., 1.2 or -0.8
    eval_date: str  # ISO format "YYYY-MM-DD"
    horizon_hours: int  # prediction horizon in hours


@dataclass
class CalibrationResult:
    """Aggregated calibration metrics for one or more predictions."""

    total_predictions: int
    directional_hits: int
    directional_accuracy: float  # 0.0–1.0
    magnitude_rmse: float  # root mean squared error of magnitude
    confidence_weighted_accuracy: float  # accuracy weighted by confidence scores
    average_confidence: float  # mean confidence across all predictions
    up_hit_rate: float  # accuracy on UP predictions only
    down_hit_rate: float  # accuracy on DOWN predictions only
    neutral_hit_rate: float  # accuracy on NEUTRAL predictions only
    mean_prediction_magnitude: float  # average |predicted_magnitude|
    mean_actual_magnitude: float  # average |actual_magnitude|
    high_confidence_accuracy: Optional[float] = None  # accuracy for conf >= 0.7
    low_confidence_accuracy: Optional[float] = None  # accuracy for conf < 0.7
    anomalies: list = field(default_factory=list)  # outlier predictions flagged


def compare_directions(predicted: str, actual: str) -> bool:
    """Return True if predicted direction matches actual direction.

    Args:
        predicted: "UP", "DOWN", or "NEUTRAL"
        actual: "UP", "DOWN", or "NEUTRAL"

    Returns:
        True if directions match, False otherwise.
    """
    return predicted.upper() == actual.upper()


def magnitude_error(predicted_mag: float, actual_mag: float) -> float:
    """Compute absolute magnitude error between prediction and actual.

    Args:
        predicted_mag: Predicted % change magnitude.
        actual_mag: Actual % change magnitude (signed).

    Returns:
        Absolute error in percentage points.
    """
    return abs(predicted_mag - actual_mag)


def calibrate(outcomes: list[PredictionOutcome]) -> CalibrationResult:
    """Analyze predictions vs. actuals and return aggregated calibration metrics.

    Args:
        outcomes: List of PredictionOutcome objects from a time window or portfolio.

    Returns:
        CalibrationResult with directional accuracy, magnitude error, and confidence metrics.
    """
    if not outcomes:
        return CalibrationResult(
            total_predictions=0,
            directional_hits=0,
            directional_accuracy=0.0,
            magnitude_rmse=0.0,
            confidence_weighted_accuracy=0.0,
            average_confidence=0.0,
            up_hit_rate=0.0,
            down_hit_rate=0.0,
            neutral_hit_rate=0.0,
            mean_prediction_magnitude=0.0,
            mean_actual_magnitude=0.0,
        )

    total = len(outcomes)
    hits = 0
    confidence_weighted_hits = 0.0
    total_confidence = 0.0
    magnitude_errors = []
    anomalies = []

    up_predictions = 0
    up_hits = 0
    down_predictions = 0
    down_hits = 0
    neutral_predictions = 0
    neutral_hits = 0

    prediction_magnitudes = []
    actual_magnitudes = []

    high_conf_hits = 0
    high_conf_total = 0
    low_conf_hits = 0
    low_conf_total = 0

    for outcome in outcomes:
        is_hit = compare_directions(outcome.predicted_direction, outcome.actual_direction)
        if is_hit:
            hits += 1

        # Confidence-weighted accuracy.
        confidence_weighted_hits += outcome.predicted_confidence if is_hit else 0.0
        total_confidence += outcome.predicted_confidence

        # Magnitude error.
        error = magnitude_error(outcome.predicted_magnitude, outcome.actual_magnitude)
        magnitude_errors.append(error ** 2)

        # Direction-specific accuracy.
        if outcome.predicted_direction.upper() == "UP":
            up_predictions += 1
            if is_hit:
                up_hits += 1
        elif outcome.predicted_direction.upper() == "DOWN":
            down_predictions += 1
            if is_hit:
                down_hits += 1
        else:
            neutral_predictions += 1
            if is_hit:
                neutral_hits += 1

        # Confidence buckets.
        if outcome.predicted_confidence >= 0.7:
            high_conf_total += 1
            if is_hit:
                high_conf_hits += 1
        else:
            low_conf_total += 1
            if is_hit:
                low_conf_hits += 1

        # Outliers: magnitude error > 5% or very low confidence with large miss.
        if error > 5.0 or (outcome.predicted_confidence > 0.75 and error > 3.0):
            anomalies.append(
                {
                    "ticker": outcome.ticker,
                    "predicted": outcome.predicted_direction,
                    "actual": outcome.actual_direction,
                    "magnitude_error": error,
                    "confidence": outcome.predicted_confidence,
                }
            )

        prediction_magnitudes.append(abs(outcome.predicted_magnitude))
        actual_magnitudes.append(abs(outcome.actual_magnitude))

    # Compute aggregates.
    directional_accuracy = hits / total if total > 0 else 0.0
    confidence_weighted_accuracy = (
        confidence_weighted_hits / total_confidence if total_confidence > 0 else 0.0
    )
    average_confidence = total_confidence / total if total > 0 else 0.0

    magnitude_rmse = (sum(magnitude_errors) / total) ** 0.5 if magnitude_errors else 0.0

    up_hit_rate = (up_hits / up_predictions) if up_predictions > 0 else None
    down_hit_rate = (down_hits / down_predictions) if down_predictions > 0 else None
    neutral_hit_rate = (neutral_hits / neutral_predictions) if neutral_predictions > 0 else None

    mean_prediction_magnitude = (
        sum(prediction_magnitudes) / len(prediction_magnitudes)
        if prediction_magnitudes
        else 0.0
    )
    mean_actual_magnitude = (
        sum(actual_magnitudes) / len(actual_magnitudes) if actual_magnitudes else 0.0
    )

    high_confidence_accuracy = (
        (high_conf_hits / high_conf_total) if high_conf_total > 0 else None
    )
    low_confidence_accuracy = (
        (low_conf_hits / low_conf_total) if low_conf_total > 0 else None
    )

    return CalibrationResult(
        total_predictions=total,
        directional_hits=hits,
        directional_accuracy=directional_accuracy,
        magnitude_rmse=magnitude_rmse,
        confidence_weighted_accuracy=confidence_weighted_accuracy,
