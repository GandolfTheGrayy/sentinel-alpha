"""
Calibrator module for Sentinel Sentiment Engine.

Compares predicted price movements against actual market outcomes, computing
directional accuracy and magnitude error. Produces CalibrationResult objects
that feed into Judge post-mortems and heuristic refinement loops.

This module bridges Predictor outputs (predicted_direction, predicted_magnitude)
with resolved market outcomes (actual_direction, actual_magnitude) to measure
model fidelity and identify systematic biases.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """
    Holds comparison metrics between predicted and actual market moves.
    
    Attributes:
        ticker: Stock symbol being evaluated.
        prediction_date: ISO date string when prediction was made.
        predicted_direction: -1 (down), 0 (hold), or +1 (up).
        predicted_magnitude: Float percentage change predicted (e.g., 2.5 for +2.5%).
        actual_direction: -1 (down), 0 (hold), or +1 (up).
        actual_magnitude: Float percentage change observed (e.g., -1.2 for -1.2%).
        directional_correct: Boolean; True if predicted_direction == actual_direction.
        magnitude_error: Float absolute difference |predicted_magnitude - actual_magnitude|.
        signed_error: Float signed difference (predicted - actual).
        magnitude_error_pct: Float magnitude_error as % of |actual_magnitude| (inf if actual == 0).
        prediction_id: Optional identifier linking to upstream prediction record.
        notes: Optional human-readable notes on anomalies or edge cases.
    """
    ticker: str
    prediction_date: str
    predicted_direction: int  # -1, 0, +1
    predicted_magnitude: float
    actual_direction: int  # -1, 0, +1
    actual_magnitude: float
    directional_correct: bool = field(init=False)
    magnitude_error: float = field(init=False)
    signed_error: float = field(init=False)
    magnitude_error_pct: float = field(init=False)
    prediction_id: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Compute derived fields after initialization."""
        self.directional_correct = self.predicted_direction == self.actual_direction
        self.magnitude_error = abs(self.predicted_magnitude - self.actual_magnitude)
        self.signed_error = self.predicted_magnitude - self.actual_magnitude
        
        if abs(self.actual_magnitude) < 1e-10:
            self.magnitude_error_pct = float('inf') if self.magnitude_error > 1e-10 else 0.0
        else:
            self.magnitude_error_pct = (self.magnitude_error / abs(self.actual_magnitude)) * 100.0


def calibrate_prediction(
    ticker: str,
    prediction_date: str,
    predicted_direction: int,
    predicted_magnitude: float,
    actual_direction: int,
    actual_magnitude: float,
    prediction_id: Optional[str] = None,
    notes: Optional[str] = None,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move and return calibration metrics.
    
    Args:
        ticker: Stock symbol.
        prediction_date: ISO date when prediction was issued.
        predicted_direction: -1, 0, or +1 from predictor output.
        predicted_magnitude: Predicted % change (e.g., 2.5).
        actual_direction: Observed -1, 0, or +1.
        actual_magnitude: Observed % change (e.g., -1.2).
        prediction_id: Optional upstream record ID.
        notes: Optional annotation.
    
    Returns:
        CalibrationResult with directional_correct, magnitude_error, and error_pct fields.
    """
    return CalibrationResult(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_magnitude=predicted_magnitude,
        actual_direction=actual_direction,
        actual_magnitude=actual_magnitude,
        prediction_id=prediction_id,
        notes=notes,
    )


def batch_calibrate(
    predictions: list[dict],
) -> list[CalibrationResult]:
    """
    Calibrate a batch of predicted vs. actual outcomes.
    
    Args:
        predictions: List of dicts with keys:
            ticker, prediction_date, predicted_direction, predicted_magnitude,
            actual_direction, actual_magnitude, prediction_id (opt), notes (opt).
    
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
            actual_direction=pred["actual_direction"],
            actual_magnitude=pred["actual_magnitude"],
            prediction_id=pred.get("prediction_id"),
            notes=pred.get("notes"),
        )
        results.append(result)
    return results


def directional_accuracy(
    results: list[CalibrationResult],
) -> float:
    """
    Compute fraction of predictions with correct direction.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Float in [0, 1]; 1.0 = perfect directional accuracy.
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.directional_correct)
    return correct / len(results)


def mean_magnitude_error(
    results: list[CalibrationResult],
) -> float:
    """
    Compute mean absolute magnitude error across results.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Float mean absolute error in percentage points.
    """
    if not results:
        return 0.0
    return sum(r.magnitude_error for r in results) / len(results)


def rmse_magnitude(
    results: list[CalibrationResult],
) -> float:
    """
    Compute root-mean-square error of magnitude predictions.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Float RMSE in percentage points.
    """
    if not results:
        return 0.0
    sum_sq = sum(r.magnitude_error ** 2 for r in results)
    return math.sqrt(sum_sq / len(results))


def bias_signed_error(
    results: list[CalibrationResult],
) -> float:
    """
    Compute mean signed error; positive = systematic over-prediction.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Float mean signed error. Positive = predicted too high, negative = too low.
    """
    if not results:
        return 0.0
    return sum(r.signed_error for r in results) / len(results)


def calibration_summary(
    results: list[CalibrationResult],
) -> dict:
    """
    Return dictionary of aggregated calibration metrics.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Dict with keys: directional_accuracy, mean_magnitude_error, rmse,
        bias, total_predictions, correct_predictions.
    """
    return {
        "directional_accuracy": directional_accuracy(results),
        "mean_magnitude_error": mean_magnitude_error(results),
        "rmse": rmse_magnitude(results),
        "bias": bias_signed_error(results),
        "total_predictions": len(results),
        "correct_predictions": sum(1 for r in results if r.directional_correct),
    }
