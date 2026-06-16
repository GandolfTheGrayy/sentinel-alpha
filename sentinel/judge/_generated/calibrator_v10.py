"""
Calibrator — Post-prediction accuracy analyzer for Sentinel.

Compares predicted price movements against actual market outcomes,
calculating directional accuracy and magnitude error metrics. Returns
structured CalibrationResult objects for Judge post-mortem analysis.

Feeds into daily_judge() workflow to refine prediction confidence
scoring and heuristic weighting.
"""

from dataclasses import dataclass, field
from typing import Optional
import math
from enum import Enum


class Direction(Enum):
    """Enumeration of market movement directions."""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


@dataclass
class PredictionRecord:
    """Single prediction: ticker, predicted move, predicted direction, confidence."""
    ticker: str
    predicted_direction: Direction
    predicted_magnitude: float
    confidence_score: float
    prediction_timestamp: str


@dataclass
class ActualOutcome:
    """Single actual market outcome: ticker, actual direction, actual magnitude."""
    ticker: str
    actual_direction: Direction
    actual_magnitude: float
    outcome_timestamp: str


@dataclass
class CalibrationResult:
    """Aggregated calibration metrics for a batch of predictions."""
    ticker: str
    directional_accuracy: float
    magnitude_mae: float
    magnitude_rmse: float
    total_predictions: int
    correct_direction_count: int
    avg_predicted_magnitude: float
    avg_actual_magnitude: float
    avg_confidence_of_correct: float
    avg_confidence_of_incorrect: float
    timestamp: str
    details: list = field(default_factory=list)

    def __repr__(self) -> str:
        """Human-readable summary of calibration."""
        return (
            f"CalibrationResult({self.ticker}) | "
            f"Dir.Acc={self.directional_accuracy:.1%} | "
            f"MAE={self.magnitude_mae:.3f} | "
            f"RMSE={self.magnitude_rmse:.3f} | "
            f"N={self.total_predictions}"
        )


def compare_direction(predicted: Direction, actual: Direction) -> bool:
    """Check if predicted direction matches actual direction."""
    return predicted == actual


def calculate_magnitude_error(predicted: float, actual: float) -> float:
    """Return absolute error between predicted and actual magnitude."""
    return abs(predicted - actual)


def batch_calibrate(
    predictions: list[PredictionRecord],
    outcomes: list[ActualOutcome],
    timestamp: str
) -> Optional[CalibrationResult]:
    """
    Compare predictions against actual outcomes, return aggregated CalibrationResult.
    
    Matches by ticker. If no outcomes found for a ticker, returns None.
    Calculates directional accuracy, MAE, RMSE, and confidence breakdowns.
    """
    if not predictions or not outcomes:
        return None

    # Group outcomes by ticker for fast lookup
    outcomes_by_ticker = {}
    for outcome in outcomes:
        if outcome.ticker not in outcomes_by_ticker:
            outcomes_by_ticker[outcome.ticker] = []
        outcomes_by_ticker[outcome.ticker].append(outcome)

    # Use first ticker from predictions for grouping (assumes homogeneous batch)
    ticker = predictions[0].ticker

    if ticker not in outcomes_by_ticker or not outcomes_by_ticker[ticker]:
        return None

    ticker_outcomes = outcomes_by_ticker[ticker]
    ticker_predictions = [p for p in predictions if p.ticker == ticker]

    if not ticker_predictions or not ticker_outcomes:
        return None

    # Match predictions to outcomes (by order; assumes alignment)
    matches = []
    for i, pred in enumerate(ticker_predictions):
        if i < len(ticker_outcomes):
            outcome = ticker_outcomes[i]
            matches.append((pred, outcome))

    if not matches:
        return None

    # Compute metrics
    correct_direction_count = 0
    magnitude_errors = []
    predicted_magnitudes = []
    actual_magnitudes = []
    correct_confidences = []
    incorrect_confidences = []
    detail_rows = []

    for pred, outcome in matches:
        direction_match = compare_direction(pred.predicted_direction, outcome.actual_direction)
        if direction_match:
            correct_direction_count += 1
            correct_confidences.append(pred.confidence_score)
        else:
            incorrect_confidences.append(pred.confidence_score)

        mag_error = calculate_magnitude_error(pred.predicted_magnitude, outcome.actual_magnitude)
        magnitude_errors.append(mag_error)
        predicted_magnitudes.append(pred.predicted_magnitude)
        actual_magnitudes.append(outcome.actual_magnitude)

        detail_rows.append({
            "predicted_direction": pred.predicted_direction.value,
            "actual_direction": outcome.actual_direction.value,
            "predicted_magnitude": pred.predicted_magnitude,
            "actual_magnitude": outcome.actual_magnitude,
            "magnitude_error": mag_error,
            "confidence": pred.confidence_score,
            "direction_correct": direction_match
        })

    # Aggregate
    directional_accuracy = correct_direction_count / len(matches) if matches else 0.0
    magnitude_mae = sum(magnitude_errors) / len(magnitude_errors) if magnitude_errors else 0.0
    magnitude_rmse = math.sqrt(sum(e ** 2 for e in magnitude_errors) / len(magnitude_errors)) if magnitude_errors else 0.0
    avg_predicted = sum(predicted_magnitudes) / len(predicted_magnitudes) if predicted_magnitudes else 0.0
    avg_actual = sum(actual_magnitudes) / len(actual_magnitudes) if actual_magnitudes else 0.0
    avg_conf_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
    avg_conf_incorrect = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0

    return CalibrationResult(
        ticker=ticker,
        directional_accuracy=directional_accuracy,
        magnitude_mae=magnitude_mae,
        magnitude_rmse=magnitude_rmse,
        total_predictions=len(matches),
        correct_direction_count=correct_direction_count,
        avg_predicted_magnitude=avg_predicted,
        avg_actual_magnitude=avg_actual,
        avg_confidence_of_correct=avg_conf_correct,
        avg_confidence_of_incorrect=avg_conf_incorrect,
        timestamp=timestamp,
        details=detail_rows
    )


def single_calibrate(
    prediction: PredictionRecord,
    outcome: ActualOutcome,
    timestamp: str
) -> CalibrationResult:
    """
    Calibrate a single prediction-outcome pair into a CalibrationResult.
    
    Convenience wrapper around batch_calibrate for one-off comparisons.
    """
    result = batch_calibrate([prediction], [outcome], timestamp)
    return result if result is not None else CalibrationResult(
        ticker=prediction.ticker,
        directional_accuracy=0.0,
        magnitude_mae=0.0,
        magnitude_rmse=0.0,
        total_predictions=0,
        correct_direction_count=0,
        avg_predicted_magnitude=0.0,
        avg_actual_magnitude=0.0,
        avg_confidence_of_correct=0.0,
        avg_confidence_of_incorrect=0.0,
        timestamp=timestamp,
        details=[]
    )
