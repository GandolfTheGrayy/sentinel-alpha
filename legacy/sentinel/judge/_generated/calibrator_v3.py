"""
Sentinel Calibrator — Post-mortem accuracy analyzer.

Compares Sentinel's predicted price movements against actual market outcomes,
calculating directional accuracy, magnitude error, and confidence-weighted metrics.
Feeds results into Judge's heuristic refinement loop for model recalibration.

Typical flow:
  1. Judge.predictor emits CalibrationInput (ticker, predicted_move, confidence).
  2. At market close, resolver.py fetches actual price deltas.
  3. calibrator.compare() ingests both and returns CalibrationResult.
  4. Judge's post-mortem aggregates results across portfolio for weekly review.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json
from datetime import datetime


@dataclass
class CalibrationInput:
    """Predicted market move from Judge predictor."""
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    predicted_magnitude: float  # percentage, e.g. 2.5 for +2.5%
    confidence_score: float  # 0.0 to 1.0
    prediction_timestamp: str  # ISO 8601
    prediction_rationale: str


@dataclass
class ActualMarketMove:
    """Actual price movement observed post-prediction."""
    ticker: str
    actual_direction: str  # "UP", "DOWN", "NEUTRAL"
    actual_magnitude: float  # percentage
    open_price: float
    close_price: float
    resolution_timestamp: str  # ISO 8601


@dataclass
class CalibrationResult:
    """Comparison outcome: predicted vs. actual."""
    ticker: str
    prediction_timestamp: str
    resolution_timestamp: str
    
    # Directional accuracy
    predicted_direction: str
    actual_direction: str
    directional_hit: bool  # True if direction matches or both neutral
    
    # Magnitude analysis
    predicted_magnitude: float
    actual_magnitude: float
    magnitude_error: float  # |predicted - actual|
    magnitude_error_pct: float  # error as pct of actual (avoid div-by-zero)
    
    # Confidence weighting
    confidence_score: float
    weighted_error: float  # magnitude_error * confidence_score
    
    # Summary
    prediction_rationale: str
    calibration_quality: str  # "EXCELLENT", "GOOD", "FAIR", "POOR"
    notes: str
    
    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def classify_direction(magnitude: float) -> str:
    """
    Classify price move into directional bucket.
    
    Args:
        magnitude: Signed percentage change (positive = up, negative = down).
    
    Returns:
        "UP", "DOWN", or "NEUTRAL" (threshold ±0.1%).
    """
    threshold = 0.1
    if magnitude > threshold:
        return "UP"
    elif magnitude < -threshold:
        return "DOWN"
    else:
        return "NEUTRAL"


def compare(
    prediction: CalibrationInput,
    actual: ActualMarketMove,
) -> CalibrationResult:
    """
    Compare predicted vs. actual market move and score calibration.
    
    Args:
        prediction: Judge predictor's output (direction, magnitude, confidence).
        actual: Market's actual close-to-close or open-to-close move.
    
    Returns:
        CalibrationResult with directional hit, magnitude error, and quality rating.
    """
    # Validate ticker consistency
    if prediction.ticker != actual.ticker:
        raise ValueError(
            f"Ticker mismatch: prediction={prediction.ticker}, actual={actual.ticker}"
        )
    
    # Directional accuracy
    directional_hit = (
        prediction.predicted_direction == actual.actual_direction
        or (prediction.predicted_direction == "NEUTRAL" and actual.actual_direction == "NEUTRAL")
    )
    
    # Magnitude error
    magnitude_error = abs(prediction.predicted_magnitude - actual.actual_magnitude)
    
    # Magnitude error as percentage of actual (handle near-zero actuals)
    if abs(actual.actual_magnitude) < 0.01:
        magnitude_error_pct = magnitude_error * 100.0 if magnitude_error > 0 else 0.0
    else:
        magnitude_error_pct = (magnitude_error / abs(actual.actual_magnitude)) * 100.0
    
    # Confidence-weighted error
    weighted_error = magnitude_error * prediction.confidence_score
    
    # Calibration quality rating
    calibration_quality = _rate_calibration(
        directional_hit=directional_hit,
        magnitude_error=magnitude_error,
        confidence_score=prediction.confidence_score,
    )
    
    # Summary notes
    direction_status = "✓ MATCH" if directional_hit else "✗ MISS"
    notes = (
        f"Predicted {prediction.predicted_magnitude:+.2f}% {prediction.predicted_direction} "
        f"@ {prediction.confidence_score:.1%} confidence. "
        f"Actual: {actual.actual_magnitude:+.2f}% {actual.actual_direction}. "
        f"Error: {magnitude_error:.2f}% ({magnitude_error_pct:.1f}% of actual). "
        f"Direction: {direction_status}."
    )
    
    return CalibrationResult(
        ticker=prediction.ticker,
        prediction_timestamp=prediction.prediction_timestamp,
        resolution_timestamp=actual.resolution_timestamp,
        predicted_direction=prediction.predicted_direction,
        actual_direction=actual.actual_direction,
        directional_hit=directional_hit,
        predicted_magnitude=prediction.predicted_magnitude,
        actual_magnitude=actual.actual_magnitude,
        magnitude_error=magnitude_error,
        magnitude_error_pct=magnitude_error_pct,
        confidence_score=prediction.confidence_score,
        weighted_error=weighted_error,
        prediction_rationale=prediction.prediction_rationale,
        calibration_quality=calibration_quality,
        notes=notes,
    )


def _rate_calibration(
    directional_hit: bool,
    magnitude_error: float,
    confidence_score: float,
) -> str:
    """
    Heuristic rating of prediction calibration quality.
    
    Args:
        directional_hit: Whether predicted direction matched actual.
        magnitude_error: Absolute difference in percentage points.
        confidence_score: Confidence 0.0–1.0.
    
    Returns:
        Quality tier: "EXCELLENT", "GOOD", "FAIR", or "POOR".
    """
    # Strong directional hit with low error
    if directional_hit and magnitude_error <= 0.5 and confidence_score >= 0.7:
        return "EXCELLENT"
    
    # Directional hit with moderate error, or high confidence despite error
    if directional_hit and magnitude_error <= 2.0:
        return "GOOD"
    
    # Directional hit with larger error, or directional miss with low error
    if directional_hit or magnitude_error <= 1.5:
        return "FAIR"
    
    # Directional miss and large error
    return "POOR"


def batch_calibrate(
    predictions: list[CalibrationInput],
    actuals: list[ActualMarketMove],
) -> list[CalibrationResult]:
    """
    Calibrate multiple prediction–actual pairs in batch.
    
    Args:
        predictions: List of CalibrationInput from Judge.
        actuals: List of ActualMarketMove from market close data.
    
    Returns:
        List of CalibrationResult, one per matched pair.
    """
    # Index actuals by ticker for O(1) lookup
    actual_by_ticker = {a.ticker: a for a in actuals}
    
    results = []
    for pred in predictions:
        if pred.ticker not in actual_by_ticker:
