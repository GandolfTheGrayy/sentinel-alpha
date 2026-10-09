"""
Calibrator for Sentinel Sentiment Engine — Post-prediction analysis module.

Compares predicted price movements against actual market outcomes, computing
directional accuracy, magnitude error, and confidence-weighted calibration metrics.
Feeds results into Judge's daily post-mortem and heuristic refinement loop.

Used by: sentinel/judge/postmortem.py (stores results), sentinel/judge/resolver.py
(aggregates across multiple predictions).
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class CalibrationResult:
    """Encapsulates a single prediction vs. actual market move comparison."""
    
    ticker: str
    """Stock symbol (e.g., 'AAPL')."""
    
    predicted_direction: str
    """Direction predicted by model: 'UP', 'DOWN', or 'NEUTRAL'."""
    
    predicted_magnitude: float
    """Percentage point change predicted (e.g., 2.5 for +2.5%)."""
    
    predicted_confidence: float
    """Model confidence score (0.0–1.0)."""
    
    actual_direction: str
    """Actual direction observed: 'UP', 'DOWN', or 'NEUTRAL'."""
    
    actual_magnitude: float
    """Actual percentage point change observed (e.g., -1.3 for -1.3%)."""
    
    directional_hit: bool
    """True if predicted direction matched actual direction."""
    
    magnitude_error: float
    """Absolute difference between predicted and actual magnitude."""
    
    confidence_weighted_error: float
    """Magnitude error scaled down by model confidence (penalizes overconfident misses)."""
    
    calibration_score: float
    """Composite score (0–100): 100 means perfect direction + zero magnitude error."""


def compute_calibration(
    ticker: str,
    predicted_direction: str,
    predicted_magnitude: float,
    predicted_confidence: float,
    actual_price_start: float,
    actual_price_end: float,
) -> CalibrationResult:
    """
    Compare predicted vs. actual price movement and compute calibration metrics.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL').
        predicted_direction: Model's direction forecast ('UP', 'DOWN', 'NEUTRAL').
        predicted_magnitude: Model's magnitude forecast (signed percentage, e.g., 2.5).
        predicted_confidence: Model's confidence (0.0–1.0).
        actual_price_start: Opening price at prediction time.
        actual_price_end: Closing price at evaluation time.
    
    Returns:
        CalibrationResult with computed directional accuracy and magnitude error.
    
    Raises:
        ValueError: If prices are non-positive, direction invalid, or confidence out of range.
    """
    if actual_price_start <= 0 or actual_price_end <= 0:
        raise ValueError(f"Prices must be positive; got start={actual_price_start}, end={actual_price_end}.")
    
    if predicted_direction not in ('UP', 'DOWN', 'NEUTRAL'):
        raise ValueError(f"Invalid direction '{predicted_direction}'; must be UP, DOWN, or NEUTRAL.")
    
    if not 0.0 <= predicted_confidence <= 1.0:
        raise ValueError(f"Confidence must be in [0, 1]; got {predicted_confidence}.")
    
    # Compute actual magnitude and direction.
    actual_magnitude = ((actual_price_end - actual_price_start) / actual_price_start) * 100.0
    
    if actual_magnitude > 0.5:  # Small threshold to avoid 'NEUTRAL' classification noise.
        actual_direction = 'UP'
    elif actual_magnitude < -0.5:
        actual_direction = 'DOWN'
    else:
        actual_direction = 'NEUTRAL'
    
    # Directional hit: True if predicted direction matches actual direction.
    directional_hit = (predicted_direction == actual_direction)
    
    # Magnitude error: absolute difference.
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    # Confidence-weighted error: penalize overconfident predictions that miss.
    # If directional hit, reduce penalty. If miss, amplify by confidence.
    if directional_hit:
        confidence_weighted_error = magnitude_error * (1.0 - predicted_confidence * 0.5)
    else:
        confidence_weighted_error = magnitude_error * (1.0 + predicted_confidence * 0.5)
    
    # Calibration score (0–100): perfect = 100, decays with magnitude error and direction miss.
    # Base score: 100 if hit, 50 if miss.
    base_score = 100.0 if directional_hit else 50.0
    
    # Penalty: magnitude error, capped at base score.
    # Each 1% magnitude error reduces score by 2 points (scales down at high errors).
    magnitude_penalty = min(base_score - 10.0, magnitude_error * 2.0)
    calibration_score = max(10.0, base_score - magnitude_penalty)
    
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
        calibration_score=calibration_score,
    )


def aggregate_calibration(results: list[CalibrationResult]) -> dict:
    """
    Summarize calibration results across multiple predictions.
    
    Args:
        results: List of CalibrationResult objects.
    
    Returns:
        Dict with aggregate metrics: directional_accuracy, mean_magnitude_error,
        mean_confidence_weighted_error, mean_calibration_score, count.
    """
    if not results:
        return {
            'directional_accuracy': 0.0,
            'mean_magnitude_error': 0.0,
            'mean_confidence_weighted_error': 0.0,
            'mean_calibration_score': 0.0,
            'count': 0,
        }
    
    directional_hits = sum(1 for r in results if r.directional_hit)
    directional_accuracy = directional_hits / len(results)
    
    mean_magnitude_error = sum(r.magnitude_error for r in results) / len(results)
    mean_confidence_weighted_error = sum(r.confidence_weighted_error for r in results) / len(results)
    mean_calibration_score = sum(r.calibration_score for r in results) / len(results)
    
    return {
        'directional_accuracy': directional_accuracy,
        'mean_magnitude_error': mean_magnitude_error,
        'mean_confidence_weighted_error': mean_confidence_weighted_error,
        'mean_calibration_score': mean_calibration_score,
        'count': len(results),
    }


def calibration_to_dict(result: CalibrationResult) -> dict:
    """
    Serialize a CalibrationResult to a plain dict for storage or transmission.
    
    Args:
        result: CalibrationResult object.
    
    Returns:
        Dict representation with all fields.
    """
    return {
        'ticker': result.ticker,
        'predicted_direction': result.predicted_direction,
        'predicted_magnitude': result.predicted_magnitude,
        'predicted_confidence': result.predicted_confidence,
        'actual_direction': result.actual_direction,
        'actual_magnitude': result.actual_magnitude,
        'directional_hit': result.directional_hit,
        'magnitude_error': result.magnitude_error,
        'confidence_weighted_error': result.confidence_weighted_error,
        'calibration_score': result.calibration_score,
    }
