"""
Anomaly Detection Module for Sentinel Sentiment Engine.

Detects when actual market moves exceed 2x the predicted residual and generates
AnomalyAlert dataclasses for post-mortem analysis and heuristic refinement.
Integrates with Judge's daily post-mortem workflow to flag surprising market behavior.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class AnomalyAlert:
    """
    Represents a detected market anomaly where actual move significantly exceeded prediction.
    
    Attributes:
        ticker: Stock ticker symbol.
        prediction_date: Date when prediction was made.
        predicted_move_pct: Predicted price move as percentage.
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        predicted_confidence: Confidence score (0.0–1.0) from predictor.
        actual_move_pct: Actual observed price move as percentage.
        actual_direction: "UP", "DOWN", or "NEUTRAL" based on realized move.
        residual_pct: abs(actual_move_pct - predicted_move_pct).
        anomaly_ratio: actual_move_pct / predicted_move_pct (if predicted != 0, else inf).
        threshold_exceeded: True if anomaly_ratio > 2.0 or predicted was near zero.
        alert_severity: "LOW", "MEDIUM", "HIGH" based on ratio magnitude.
        market_context: Optional free-text note on macro/news context.
        flagged_at: Timestamp when anomaly was detected.
    """
    
    ticker: str
    prediction_date: str
    predicted_move_pct: float
    predicted_direction: str
    predicted_confidence: float
    actual_move_pct: float
    actual_direction: str
    residual_pct: float
    anomaly_ratio: float
    threshold_exceeded: bool
    alert_severity: str
    market_context: Optional[str] = None
    flagged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        """Serialize AnomalyAlert to dictionary for logging/storage."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date,
            "predicted_move_pct": self.predicted_move_pct,
            "predicted_direction": self.predicted_direction,
            "predicted_confidence": self.predicted_confidence,
            "actual_move_pct": self.actual_move_pct,
            "actual_direction": self.actual_direction,
            "residual_pct": self.residual_pct,
            "anomaly_ratio": self.anomaly_ratio,
            "threshold_exceeded": self.threshold_exceeded,
            "alert_severity": self.alert_severity,
            "market_context": self.market_context,
            "flagged_at": self.flagged_at,
        }
    
    def to_json(self) -> str:
        """Serialize AnomalyAlert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def compute_anomaly_alert(
    ticker: str,
    prediction_date: str,
    predicted_move_pct: float,
    predicted_direction: str,
    predicted_confidence: float,
    actual_move_pct: float,
    market_context: Optional[str] = None,
) -> AnomalyAlert:
    """
    Compute anomaly alert by comparing predicted vs. actual market move.
    
    Args:
        ticker: Stock symbol.
        prediction_date: ISO date string of prediction.
        predicted_move_pct: Predicted percentage move (can be negative).
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        predicted_confidence: Confidence score (0.0–1.0).
        actual_move_pct: Actual realized percentage move.
        market_context: Optional explanatory context.
    
    Returns:
        AnomalyAlert dataclass with anomaly metrics and severity.
    """
    
    # Determine actual direction
    if actual_move_pct > 0.5:
        actual_direction = "UP"
    elif actual_move_pct < -0.5:
        actual_direction = "DOWN"
    else:
        actual_direction = "NEUTRAL"
    
    # Compute residual
    residual_pct = abs(actual_move_pct - predicted_move_pct)
    
    # Compute anomaly ratio with safe division
    if abs(predicted_move_pct) > 0.01:
        anomaly_ratio = abs(actual_move_pct) / abs(predicted_move_pct)
    else:
        # If prediction was near zero, treat any non-negligible move as anomalous
        anomaly_ratio = float("inf") if abs(actual_move_pct) > 0.5 else 0.0
    
    # Threshold check: 2x the predicted residual
    threshold_exceeded = anomaly_ratio > 2.0 or (
        abs(predicted_move_pct) < 0.01 and abs(actual_move_pct) > 0.5
    )
    
    # Severity scoring
    if anomaly_ratio == float("inf") or anomaly_ratio > 5.0:
        alert_severity = "HIGH"
    elif anomaly_ratio > 2.0:
        alert_severity = "MEDIUM"
    else:
        alert_severity = "LOW"
    
    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_move_pct=predicted_move_pct,
        predicted_direction=predicted_direction,
        predicted_confidence=predicted_confidence,
        actual_move_pct=actual_move_pct,
        actual_direction=actual_direction,
        residual_pct=residual_pct,
        anomaly_ratio=anomaly_ratio if anomaly_ratio != float("inf") else -1.0,
        threshold_exceeded=threshold_exceeded,
        alert_severity=alert_severity,
        market_context=market_context,
    )


def filter_anomalies_by_severity(
    alerts: list[AnomalyAlert], min_severity: str = "MEDIUM"
) -> list[AnomalyAlert]:
    """
    Filter anomaly alerts by minimum severity level.
    
    Args:
        alerts: List of AnomalyAlert objects.
        min_severity: Minimum level to include ("LOW", "MEDIUM", "HIGH").
    
    Returns:
        Filtered list containing only alerts at or above min_severity.
    """
    severity_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    min_rank = severity_rank.get(min_severity, 1)
    return [a for a in alerts if severity_rank.get(a.alert_severity, 0) >= min_rank]


def summarize_anomalies(alerts: list[AnomalyAlert]) -> dict:
    """
    Generate summary statistics over a batch of anomaly alerts.
    
    Args:
        alerts: List of AnomalyAlert objects.
    
    Returns:
        Dict with counts by severity, mean anomaly ratio, and flagged tickers.
    """
    if not alerts:
        return {
            "total_anomalies": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "mean_anomaly_ratio": 0.0,
            "flagged_tickers": [],
        }
    
    high = sum(1 for a in alerts if a.alert_severity == "HIGH")
    medium = sum(1 for a in alerts if a.alert_severity == "MEDIUM")
    low = sum(1 for a in alerts if a.alert_severity == "LOW")
    
    # Mean anomaly ratio (excluding infinity markers)
    valid_ratios = [a.anomaly_ratio for a in alerts if a.anomaly_ratio > 0]
    mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_
