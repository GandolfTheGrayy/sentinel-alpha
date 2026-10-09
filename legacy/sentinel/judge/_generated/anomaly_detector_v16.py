"""
Anomaly Detector for Sentinel Sentiment Engine.

Detects when actual market moves exceed 2x the predicted residual,
generating AnomalyAlert dataclasses for post-mortem analysis and
heuristic refinement. Feeds into Judge calibration loop.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """Encapsulates a single market anomaly detection event."""
    
    ticker: str
    prediction_date: datetime
    predicted_direction: str  # "up", "down", "neutral"
    predicted_confidence: float  # 0.0 to 1.0
    actual_move_pct: float  # Observed daily/period return %
    predicted_residual_pct: float  # Expected prediction error margin %
    anomaly_ratio: float  # actual_move_pct / (2.0 * predicted_residual_pct)
    severity: AnomalySeverity
    reasoning: str = ""  # Human-readable explanation
    metadata: dict = field(default_factory=dict)  # Additional context
    detected_at: datetime = field(default_factory=datetime.utcnow)


def detect_anomaly(
    ticker: str,
    prediction_date: datetime,
    predicted_direction: str,
    predicted_confidence: float,
    actual_move_pct: float,
    predicted_residual_pct: float,
    baseline_threshold: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds 2x the predicted residual.
    
    Returns AnomalyAlert if anomaly detected, None otherwise.
    """
    if predicted_residual_pct <= 0.0:
        return None
    
    threshold = baseline_threshold * predicted_residual_pct
    anomaly_ratio = abs(actual_move_pct) / threshold if threshold > 0 else 0.0
    
    if anomaly_ratio <= 1.0:
        return None
    
    # Severity scoring: ratio of 1.0-2.0 is LOW, 2.0-4.0 is MEDIUM,
    # 4.0-8.0 is HIGH, 8.0+ is CRITICAL.
    if anomaly_ratio < 2.0:
        severity = AnomalySeverity.LOW
    elif anomaly_ratio < 4.0:
        severity = AnomalySeverity.MEDIUM
    elif anomaly_ratio < 8.0:
        severity = AnomalySeverity.HIGH
    else:
        severity = AnomalySeverity.CRITICAL
    
    # Build reasoning string.
    direction_match = (actual_move_pct > 0 and predicted_direction == "up") or \
                      (actual_move_pct < 0 and predicted_direction == "down") or \
                      (predicted_direction == "neutral")
    direction_str = "matched" if direction_match else "contradicted"
    
    reasoning = (
        f"Predicted {predicted_direction} (confidence {predicted_confidence:.2%}), "
        f"expected residual ±{predicted_residual_pct:.2f}%. "
        f"Actual move {actual_move_pct:+.2f}% {direction_str} prediction. "
        f"Anomaly ratio: {anomaly_ratio:.2f}x threshold."
    )
    
    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_confidence=predicted_confidence,
        actual_move_pct=actual_move_pct,
        predicted_residual_pct=predicted_residual_pct,
        anomaly_ratio=anomaly_ratio,
        severity=severity,
        reasoning=reasoning,
    )


def batch_detect_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    baseline_threshold: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Batch-detect anomalies across multiple ticker predictions.
    
    Expected dict keys: ticker, prediction_date, predicted_direction,
    predicted_confidence, actual_move_pct, predicted_residual_pct.
    """
    alerts = []
    for pred, actual in zip(predictions, actuals):
        if pred.get("ticker") != actual.get("ticker"):
            continue
        
        alert = detect_anomaly(
            ticker=pred["ticker"],
            prediction_date=pred.get("prediction_date", datetime.utcnow()),
            predicted_direction=pred.get("predicted_direction", "neutral"),
            predicted_confidence=pred.get("predicted_confidence", 0.5),
            actual_move_pct=actual.get("actual_move_pct", 0.0),
            predicted_residual_pct=pred.get("predicted_residual_pct", 1.0),
            baseline_threshold=baseline_threshold,
        )
        if alert:
            alerts.append(alert)
    
    return alerts


def summarize_anomalies(alerts: list[AnomalyAlert]) -> dict:
    """
    Produce summary stats from a list of anomaly alerts.
    
    Returns dict with counts by severity, average ratio, most critical tickers.
    """
    if not alerts:
        return {
            "total_anomalies": 0,
            "by_severity": {},
            "mean_ratio": 0.0,
            "top_critical_tickers": [],
        }
    
    severity_counts = {}
    for severity in AnomalySeverity:
        severity_counts[severity.value] = sum(
            1 for a in alerts if a.severity == severity
        )
    
    mean_ratio = sum(a.anomaly_ratio for a in alerts) / len(alerts)
    
    # Top 5 most severe anomalies.
    sorted_alerts = sorted(alerts, key=lambda a: a.anomaly_ratio, reverse=True)
    top_tickers = [
        (a.ticker, a.anomaly_ratio, a.severity.value)
        for a in sorted_alerts[:5]
    ]
    
    return {
        "total_anomalies": len(alerts),
        "by_severity": severity_counts,
        "mean_ratio": mean_ratio,
        "top_critical_tickers": top_tickers,
    }
