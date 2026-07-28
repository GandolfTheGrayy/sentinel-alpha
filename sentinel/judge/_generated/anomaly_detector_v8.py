"""
Anomaly Detector for Sentinel Sentiment Engine.

Detects when actual market moves deviate significantly from predicted residuals,
flagging outliers that warrant post-mortem investigation. Compares predicted vs.
actual price deltas and raises AnomalyAlert when moves exceed 2x the predicted
residual threshold, indicating model miscalibration or unforeseen events.

Integrates with Judge postmortem pipeline for heuristic refinement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class AnomalyAlert:
    """Dataclass representing a detected market move anomaly."""

    ticker: str
    prediction_date: str
    predicted_delta_pct: float
    actual_delta_pct: float
    predicted_residual: float
    anomaly_multiplier: float
    severity: str  # "low", "medium", "high", "critical"
    reasoning: str
    alert_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    recommendation: Optional[str] = None
    model_version: str = "sentinel-v1"

    def to_dict(self) -> dict:
        """Serialize AnomalyAlert to dictionary for logging/storage."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date,
            "predicted_delta_pct": round(self.predicted_delta_pct, 4),
            "actual_delta_pct": round(self.actual_delta_pct, 4),
            "predicted_residual": round(self.predicted_residual, 4),
            "anomaly_multiplier": round(self.anomaly_multiplier, 2),
            "severity": self.severity,
            "reasoning": self.reasoning,
            "recommendation": self.recommendation,
            "alert_timestamp": self.alert_timestamp,
            "model_version": self.model_version,
        }

    def to_json(self) -> str:
        """Serialize AnomalyAlert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def detect_anomaly(
    ticker: str,
    prediction_date: str,
    predicted_delta_pct: float,
    actual_delta_pct: float,
    predicted_residual: float,
    anomaly_threshold_multiplier: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect market move anomalies by comparing actual vs. predicted deltas.

    Args:
        ticker: Stock ticker symbol.
        prediction_date: ISO date string when prediction was made.
        predicted_delta_pct: Predicted price change (%).
        actual_delta_pct: Actual observed price change (%).
        predicted_residual: Model's estimated prediction error margin (%).
        anomaly_threshold_multiplier: Multiplier threshold (default 2.0x residual).

    Returns:
        AnomalyAlert if anomaly detected, None otherwise.
    """
    if predicted_residual <= 0:
        return None

    actual_error = abs(actual_delta_pct - predicted_delta_pct)
    threshold = predicted_residual * anomaly_threshold_multiplier
    multiplier = actual_error / predicted_residual if predicted_residual > 0 else 0

    if actual_error <= threshold:
        return None

    severity = _classify_severity(multiplier, actual_error)
    reasoning = _generate_reasoning(
        ticker, predicted_delta_pct, actual_delta_pct, multiplier, threshold
    )
    recommendation = _generate_recommendation(severity, ticker)

    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_delta_pct=predicted_delta_pct,
        actual_delta_pct=actual_delta_pct,
        predicted_residual=predicted_residual,
        anomaly_multiplier=multiplier,
        severity=severity,
        reasoning=reasoning,
        recommendation=recommendation,
    )


def _classify_severity(multiplier: float, actual_error: float) -> str:
    """Classify anomaly severity based on multiplier and absolute error."""
    if multiplier >= 5.0 or actual_error >= 10.0:
        return "critical"
    elif multiplier >= 3.5 or actual_error >= 7.0:
        return "high"
    elif multiplier >= 2.5 or actual_error >= 5.0:
        return "medium"
    else:
        return "low"


def _generate_reasoning(
    ticker: str,
    predicted_delta: float,
    actual_delta: float,
    multiplier: float,
    threshold: float,
) -> str:
    """Generate human-readable explanation for anomaly."""
    direction = "down" if actual_delta < predicted_delta else "up"
    return (
        f"{ticker} moved {direction} by {abs(actual_delta):.2f}% vs. prediction "
        f"of {predicted_delta:.2f}%. Actual error ({abs(actual_delta - predicted_delta):.2f}%) "
        f"exceeded threshold ({threshold:.2f}%) by {multiplier:.1f}x. "
        "Potential model miscalibration, undisclosed event, or sentiment shift."
    )


def _generate_recommendation(severity: str, ticker: str) -> str:
    """Generate post-mortem action recommendation."""
    if severity == "critical":
        return (
            f"URGENT: {ticker} warrants immediate deep-dive postmortem. "
            "Check for SEC filings, insider trades, or viral sentiment swings."
        )
    elif severity == "high":
        return (
            f"{ticker} anomaly suggests model blind spot. "
            "Review linguist certainty scores and RAG context retrieval."
        )
    elif severity == "medium":
        return f"{ticker} flagged for routine heuristic tuning in next weekly retrospective."
    else:
        return f"{ticker} minor deviation; monitor for trend."


def batch_detect_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    anomaly_threshold_multiplier: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Detect anomalies across multiple prediction-actual pairs.

    Args:
        predictions: List of dicts with keys: ticker, prediction_date, predicted_delta_pct, predicted_residual.
        actuals: List of dicts with keys: ticker, actual_delta_pct.
        anomaly_threshold_multiplier: Multiplier threshold.

    Returns:
        List of AnomalyAlert objects for detected anomalies.
    """
    alerts = []
    pred_map = {p["ticker"]: p for p in predictions}

    for actual in actuals:
        ticker = actual["ticker"]
        if ticker not in pred_map:
            continue

        pred = pred_map[ticker]
        alert = detect_anomaly(
            ticker=ticker,
            prediction_date=pred.get("prediction_date", "unknown"),
            predicted_delta_pct=pred["predicted_delta_pct"],
            actual_delta_pct=actual["actual_delta_pct"],
            predicted_residual=pred["predicted_residual"],
            anomaly_threshold_multiplier=anomaly_threshold_multiplier,
        )

        if alert:
            alerts.append(alert)

    return alerts


def filter_anomalies_by_severity(
    alerts: list[AnomalyAlert], min_severity: str
) -> list[AnomalyAlert]:
    """
    Filter anomalies by minimum severity threshold.

    Args:
        alerts: List of AnomalyAlert objects.
        min_severity: Minimum severity to include ("low", "medium", "high", "critical").

    Returns:
        Filtered list of alerts meeting severity threshold.
    """
    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    min_rank = severity_rank.get(min_severity, 1)

    return [a for a in alerts if severity_rank.get(a.severity, 0) >= min_rank]
