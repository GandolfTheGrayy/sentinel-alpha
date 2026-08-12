"""
Anomaly Detector — Sentinel Judge Pillar
=========================================

Detects when actual market moves exceed 2x the predicted residual.
Generates AnomalyAlert dataclass for downstream post-mortem analysis.

Used by sentinel/judge/postmortem.py to flag unusual prediction misses
and trigger deeper investigation into market microstructure or model drift.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AnomalyAlert:
    """A market move that exceeded 2x predicted residual threshold."""
    
    ticker: str
    date: datetime
    predicted_move_pct: float
    actual_move_pct: float
    residual_pct: float
    threshold_pct: float
    severity: str  # "low", "medium", "high"
    reason: str
    confidence: float  # 0.0-1.0, how confident we are in flagging


def compute_residual(predicted_move_pct: float, actual_move_pct: float) -> float:
    """
    Compute absolute prediction error as percentage points.
    
    Args:
        predicted_move_pct: Model's predicted daily move (%).
        actual_move_pct: Observed daily move (%).
    
    Returns:
        Absolute residual in percentage points.
    """
    return abs(actual_move_pct - predicted_move_pct)


def flag_anomaly(
    ticker: str,
    predicted_move_pct: float,
    actual_move_pct: float,
    multiplier: float = 2.0,
    min_threshold_pct: float = 0.5,
) -> Optional[AnomalyAlert]:
    """
    Check if actual move exceeds 2x predicted residual; return AnomalyAlert if so.
    
    Args:
        ticker: Stock symbol.
        predicted_move_pct: Model's predicted move (%).
        actual_move_pct: Observed move (%).
        multiplier: Factor by which residual must exceed abs(predicted) (default 2.0).
        min_threshold_pct: Minimum threshold to avoid flagging tiny residuals (default 0.5%).
    
    Returns:
        AnomalyAlert if threshold exceeded, else None.
    """
    residual = compute_residual(predicted_move_pct, actual_move_pct)
    threshold = max(multiplier * abs(predicted_move_pct), min_threshold_pct)
    
    if residual > threshold:
        # Determine severity
        severity_ratio = residual / threshold
        if severity_ratio > 3.0:
            severity = "high"
            confidence = 0.95
        elif severity_ratio > 2.0:
            severity = "medium"
            confidence = 0.85
        else:
            severity = "low"
            confidence = 0.70
        
        # Reason
        if abs(predicted_move_pct) < 0.1:
            reason = "Expected near-zero move; significant surprise occurred."
        elif (predicted_move_pct > 0 and actual_move_pct < -1.0) or \
             (predicted_move_pct < 0 and actual_move_pct > 1.0):
            reason = "Prediction direction completely reversed."
        else:
            reason = f"Move magnitude {severity_ratio:.1f}x larger than threshold."
        
        return AnomalyAlert(
            ticker=ticker,
            date=datetime.utcnow(),
            predicted_move_pct=predicted_move_pct,
            actual_move_pct=actual_move_pct,
            residual_pct=residual,
            threshold_pct=threshold,
            severity=severity,
            reason=reason,
            confidence=confidence,
        )
    
    return None


def batch_flag_anomalies(
    predictions: list[dict],
    multiplier: float = 2.0,
    min_threshold_pct: float = 0.5,
) -> list[AnomalyAlert]:
    """
    Batch anomaly detection across multiple predictions.
    
    Args:
        predictions: List of dicts with keys: ticker, predicted_move_pct, actual_move_pct.
        multiplier: Threshold multiplier (default 2.0).
        min_threshold_pct: Minimum threshold (default 0.5%).
    
    Returns:
        List of AnomalyAlert objects.
    """
    alerts = []
    for pred in predictions:
        alert = flag_anomaly(
            ticker=pred["ticker"],
            predicted_move_pct=pred["predicted_move_pct"],
            actual_move_pct=pred["actual_move_pct"],
            multiplier=multiplier,
            min_threshold_pct=min_threshold_pct,
        )
        if alert:
            alerts.append(alert)
    return alerts


def summarize_anomalies(alerts: list[AnomalyAlert]) -> dict:
    """
    Summarize anomaly alerts into counts and top severity cases.
    
    Args:
        alerts: List of AnomalyAlert objects.
    
    Returns:
        Dict with total count, counts by severity, and top 3 highest-residual alerts.
    """
    if not alerts:
        return {
            "total": 0,
            "by_severity": {"high": 0, "medium": 0, "low": 0},
            "top_residuals": [],
        }
    
    by_sev = {"high": 0, "medium": 0, "low": 0}
    for alert in alerts:
        by_sev[alert.severity] += 1
    
    top_residuals = sorted(alerts, key=lambda a: a.residual_pct, reverse=True)[:3]
    
    return {
        "total": len(alerts),
        "by_severity": by_sev,
        "top_residuals": [
            {
                "ticker": a.ticker,
                "residual_pct": a.residual_pct,
                "severity": a.severity,
                "reason": a.reason,
            }
            for a in top_residuals
        ],
    }
