"""
Anomaly detection system for Sentinel Sentiment Engine.

Detects when actual market moves significantly exceed predicted residuals (>2x),
generates AnomalyAlert dataclasses, and logs flagged tickers for manual review.
Integrates with Judge postmortem workflow to identify model blind spots and
trigger heuristic refinement in the daily calibration cycle.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """Triggered when actual move exceeds 2x predicted residual."""
    
    ticker: str
    prediction_date: str
    predicted_move_pct: float
    predicted_residual_pct: float
    actual_move_pct: float
    anomaly_ratio: float
    alert_severity: str  # "moderate" | "high" | "extreme"
    triggered_at: str
    notes: Optional[str] = None


def compute_residual(
    predicted_move_pct: float,
    historical_volatility_pct: float,
) -> float:
    """
    Compute predicted residual as (predicted_move - historical_vol * 0.5).
    
    Represents the signal magnitude beyond baseline market noise.
    """
    baseline_noise = historical_volatility_pct * 0.5
    residual = abs(predicted_move_pct) - baseline_noise
    return max(residual, 0.01)  # Floor at 0.01% to avoid division by zero


def detect_anomalies(
    ticker: str,
    predicted_move_pct: float,
    actual_move_pct: float,
    historical_volatility_pct: float,
    prediction_date: str,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds 2x the predicted residual; return AnomalyAlert or None.
    
    Args:
        ticker: Stock ticker symbol.
        predicted_move_pct: Model's predicted price change (%).
        actual_move_pct: Observed price change (%).
        historical_volatility_pct: 30-day realized volatility (%).
        prediction_date: ISO date string of prediction.
    
    Returns:
        AnomalyAlert if anomaly detected, None otherwise.
    """
    residual = compute_residual(predicted_move_pct, historical_volatility_pct)
    anomaly_threshold = residual * 2.0
    actual_abs = abs(actual_move_pct)
    
    if actual_abs <= anomaly_threshold:
        return None
    
    anomaly_ratio = actual_abs / residual if residual > 0 else 0.0
    
    if anomaly_ratio > 3.0:
        severity = "extreme"
    elif anomaly_ratio > 2.5:
        severity = "high"
    else:
        severity = "moderate"
    
    notes = (
        f"Predicted move: {predicted_move_pct:+.2f}%, "
        f"Actual: {actual_move_pct:+.2f}%, "
        f"Residual: {residual:.2f}%, "
        f"Ratio: {anomaly_ratio:.2f}x"
    )
    
    alert = AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_move_pct=predicted_move_pct,
        predicted_residual_pct=residual,
        actual_move_pct=actual_move_pct,
        anomaly_ratio=anomaly_ratio,
        alert_severity=severity,
        triggered_at=datetime.utcnow().isoformat(),
        notes=notes,
    )
    return alert


def log_anomaly_alert(
    alert: AnomalyAlert,
    db_path: str = "sentinel_anomalies.db",
) -> None:
    """
    Persist AnomalyAlert to SQLite for audit trail and pattern analysis.
    
    Args:
        alert: AnomalyAlert dataclass instance.
        db_path: Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomalies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            prediction_date TEXT,
            predicted_move_pct REAL,
            predicted_residual_pct REAL,
            actual_move_pct REAL,
            anomaly_ratio REAL,
            alert_severity TEXT,
            triggered_at TEXT,
            notes TEXT
        )
    """)
    
    cursor.execute("""
        INSERT INTO anomalies (
            ticker, prediction_date, predicted_move_pct, predicted_residual_pct,
            actual_move_pct, anomaly_ratio, alert_severity, triggered_at, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.ticker,
        alert.prediction_date,
        alert.predicted_move_pct,
        alert.predicted_residual_pct,
        alert.actual_move_pct,
        alert.anomaly_ratio,
        alert.alert_severity,
        alert.triggered_at,
        alert.notes,
    ))
    
    conn.commit()
    conn.close()


def batch_anomaly_detection(
    results: list[dict],
    db_path: str = "sentinel_anomalies.db",
) -> list[AnomalyAlert]:
    """
    Process batch of prediction+actual pairs; return list of triggered alerts.
    
    Args:
        results: List of dicts with keys: ticker, predicted_move_pct, actual_move_pct,
                 historical_volatility_pct, prediction_date.
        db_path: Path to anomaly log database.
    
    Returns:
        List of AnomalyAlert instances (empty if none triggered).
    """
    alerts = []
    
    for record in results:
        alert = detect_anomalies(
            ticker=record["ticker"],
            predicted_move_pct=record["predicted_move_pct"],
            actual_move_pct=record["actual_move_pct"],
            historical_volatility_pct=record["historical_volatility_pct"],
            prediction_date=record["prediction_date"],
        )
        
        if alert:
            alerts.append(alert)
            log_anomaly_alert(alert, db_path)
    
    return alerts


def summarize_anomalies(
    alerts: list[AnomalyAlert],
) -> dict:
    """
    Generate summary stats from anomaly batch for daily postmortem.
    
    Args:
        alerts: List of AnomalyAlert instances.
    
    Returns:
        Dict with keys: total_count, by_severity, avg_ratio, extreme_tickers.
    """
    if not alerts:
        return {
            "total_count": 0,
            "by_severity": {"moderate": 0, "high": 0, "extreme": 0},
            "avg_ratio": 0.0,
            "extreme_tickers": [],
        }
    
    by_severity = {"moderate": 0, "high": 0, "extreme": 0}
    extreme_tickers = []
    
    for alert in alerts:
        by_severity[alert.alert_severity] += 1
        if alert.alert_severity == "extreme":
            extreme_tickers.append({
                "ticker": alert.ticker,
                "ratio": alert.anomaly_ratio,
                "actual": alert.actual_move_pct,
                "predicted": alert.predicted_move_pct,
            })
    
    avg_ratio = sum(a.anomaly_ratio for a in alerts) / len(alerts)
    
    extreme_tickers.sort(key=lambda x: x["ratio"], reverse=True)
    
    return {
        "total_count": len(alerts),
        "by_severity": by_severity,
        "avg_ratio": avg_ratio,
        "extreme_tickers": extreme_tickers[:10
