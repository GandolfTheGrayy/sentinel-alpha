"""
Sentinel Anomaly Detector — Detects when actual market moves exceed 2x the predicted residual.

This module compares predicted price movements (from Judge.predictor) against
actual market outcomes. When the absolute actual move exceeds 2x the predicted
residual (prediction error), an AnomalyAlert is raised and logged.

Integrates with Judge post-mortem pipeline to flag unexpected market behavior
for heuristic refinement and investigative review.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """Represents a detected anomaly when actual move >> predicted move."""
    
    ticker: str
    prediction_date: str
    predicted_move_pct: float
    actual_move_pct: float
    predicted_residual_pct: float
    anomaly_ratio: float
    threshold_multiple: float
    alert_level: str
    summary: str
    timestamp: str


def create_anomaly_alert(
    ticker: str,
    prediction_date: str,
    predicted_move_pct: float,
    actual_move_pct: float,
    predicted_residual_pct: float,
    threshold_multiple: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Create an AnomalyAlert if actual move exceeds threshold_multiple × residual.
    
    Returns None if no anomaly detected; otherwise returns fully populated AnomalyAlert.
    """
    if predicted_residual_pct == 0.0:
        predicted_residual_pct = 0.01
    
    anomaly_ratio = abs(actual_move_pct) / abs(predicted_residual_pct)
    
    if anomaly_ratio < threshold_multiple:
        return None
    
    if anomaly_ratio >= 5.0:
        alert_level = "CRITICAL"
    elif anomaly_ratio >= 3.0:
        alert_level = "HIGH"
    else:
        alert_level = "MEDIUM"
    
    summary = (
        f"{ticker} {alert_level}: actual move {actual_move_pct:.2f}% "
        f"vs predicted {predicted_move_pct:.2f}% (residual {predicted_residual_pct:.2f}%, "
        f"ratio {anomaly_ratio:.1f}x threshold)"
    )
    
    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_move_pct=predicted_move_pct,
        actual_move_pct=actual_move_pct,
        predicted_residual_pct=predicted_residual_pct,
        anomaly_ratio=anomaly_ratio,
        threshold_multiple=threshold_multiple,
        alert_level=alert_level,
        summary=summary,
        timestamp=datetime.utcnow().isoformat(),
    )


def log_anomaly_alert(alert: AnomalyAlert, db_path: str = "sentinel.db") -> None:
    """Log an AnomalyAlert to SQLite for post-mortem review and heuristic refinement."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_move_pct REAL NOT NULL,
            actual_move_pct REAL NOT NULL,
            predicted_residual_pct REAL NOT NULL,
            anomaly_ratio REAL NOT NULL,
            threshold_multiple REAL NOT NULL,
            alert_level TEXT NOT NULL,
            summary TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """
    )
    
    cursor.execute(
        """
        INSERT INTO anomaly_alerts (
            ticker, prediction_date, predicted_move_pct, actual_move_pct,
            predicted_residual_pct, anomaly_ratio, threshold_multiple,
            alert_level, summary, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert.ticker,
            alert.prediction_date,
            alert.predicted_move_pct,
            alert.actual_move_pct,
            alert.predicted_residual_pct,
            alert.anomaly_ratio,
            alert.threshold_multiple,
            alert.alert_level,
            alert.summary,
            alert.timestamp,
        ),
    )
    
    conn.commit()
    conn.close()


def fetch_anomaly_alerts(
    db_path: str = "sentinel.db",
    limit: int = 100,
    alert_level: Optional[str] = None,
) -> list[AnomalyAlert]:
    """Fetch logged anomaly alerts from SQLite, optionally filtered by alert_level."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if alert_level:
        cursor.execute(
            """
            SELECT * FROM anomaly_alerts
            WHERE alert_level = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (alert_level, limit),
        )
    else:
        cursor.execute(
            """
            SELECT * FROM anomaly_alerts
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = [
        AnomalyAlert(
            ticker=row["ticker"],
            prediction_date=row["prediction_date"],
            predicted_move_pct=row["predicted_move_pct"],
            actual_move_pct=row["actual_move_pct"],
            predicted_residual_pct=row["predicted_residual_pct"],
            anomaly_ratio=row["anomaly_ratio"],
            threshold_multiple=row["threshold_multiple"],
            alert_level=row["alert_level"],
            summary=row["summary"],
            timestamp=row["timestamp"],
        )
        for row in rows
    ]
    
    return alerts


def batch_check_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    threshold_multiple: float = 2.0,
    db_path: str = "sentinel.db",
) -> list[AnomalyAlert]:
    """
    Batch-check predictions vs actuals; log and return all detected anomalies.
    
    Each item in predictions and actuals should have keys:
      ticker, prediction_date, predicted_move_pct, actual_move_pct, predicted_residual_pct
    """
    alerts = []
    
    actual_map = {(a["ticker"], a["prediction_date"]): a for a in actuals}
    
    for pred in predictions:
        key = (pred["ticker"], pred["prediction_date"])
        if key not in actual_map:
            continue
        
        actual = actual_map[key]
        
        alert = create_anomaly_alert(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_move_pct=pred["predicted_move_pct"],
            actual_move_pct=actual["actual_move_pct"],
            predicted_residual_pct=pred.get("predicted_residual_pct", 0.0),
            threshold_multiple=threshold_multiple,
        )
        
        if alert:
            log_anomaly_alert(alert, db_path=db_path)
            alerts.append(alert)
    
    return alerts


def summarize_anomalies(alerts: list[AnomalyAlert]) -> dict:
    """Generate a summary dict of anomaly detection results."""
    if not alerts:
        return {
            "total_anomalies": 0,
            "by_level": {},
            "avg_anomaly_ratio": 0.0,
            "max_anomaly_ratio": 0.0,
        }
    
    by_level = {}
    for alert in alerts:
