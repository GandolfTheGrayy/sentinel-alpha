"""
Anomaly detection system for Sentinel Judge pillar.

Detects when actual market moves exceed 2x the predicted residual, generating
AnomalyAlert dataclasses for post-mortem analysis and heuristic refinement.
Integrates with the daily resolver to flag outlier market behavior.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlite3


@dataclass
class AnomalyAlert:
    """Dataclass representing a detected market anomaly."""
    ticker: str
    prediction_date: str
    predicted_move_pct: float
    actual_move_pct: float
    residual_pct: float
    anomaly_factor: float
    severity: str
    alert_reason: str
    generated_at: str


def calculate_residual(predicted_move: float, actual_move: float) -> float:
    """Calculate prediction residual as (actual - predicted)."""
    return actual_move - predicted_move


def detect_anomaly(
    ticker: str,
    prediction_date: str,
    predicted_move_pct: float,
    actual_move_pct: float,
    anomaly_threshold: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds 2x the predicted residual magnitude.
    
    Returns AnomalyAlert if anomaly detected, None otherwise.
    """
    residual = calculate_residual(predicted_move_pct, actual_move_pct)
    
    if residual == 0:
        return None
    
    anomaly_factor = abs(actual_move_pct) / (abs(residual) + 1e-6)
    
    if anomaly_factor < anomaly_threshold:
        return None
    
    # Determine severity based on anomaly factor
    if anomaly_factor >= 3.0:
        severity = "CRITICAL"
    elif anomaly_factor >= 2.5:
        severity = "HIGH"
    else:
        severity = "MEDIUM"
    
    # Generate alert reason
    direction = "up" if actual_move_pct > 0 else "down"
    alert_reason = (
        f"Actual move {direction} {abs(actual_move_pct):.2f}% "
        f"exceeded predicted {predicted_move_pct:.2f}% by {anomaly_factor:.1f}x"
    )
    
    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_move_pct=predicted_move_pct,
        actual_move_pct=actual_move_pct,
        residual_pct=residual,
        anomaly_factor=anomaly_factor,
        severity=severity,
        alert_reason=alert_reason,
        generated_at=datetime.utcnow().isoformat(),
    )


def batch_detect_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    anomaly_threshold: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Detect anomalies across a batch of ticker predictions and actuals.
    
    Each prediction dict must have: ticker, prediction_date, predicted_move_pct.
    Each actual dict must have: ticker, prediction_date, actual_move_pct.
    """
    alerts = []
    
    # Index actuals by (ticker, prediction_date) for O(1) lookup
    actual_map = {
        (a["ticker"], a["prediction_date"]): a["actual_move_pct"]
        for a in actuals
    }
    
    for pred in predictions:
        key = (pred["ticker"], pred["prediction_date"])
        if key not in actual_map:
            continue
        
        alert = detect_anomaly(
            ticker=pred["ticker"],
            prediction_date=pred["prediction_date"],
            predicted_move_pct=pred["predicted_move_pct"],
            actual_move_pct=actual_map[key],
            anomaly_threshold=anomaly_threshold,
        )
        
        if alert:
            alerts.append(alert)
    
    return alerts


def store_anomaly_alert(
    alert: AnomalyAlert,
    db_path: str = "sentinel.db",
) -> None:
    """Store AnomalyAlert to SQLite for post-mortem analysis."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_move_pct REAL,
            actual_move_pct REAL,
            residual_pct REAL,
            anomaly_factor REAL,
            severity TEXT,
            alert_reason TEXT,
            generated_at TEXT,
            UNIQUE(ticker, prediction_date)
        )
        """
    )
    
    cursor.execute(
        """
        INSERT OR REPLACE INTO anomaly_alerts
        (ticker, prediction_date, predicted_move_pct, actual_move_pct,
         residual_pct, anomaly_factor, severity, alert_reason, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert.ticker,
            alert.prediction_date,
            alert.predicted_move_pct,
            alert.actual_move_pct,
            alert.residual_pct,
            alert.anomaly_factor,
            alert.severity,
            alert.alert_reason,
            alert.generated_at,
        ),
    )
    
    conn.commit()
    conn.close()


def retrieve_anomalies(
    db_path: str = "sentinel.db",
    severity_filter: Optional[str] = None,
    limit: int = 100,
) -> list[AnomalyAlert]:
    """Retrieve stored anomaly alerts from SQLite, optionally filtered by severity."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if severity_filter:
        cursor.execute(
            """
            SELECT ticker, prediction_date, predicted_move_pct, actual_move_pct,
                   residual_pct, anomaly_factor, severity, alert_reason, generated_at
            FROM anomaly_alerts
            WHERE severity = ?
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            (severity_filter, limit),
        )
    else:
        cursor.execute(
            """
            SELECT ticker, prediction_date, predicted_move_pct, actual_move_pct,
                   residual_pct, anomaly_factor, severity, alert_reason, generated_at
            FROM anomaly_alerts
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    alerts = [
        AnomalyAlert(
            ticker=row[0],
            prediction_date=row[1],
            predicted_move_pct=row[2],
            actual_move_pct=row[3],
            residual_pct=row[4],
            anomaly_factor=row[5],
            severity=row[6],
            alert_reason=row[7],
            generated_at=row[8],
        )
        for row in rows
    ]
    
    return alerts


def summarize_anomalies(
    alerts: list[AnomalyAlert],
) -> dict:
    """Generate summary statistics across anomaly alerts."""
    if not alerts:
        return {
            "total_anomalies": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "avg_anomaly_factor": 0.0,
            "max_anomaly_factor": 0.0,
        }
    
    severity_counts = {
        "CRITICAL": sum(1 for a in alerts if a.severity == "CRITICAL"),
        "HIGH": sum
