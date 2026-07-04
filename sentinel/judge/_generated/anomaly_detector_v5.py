"""
Anomaly Detector for Sentinel Sentiment Engine.

This module detects when actual market moves significantly exceed predicted
residuals (2x threshold). It generates AnomalyAlert dataclasses for the Judge
pillar to flag unexpected market behavior for post-mortem analysis and model
refinement. Feeds into judge/postmortem.py for daily reconciliation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlite3
import os


@dataclass
class AnomalyAlert:
    """Represents a detected anomaly in market movement vs. prediction."""
    
    ticker: str
    prediction_date: str
    actual_move_pct: float
    predicted_move_pct: float
    residual_pct: float
    anomaly_ratio: float  # actual / predicted
    severity: str  # "low", "medium", "high"
    alert_timestamp: str
    notes: str


def calculate_residual(actual_move: float, predicted_move: float) -> float:
    """Calculate absolute residual between actual and predicted move."""
    return abs(actual_move - predicted_move)


def classify_severity(anomaly_ratio: float) -> str:
    """Classify anomaly severity based on ratio of actual to predicted residual."""
    if anomaly_ratio >= 4.0:
        return "high"
    elif anomaly_ratio >= 2.5:
        return "medium"
    else:
        return "low"


def detect_anomalies(
    ticker: str,
    actual_move_pct: float,
    predicted_move_pct: float,
    prediction_date: str,
    threshold_multiplier: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds predicted residual by threshold_multiplier.
    
    Returns AnomalyAlert if detected, None otherwise.
    """
    residual = calculate_residual(actual_move_pct, predicted_move_pct)
    
    # Avoid division by zero; treat tiny predicted moves as baseline
    predicted_abs = abs(predicted_move_pct)
    if predicted_abs < 0.01:
        predicted_abs = 0.01
    
    anomaly_ratio = residual / predicted_abs
    
    if anomaly_ratio >= threshold_multiplier:
        severity = classify_severity(anomaly_ratio)
        alert = AnomalyAlert(
            ticker=ticker,
            prediction_date=prediction_date,
            actual_move_pct=round(actual_move_pct, 3),
            predicted_move_pct=round(predicted_move_pct, 3),
            residual_pct=round(residual, 3),
            anomaly_ratio=round(anomaly_ratio, 2),
            severity=severity,
            alert_timestamp=datetime.utcnow().isoformat(),
            notes=f"Actual move {anomaly_ratio:.1f}x larger than predicted residual.",
        )
        return alert
    
    return None


def store_anomaly_alert(alert: AnomalyAlert, db_path: str = "sentinel_anomalies.db") -> None:
    """Store anomaly alert in SQLite for historical tracking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT,
            actual_move_pct REAL,
            predicted_move_pct REAL,
            residual_pct REAL,
            anomaly_ratio REAL,
            severity TEXT,
            alert_timestamp TEXT,
            notes TEXT
        )
    """)
    
    cursor.execute("""
        INSERT INTO anomaly_alerts (
            ticker, prediction_date, actual_move_pct, predicted_move_pct,
            residual_pct, anomaly_ratio, severity, alert_timestamp, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.ticker,
        alert.prediction_date,
        alert.actual_move_pct,
        alert.predicted_move_pct,
        alert.residual_pct,
        alert.anomaly_ratio,
        alert.severity,
        alert.alert_timestamp,
        alert.notes,
    ))
    
    conn.commit()
    conn.close()


def retrieve_anomalies(
    ticker: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    db_path: str = "sentinel_anomalies.db",
) -> list[AnomalyAlert]:
    """Retrieve anomaly alerts from database with optional filtering."""
    if not os.path.exists(db_path):
        return []
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM anomaly_alerts WHERE 1=1"
    params = []
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    query += " ORDER BY alert_timestamp DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    alerts = []
    for row in rows:
        alert = AnomalyAlert(
            ticker=row[1],
            prediction_date=row[2],
            actual_move_pct=row[3],
            predicted_move_pct=row[4],
            residual_pct=row[5],
            anomaly_ratio=row[6],
            severity=row[7],
            alert_timestamp=row[8],
            notes=row[9],
        )
        alerts.append(alert)
    
    return alerts


def batch_detect_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    threshold_multiplier: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Batch-detect anomalies across multiple tickers.
    
    predictions: list of dicts with keys ticker, move_pct, date
    actuals: list of dicts with keys ticker, move_pct, date
    """
    alerts = []
    
    # Build lookup for actual moves
    actual_map = {(a["ticker"], a["date"]): a["move_pct"] for a in actuals}
    
    for pred in predictions:
        key = (pred["ticker"], pred["date"])
        if key in actual_map:
            alert = detect_anomalies(
                ticker=pred["ticker"],
                actual_move_pct=actual_map[key],
                predicted_move_pct=pred["move_pct"],
                prediction_date=pred["date"],
                threshold_multiplier=threshold_multiplier,
            )
            if alert:
                alerts.append(alert)
    
    return alerts
