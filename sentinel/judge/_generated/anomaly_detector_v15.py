"""
Anomaly Detection System for Sentinel Sentiment Engine.

This module detects when actual market moves deviate significantly from predicted
moves, flagging outlier trading days for post-mortem analysis and model refinement.
It generates AnomalyAlert dataclasses that feed into the Judge's daily calibration
loop, helping identify market regime shifts, surprise earnings, or systematic
prediction errors.

Part of sentinel/judge/ — anomaly detection & heuristic refinement.
"""

import dataclasses
from datetime import datetime, timedelta
from typing import Optional
import sqlite3
import numpy as np


@dataclasses.dataclass
class AnomalyAlert:
    """Flagged discrepancy between predicted and actual market move."""

    ticker: str
    prediction_date: str  # YYYY-MM-DD
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"
    predicted_magnitude: float  # percentage, e.g., 2.5
    actual_move: float  # percentage, e.g., -5.8
    residual: float  # actual - predicted, e.g., -8.3
    anomaly_ratio: float  # |residual| / (|predicted_magnitude| + 0.1), e.g., 3.32
    severity: str  # "MINOR" (<2.0x), "MODERATE" (2–4x), "SEVERE" (>4x)
    alert_time: str  # ISO timestamp when flagged
    context: Optional[str] = None  # e.g., "earnings surprise", "gap event"


def compute_residual(
    predicted_move: float, actual_move: float
) -> tuple[float, float]:
    """
    Compute residual and anomaly ratio.

    Returns:
        (residual, anomaly_ratio) where ratio = |residual| / (|predicted| + 0.1)
    """
    residual = actual_move - predicted_move
    # Add 0.1 to avoid division by zero on near-zero predictions
    anomaly_ratio = abs(residual) / (abs(predicted_move) + 0.1)
    return residual, anomaly_ratio


def classify_severity(anomaly_ratio: float) -> str:
    """Classify anomaly severity by residual magnitude threshold."""
    if anomaly_ratio < 2.0:
        return "MINOR"
    elif anomaly_ratio < 4.0:
        return "MODERATE"
    else:
        return "SEVERE"


def flag_anomalies(
    predictions: dict,
    actuals: dict,
    threshold_ratio: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Scan predictions vs. actuals and flag moves exceeding threshold.

    Args:
        predictions: dict with keys (ticker, date) -> {"direction": str, "magnitude": float}
        actuals: dict with keys (ticker, date) -> float (% move)
        threshold_ratio: anomaly_ratio threshold for flagging (default 2.0x)

    Returns:
        List of AnomalyAlert objects for moves exceeding threshold.
    """
    alerts = []
    alert_time = datetime.utcnow().isoformat() + "Z"

    for (ticker, pred_date), pred_data in predictions.items():
        key = (ticker, pred_date)
        if key not in actuals:
            continue  # No actual data yet

        actual_move = actuals[key]
        predicted_mag = pred_data.get("magnitude", 0.0)
        predicted_dir = pred_data.get("direction", "NEUTRAL")

        residual, anomaly_ratio = compute_residual(predicted_mag, actual_move)

        if anomaly_ratio >= threshold_ratio:
            severity = classify_severity(anomaly_ratio)
            alert = AnomalyAlert(
                ticker=ticker,
                prediction_date=pred_date,
                predicted_direction=predicted_dir,
                predicted_magnitude=predicted_mag,
                actual_move=actual_move,
                residual=residual,
                anomaly_ratio=anomaly_ratio,
                severity=severity,
                alert_time=alert_time,
            )
            alerts.append(alert)

    return sorted(alerts, key=lambda a: a.anomaly_ratio, reverse=True)


def ingest_anomalies_to_sqlite(
    alerts: list[AnomalyAlert], db_path: str
) -> int:
    """
    Persist anomaly alerts to SQLite for historical review.

    Args:
        alerts: List of AnomalyAlert objects
        db_path: Path to SQLite database

    Returns:
        Number of rows inserted
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            predicted_magnitude REAL,
            actual_move REAL,
            residual REAL,
            anomaly_ratio REAL,
            severity TEXT,
            alert_time TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    inserted = 0
    for alert in alerts:
        cursor.execute(
            """
            INSERT INTO anomaly_alerts (
                ticker, prediction_date, predicted_direction, predicted_magnitude,
                actual_move, residual, anomaly_ratio, severity, alert_time, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.ticker,
                alert.prediction_date,
                alert.predicted_direction,
                alert.predicted_magnitude,
                alert.actual_move,
                alert.residual,
                alert.anomaly_ratio,
                alert.severity,
                alert.alert_time,
                alert.context,
            ),
        )
        inserted += 1

    conn.commit()
    conn.close()
    return inserted


def query_recent_anomalies(
    db_path: str, days: int = 7, severity_filter: Optional[str] = None
) -> list[AnomalyAlert]:
    """
    Retrieve recent anomalies from SQLite for analysis.

    Args:
        db_path: Path to SQLite database
        days: Look back N days (default 7)
        severity_filter: Filter by severity ("MINOR", "MODERATE", "SEVERE") or None

    Returns:
        List of AnomalyAlert objects matching criteria
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    query = "SELECT * FROM anomaly_alerts WHERE alert_time >= ?"
    params: list = [cutoff_date]

    if severity_filter:
        query += " AND severity = ?"
        params.append(severity_filter)

    query += " ORDER BY alert_time DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    alerts = []
    if rows:
        columns = [desc[0] for desc in cursor.description]
        for row in rows:
            row_dict = dict(zip(columns, row))
            alert = AnomalyAlert(
                ticker=row_dict["ticker"],
                prediction_date=row_dict["prediction_date"],
                predicted_direction=row_dict["predicted_direction"],
                predicted_magnitude=row_dict["predicted_magnitude"],
                actual_move=row_dict["actual_move"],
                residual=row_dict["residual"],
                anomaly_ratio=row_dict["anomaly_ratio"],
                severity=row_dict["severity"],
                alert_time=row_dict["alert_time"],
                context=row_dict.get("context"),
            )
            alerts.append(alert)

    return alerts


def compute_anomaly_stats(alerts: list[AnomalyAlert]) -> dict:
    """
    Compute summary statistics from anomaly alerts.

    Args:
        alerts: List of AnomalyAlert objects

    Returns:
        Dict
