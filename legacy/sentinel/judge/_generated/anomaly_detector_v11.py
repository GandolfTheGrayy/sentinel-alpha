"""
Anomaly flagging system for Sentinel Sentiment Engine.

Detects when actual market moves exceed 2x the predicted residual and generates
AnomalyAlert dataclasses. Used by judge/postmortem.py to identify prediction
failures and trigger heuristic refinement in the daily post-mortem cycle.

This module compares predicted price deltas against realized moves, computes
residuals, and flags outliers for manual review and model calibration.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """
    Represents a detected anomaly between predicted and actual market moves.
    """
    ticker: str
    prediction_date: str
    predicted_delta_pct: float
    actual_delta_pct: float
    residual_pct: float
    anomaly_ratio: float
    severity: str
    message: str
    flagged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert AnomalyAlert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date,
            "predicted_delta_pct": self.predicted_delta_pct,
            "actual_delta_pct": self.actual_delta_pct,
            "residual_pct": self.residual_pct,
            "anomaly_ratio": self.anomaly_ratio,
            "severity": self.severity,
            "message": self.message,
            "flagged_at": self.flagged_at,
            "details": self.details,
        }


def compute_residual(predicted_delta_pct: float, actual_delta_pct: float) -> float:
    """
    Compute the residual error between predicted and actual price delta.
    """
    return actual_delta_pct - predicted_delta_pct


def compute_anomaly_ratio(residual_pct: float, predicted_delta_pct: float) -> float:
    """
    Compute the anomaly ratio: how many times the residual exceeds predicted magnitude.
    Returns absolute ratio; handles zero-prediction edge case by returning high ratio.
    """
    if abs(predicted_delta_pct) < 1e-6:
        return abs(residual_pct) * 100.0 if abs(residual_pct) > 1e-6 else 0.0
    return abs(residual_pct) / abs(predicted_delta_pct)


def classify_severity(anomaly_ratio: float) -> str:
    """
    Classify anomaly severity based on anomaly ratio threshold.
    Ratio >= 2.0 is flagged as anomalous.
    """
    if anomaly_ratio >= 5.0:
        return "CRITICAL"
    elif anomaly_ratio >= 3.0:
        return "HIGH"
    elif anomaly_ratio >= 2.0:
        return "MEDIUM"
    else:
        return "LOW"


def detect_anomaly(
    ticker: str,
    prediction_date: str,
    predicted_delta_pct: float,
    actual_delta_pct: float,
    threshold_ratio: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds predicted residual by threshold_ratio multiple.
    Returns AnomalyAlert if anomaly detected, None otherwise.
    """
    residual = compute_residual(predicted_delta_pct, actual_delta_pct)
    ratio = compute_anomaly_ratio(residual, predicted_delta_pct)

    if ratio >= threshold_ratio:
        severity = classify_severity(ratio)
        message = (
            f"{ticker}: predicted {predicted_delta_pct:+.2f}%, "
            f"actual {actual_delta_pct:+.2f}%, "
            f"residual {residual:+.2f}% ({ratio:.2f}x magnitude)"
        )
        return AnomalyAlert(
            ticker=ticker,
            prediction_date=prediction_date,
            predicted_delta_pct=predicted_delta_pct,
            actual_delta_pct=actual_delta_pct,
            residual_pct=residual,
            anomaly_ratio=ratio,
            severity=severity,
            message=message,
            details={
                "threshold_ratio": threshold_ratio,
                "direction_match": (predicted_delta_pct * actual_delta_pct) > 0,
            },
        )
    return None


def batch_detect_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    threshold_ratio: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Detect anomalies across a batch of predictions vs. actuals.
    predictions and actuals are dicts with keys: ticker, date, delta_pct.
    """
    alerts = []
    pred_by_key = {(p["ticker"], p["date"]): p for p in predictions}
    
    for actual in actuals:
        key = (actual["ticker"], actual["date"])
        if key in pred_by_key:
            pred = pred_by_key[key]
            alert = detect_anomaly(
                ticker=actual["ticker"],
                prediction_date=actual["date"],
                predicted_delta_pct=pred["delta_pct"],
                actual_delta_pct=actual["delta_pct"],
                threshold_ratio=threshold_ratio,
            )
            if alert:
                alerts.append(alert)
    
    return alerts


def store_anomaly_alert(
    db_path: str,
    alert: AnomalyAlert,
) -> None:
    """
    Store an AnomalyAlert in SQLite for audit trail and historical analysis.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_delta_pct REAL,
            actual_delta_pct REAL,
            residual_pct REAL,
            anomaly_ratio REAL,
            severity TEXT,
            message TEXT,
            flagged_at TEXT,
            details TEXT
        )
    """)
    
    cursor.execute("""
        INSERT INTO anomaly_alerts
        (ticker, prediction_date, predicted_delta_pct, actual_delta_pct,
         residual_pct, anomaly_ratio, severity, message, flagged_at, details)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.ticker,
        alert.prediction_date,
        alert.predicted_delta_pct,
        alert.actual_delta_pct,
        alert.residual_pct,
        alert.anomaly_ratio,
        alert.severity,
        alert.message,
        alert.flagged_at,
        json.dumps(alert.details),
    ))
    
    conn.commit()
    conn.close()


def load_anomaly_alerts(
    db_path: str,
    ticker: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
) -> list[AnomalyAlert]:
    """
    Load anomaly alerts from SQLite with optional filtering by ticker or severity.
    """
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
    
    query += " ORDER BY flagged_at DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows =
