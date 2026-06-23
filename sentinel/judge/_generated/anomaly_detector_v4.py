"""
Anomaly Detector — Sentinel Judge pillar module for flagging unexpected market moves.

This module compares predicted vs. actual price movements and raises AnomalyAlert
when realized volatility exceeds 2x the predicted residual. Used by the daily
post-mortem pipeline to identify prediction failures, model drift, or exogenous shocks.
"""

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import json


@dataclass
class AnomalyAlert:
    """
    Represents a detected anomaly when actual market move exceeds prediction bounds.
    
    Attributes:
        ticker: Stock symbol.
        prediction_date: Date prediction was made (ISO format).
        predicted_direction: Predicted move direction ('UP', 'DOWN', 'NEUTRAL').
        predicted_confidence: Confidence score [0.0, 1.0].
        actual_move_pct: Realized price change as percentage.
        predicted_residual: Expected volatility bound (e.g., 2% move).
        anomaly_ratio: Actual move / predicted residual (threshold: >2.0).
        severity: Categorical severity ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        alert_reason: Human-readable explanation.
        generated_at: ISO timestamp when alert was created.
    """
    ticker: str
    prediction_date: str
    predicted_direction: str
    predicted_confidence: float
    actual_move_pct: float
    predicted_residual: float
    anomaly_ratio: float
    severity: str
    alert_reason: str
    generated_at: str


def classify_severity(anomaly_ratio: float, confidence: float) -> str:
    """Classify anomaly severity based on ratio and prediction confidence."""
    if anomaly_ratio < 2.0:
        return "LOW"
    if anomaly_ratio >= 2.0 and anomaly_ratio < 3.5:
        return "MEDIUM" if confidence > 0.7 else "HIGH"
    if anomaly_ratio >= 3.5 and anomaly_ratio < 5.0:
        return "HIGH"
    return "CRITICAL"


def detect_anomaly(
    ticker: str,
    prediction_date: str,
    predicted_direction: str,
    predicted_confidence: float,
    actual_move_pct: float,
    predicted_residual: float,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual market move constitutes an anomaly vs. prediction.
    
    Returns AnomalyAlert if anomaly_ratio > 2.0, else None.
    """
    if predicted_residual <= 0:
        return None
    
    anomaly_ratio = abs(actual_move_pct) / predicted_residual
    
    if anomaly_ratio <= 2.0:
        return None
    
    severity = classify_severity(anomaly_ratio, predicted_confidence)
    
    direction_match = (
        (predicted_direction == "UP" and actual_move_pct > 0)
        or (predicted_direction == "DOWN" and actual_move_pct < 0)
        or (predicted_direction == "NEUTRAL")
    )
    
    if direction_match:
        reason = (
            f"High-confidence ({predicted_confidence:.0%}) prediction {predicted_direction} "
            f"was correct but magnitude {anomaly_ratio:.1f}x larger than expected "
            f"({actual_move_pct:.1f}% vs. {predicted_residual:.1f}% bound)."
        )
    else:
        reason = (
            f"Prediction {predicted_direction} at {predicted_confidence:.0%} confidence "
            f"was contradicted. Actual move {actual_move_pct:+.1f}% "
            f"({anomaly_ratio:.1f}x predicted residual {predicted_residual:.1f}%)."
        )
    
    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_direction=predicted_direction,
        predicted_confidence=predicted_confidence,
        actual_move_pct=actual_move_pct,
        predicted_residual=predicted_residual,
        anomaly_ratio=anomaly_ratio,
        severity=severity,
        alert_reason=reason,
        generated_at=datetime.utcnow().isoformat(),
    )


def save_anomaly_alerts(db_path: str, alerts: list[AnomalyAlert]) -> int:
    """Save detected anomalies to SQLite database; return count inserted."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_direction TEXT,
            predicted_confidence REAL,
            actual_move_pct REAL,
            predicted_residual REAL,
            anomaly_ratio REAL,
            severity TEXT,
            alert_reason TEXT,
            generated_at TEXT,
            UNIQUE(ticker, prediction_date)
        )
    """)
    
    inserted = 0
    for alert in alerts:
        try:
            cur.execute("""
                INSERT INTO anomaly_alerts (
                    ticker, prediction_date, predicted_direction, predicted_confidence,
                    actual_move_pct, predicted_residual, anomaly_ratio, severity,
                    alert_reason, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.ticker, alert.prediction_date, alert.predicted_direction,
                alert.predicted_confidence, alert.actual_move_pct, alert.predicted_residual,
                alert.anomaly_ratio, alert.severity, alert.alert_reason, alert.generated_at,
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()
    return inserted


def load_anomaly_alerts(db_path: str, ticker: Optional[str] = None, severity: Optional[str] = None) -> list[AnomalyAlert]:
    """Load anomaly alerts from database; optionally filter by ticker and/or severity."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    query = "SELECT * FROM anomaly_alerts WHERE 1=1"
    params = []
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    
    query += " ORDER BY generated_at DESC"
    
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    
    alerts = []
    if rows:
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        for row in rows:
            alert_dict = dict(zip(col_names, row))
            alert_dict.pop('id', None)
            alerts.append(AnomalyAlert(**alert_dict))
    
    return alerts


def anomaly_summary(alerts: list[AnomalyAlert]) -> dict:
    """Generate summary statistics from a list of anomaly alerts."""
    if not alerts:
        return {
            "total_alerts": 0,
            "by_severity": {},
            "by_ticker": {},
            "avg_anomaly_ratio": 0.0,
        }
    
    by_severity = {}
    by_ticker = {}
    
    for alert in alerts:
        by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
        by_ticker[alert.ticker] = by_ticker.get(alert.ticker, 0) + 1
    
    avg_ratio = sum(a.anomaly_ratio for a in alerts) / len(alerts)
    
    return {
        "total_alerts": len(alerts),
        "by_severity": by_severity,
        "by_ticker": by_ticker,
        "avg_anomaly_ratio": round(avg_
