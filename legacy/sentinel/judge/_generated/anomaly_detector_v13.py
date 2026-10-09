"""
Anomaly detection for the Sentinel Sentiment Engine's judge pillar.

Detects when actual market moves deviate significantly from predicted residuals,
flagging outlier trading days for manual review and model recalibration.
Generates AnomalyAlert dataclass instances for integration into daily post-mortems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """
    Represents a detected anomaly in market behavior vs. prediction.
    
    Fields:
        ticker: Stock symbol being monitored.
        alert_date: Date when anomaly was detected.
        predicted_move_pct: Model's predicted price movement (%).
        actual_move_pct: Observed actual price movement (%).
        residual_pct: Absolute difference between actual and predicted.
        severity_score: 0.0–1.0; 1.0 = extreme outlier.
        threshold_multiple: How many times the residual exceeded the 2x threshold.
        anomaly_type: Category ("price_spike", "reversal", "volume_shock", "gap").
        signal_sources: List of contributing data sources that may have missed.
        confidence_in_anomaly: 0.0–1.0; confidence in the flagging itself.
        notes: Free-form analysis of the anomaly.
        created_at: Timestamp of alert generation.
    """
    ticker: str
    alert_date: str
    predicted_move_pct: float
    actual_move_pct: float
    residual_pct: float
    severity_score: float
    threshold_multiple: float
    anomaly_type: str
    signal_sources: list[str] = field(default_factory=list)
    confidence_in_anomaly: float = 0.5
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def detect_anomalies(
    ticker: str,
    predicted_move_pct: float,
    actual_move_pct: float,
    residual_threshold: float = 2.0,
    severity_cutoff: float = 0.7,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds predicted residual by a multiple threshold.
    
    Args:
        ticker: Stock symbol.
        predicted_move_pct: Model's predicted daily move in percentage.
        actual_move_pct: Observed daily move in percentage.
        residual_threshold: Multiple of typical residual (default 2.0x).
        severity_cutoff: Minimum severity score to trigger alert (0.0–1.0).
    
    Returns:
        AnomalyAlert if anomaly detected, None otherwise.
    """
    residual = abs(actual_move_pct - predicted_move_pct)
    
    # Estimate typical daily residual (~1.5% for stocks with moderate volatility).
    typical_residual = 1.5
    threshold_multiple = residual / typical_residual
    
    if threshold_multiple < residual_threshold:
        return None
    
    # Severity: sigmoid-like scaling from residual multiple.
    severity_score = min(1.0, (threshold_multiple - residual_threshold) / 3.0)
    
    if severity_score < severity_cutoff:
        return None
    
    # Classify anomaly type.
    anomaly_type = _classify_anomaly(predicted_move_pct, actual_move_pct)
    
    alert = AnomalyAlert(
        ticker=ticker,
        alert_date=datetime.utcnow().strftime("%Y-%m-%d"),
        predicted_move_pct=predicted_move_pct,
        actual_move_pct=actual_move_pct,
        residual_pct=residual,
        severity_score=severity_score,
        threshold_multiple=threshold_multiple,
        anomaly_type=anomaly_type,
        confidence_in_anomaly=min(0.99, 0.5 + severity_score * 0.4),
    )
    
    return alert


def _classify_anomaly(predicted_pct: float, actual_pct: float) -> str:
    """
    Classify anomaly into price_spike, reversal, or volume_shock.
    
    Args:
        predicted_pct: Predicted move percentage.
        actual_pct: Actual move percentage.
    
    Returns:
        Anomaly type string.
    """
    abs_predicted = abs(predicted_pct)
    abs_actual = abs(actual_pct)
    
    # Reversal: prediction and actual have opposite signs and large magnitude.
    if (predicted_pct > 0.5 and actual_pct < -0.5) or (predicted_pct < -0.5 and actual_pct > 0.5):
        return "reversal"
    
    # Price spike: actual move >> predicted move.
    if abs_actual > abs_predicted * 1.5 and abs_actual > 3.0:
        return "price_spike"
    
    # Volume shock: used when high residual but moderate price move (inferred from other signals).
    return "volume_shock"


def batch_detect_anomalies(
    predictions: list[dict],
) -> list[AnomalyAlert]:
    """
    Detect anomalies across multiple ticker predictions.
    
    Args:
        predictions: List of dicts with keys: ticker, predicted_move_pct, actual_move_pct.
    
    Returns:
        List of AnomalyAlert instances.
    """
    alerts = []
    for pred in predictions:
        alert = detect_anomalies(
            ticker=pred.get("ticker", "UNKNOWN"),
            predicted_move_pct=pred.get("predicted_move_pct", 0.0),
            actual_move_pct=pred.get("actual_move_pct", 0.0),
        )
        if alert:
            alerts.append(alert)
    return alerts


def save_anomaly_alert(alert: AnomalyAlert, db_path: str = ":memory:") -> None:
    """
    Persist AnomalyAlert to SQLite database for historical tracking.
    
    Args:
        alert: AnomalyAlert instance to save.
        db_path: Path to SQLite database file (default in-memory).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            alert_date TEXT NOT NULL,
            predicted_move_pct REAL,
            actual_move_pct REAL,
            residual_pct REAL,
            severity_score REAL,
            threshold_multiple REAL,
            anomaly_type TEXT,
            signal_sources TEXT,
            confidence_in_anomaly REAL,
            notes TEXT,
            created_at TEXT
        )
        """
    )
    
    cursor.execute(
        """
        INSERT INTO anomaly_alerts (
            ticker, alert_date, predicted_move_pct, actual_move_pct,
            residual_pct, severity_score, threshold_multiple, anomaly_type,
            signal_sources, confidence_in_anomaly, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert.ticker,
            alert.alert_date,
            alert.predicted_move_pct,
            alert.actual_move_pct,
            alert.residual_pct,
            alert.severity_score,
            alert.threshold_multiple,
            alert.anomaly_type,
            json.dumps(alert.signal_sources),
            alert.confidence_in_anomaly,
            alert.notes,
            alert.created_at,
        ),
    )
    
    conn.commit()
    conn.close()


def load_anomaly_alerts(
    ticker: Optional[str] = None,
    db_path: str = ":
