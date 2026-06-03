"""
Anomaly detection for Sentinel — flags when actual market moves exceed 2x predicted residuals.

This module compares predicted price movements (from judge/predictor.py) against realized
market moves and raises AnomalyAlert when residuals are anomalously large. Used by the
daily post-mortem (judge/postmortem.py) to surface unexpected market behavior and trigger
heuristic refinement in the Judge pillar.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import sqlite3
import numpy as np


@dataclass
class AnomalyAlert:
    """Represents a flagged anomaly when realized move deviates from prediction."""
    
    ticker: str
    prediction_date: datetime
    predicted_move_pct: float
    actual_move_pct: float
    residual_pct: float
    severity: str  # "mild" | "moderate" | "severe"
    threshold_multiplier: float
    message: str
    flagged_at: datetime


def compute_residual(predicted_move: float, actual_move: float) -> float:
    """
    Compute prediction residual as absolute difference between actual and predicted moves.
    
    Args:
        predicted_move: Predicted price movement in percent.
        actual_move: Realized price movement in percent.
    
    Returns:
        Absolute residual in percent.
    """
    return abs(actual_move - predicted_move)


def classify_severity(residual: float, baseline_volatility: float) -> str:
    """
    Classify anomaly severity based on residual relative to baseline volatility.
    
    Args:
        residual: Absolute residual in percent.
        baseline_volatility: Historical volatility baseline for ticker.
    
    Returns:
        Severity level: "mild", "moderate", or "severe".
    """
    if baseline_volatility == 0:
        baseline_volatility = 1.0
    
    normalized = residual / baseline_volatility
    
    if normalized < 1.5:
        return "mild"
    elif normalized < 3.0:
        return "moderate"
    else:
        return "severe"


def flag_anomaly(
    ticker: str,
    prediction_date: datetime,
    predicted_move_pct: float,
    actual_move_pct: float,
    baseline_volatility: float = 2.5,
    threshold_multiplier: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Flag anomaly if absolute residual exceeds threshold_multiplier × baseline_volatility.
    
    Args:
        ticker: Stock ticker symbol.
        prediction_date: Date prediction was made.
        predicted_move_pct: Predicted move in percent.
        actual_move_pct: Realized move in percent.
        baseline_volatility: Historical volatility baseline (default 2.5%).
        threshold_multiplier: Multiplier on volatility (default 2.0).
    
    Returns:
        AnomalyAlert if residual exceeds threshold, else None.
    """
    residual = compute_residual(predicted_move_pct, actual_move_pct)
    threshold = threshold_multiplier * baseline_volatility
    
    if residual > threshold:
        severity = classify_severity(residual, baseline_volatility)
        message = (
            f"{ticker} anomaly: predicted {predicted_move_pct:.2f}% "
            f"but realized {actual_move_pct:.2f}% "
            f"(residual {residual:.2f}%, threshold {threshold:.2f}%)"
        )
        
        return AnomalyAlert(
            ticker=ticker,
            prediction_date=prediction_date,
            predicted_move_pct=predicted_move_pct,
            actual_move_pct=actual_move_pct,
            residual_pct=residual,
            severity=severity,
            threshold_multiplier=threshold_multiplier,
            message=message,
            flagged_at=datetime.utcnow(),
        )
    
    return None


def batch_flag_anomalies(
    predictions: list[dict],
    actuals: list[dict],
    baseline_volatilities: dict[str, float],
    threshold_multiplier: float = 2.0,
) -> list[AnomalyAlert]:
    """
    Flag anomalies across a batch of predictions vs. actuals.
    
    Args:
        predictions: List of dicts with keys: ticker, prediction_date, predicted_move_pct.
        actuals: List of dicts with keys: ticker, actual_move_pct.
        baseline_volatilities: Dict mapping ticker → historical volatility.
        threshold_multiplier: Multiplier on volatility for threshold.
    
    Returns:
        List of AnomalyAlert objects for flagged anomalies.
    """
    alerts = []
    
    # Build lookup of actuals by ticker
    actual_by_ticker = {a["ticker"]: a for a in actuals}
    
    for pred in predictions:
        ticker = pred["ticker"]
        
        if ticker not in actual_by_ticker:
            continue
        
        actual = actual_by_ticker[ticker]
        baseline_vol = baseline_volatilities.get(ticker, 2.5)
        
        alert = flag_anomaly(
            ticker=ticker,
            prediction_date=pred["prediction_date"],
            predicted_move_pct=pred["predicted_move_pct"],
            actual_move_pct=actual["actual_move_pct"],
            baseline_volatility=baseline_vol,
            threshold_multiplier=threshold_multiplier,
        )
        
        if alert:
            alerts.append(alert)
    
    return alerts


def store_anomaly_alert(
    alert: AnomalyAlert,
    db_path: str = "sentinel.db",
) -> None:
    """
    Persist AnomalyAlert to SQLite for post-mortem audit trail.
    
    Args:
        alert: AnomalyAlert to store.
        db_path: Path to SQLite database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_move_pct REAL NOT NULL,
            actual_move_pct REAL NOT NULL,
            residual_pct REAL NOT NULL,
            severity TEXT NOT NULL,
            threshold_multiplier REAL NOT NULL,
            message TEXT NOT NULL,
            flagged_at TEXT NOT NULL
        )
    """)
    
    cursor.execute("""
        INSERT INTO anomaly_alerts (
            ticker, prediction_date, predicted_move_pct, actual_move_pct,
            residual_pct, severity, threshold_multiplier, message, flagged_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        alert.ticker,
        alert.prediction_date.isoformat(),
        alert.predicted_move_pct,
        alert.actual_move_pct,
        alert.residual_pct,
        alert.severity,
        alert.threshold_multiplier,
        alert.message,
        alert.flagged_at.isoformat(),
    ))
    
    conn.commit()
    conn.close()


def query_anomalies_by_severity(
    severity: str,
    db_path: str = "sentinel.db",
) -> list[AnomalyAlert]:
    """
    Retrieve all anomalies from DB filtered by severity level.
    
    Args:
        severity: One of "mild", "moderate", "severe".
        db_path: Path to SQLite database.
    
    Returns:
        List of AnomalyAlert objects.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, prediction_date, predicted_move_pct, actual_move_pct,
               residual_pct, severity, threshold_multiplier, message, flagged_at
        FROM anomaly
