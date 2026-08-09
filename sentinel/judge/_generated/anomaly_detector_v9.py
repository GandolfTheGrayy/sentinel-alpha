"""
Anomaly Detector for Sentinel Sentiment Engine.

Detects when actual market moves exceed 2x the predicted residual and generates
AnomalyAlert dataclasses. Flags unexpected market behavior for post-mortem analysis
and heuristic refinement. Integrated into the Judge pillar's daily calibration loop.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """Represents a flagged anomaly in market behavior vs. prediction."""

    ticker: str
    prediction_date: datetime
    predicted_movement_pct: float
    actual_movement_pct: float
    residual_pct: float
    anomaly_ratio: float
    severity: str  # "HIGH", "MEDIUM", "LOW"
    alert_message: str
    confidence_score: Optional[float] = None
    predicted_direction: Optional[str] = None  # "UP", "DOWN", "NEUTRAL"
    actual_direction: Optional[str] = None  # "UP", "DOWN", "NEUTRAL"
    metadata: dict = field(default_factory=dict)
    flagged_at: datetime = field(default_factory=datetime.utcnow)


def compute_residual(
    predicted_pct: float, actual_pct: float
) -> float:
    """Compute absolute residual between predicted and actual movement."""
    return abs(actual_pct - predicted_pct)


def compute_anomaly_ratio(residual_pct: float) -> float:
    """Compute how many times the residual exceeds a baseline (1% threshold)."""
    baseline_threshold = 1.0
    if residual_pct < baseline_threshold:
        return 0.0
    return residual_pct / baseline_threshold


def classify_severity(anomaly_ratio: float) -> str:
    """Classify anomaly severity based on ratio threshold."""
    if anomaly_ratio >= 3.0:
        return "HIGH"
    elif anomaly_ratio >= 2.0:
        return "MEDIUM"
    else:
        return "LOW"


def infer_direction(movement_pct: float) -> str:
    """Infer direction from signed percentage movement."""
    if movement_pct > 0.5:
        return "UP"
    elif movement_pct < -0.5:
        return "DOWN"
    else:
        return "NEUTRAL"


def detect_anomalies(
    ticker: str,
    predicted_movement_pct: float,
    actual_movement_pct: float,
    prediction_date: datetime,
    confidence_score: Optional[float] = None,
    metadata: Optional[dict] = None,
    threshold_ratio: float = 2.0,
) -> Optional[AnomalyAlert]:
    """
    Detect if actual move exceeds 2x the predicted residual; return AnomalyAlert or None.

    Args:
        ticker: Stock ticker symbol.
        predicted_movement_pct: Model's predicted percentage move.
        actual_movement_pct: Realized percentage move.
        prediction_date: When the prediction was issued.
        confidence_score: Optional model confidence (0-1).
        metadata: Optional dict of extra context.
        threshold_ratio: Multiplier threshold (default 2.0x).

    Returns:
        AnomalyAlert if anomaly_ratio >= threshold_ratio, else None.
    """
    if metadata is None:
        metadata = {}

    residual = compute_residual(predicted_movement_pct, actual_movement_pct)
    anomaly_ratio = compute_anomaly_ratio(residual)

    if anomaly_ratio < threshold_ratio:
        return None

    severity = classify_severity(anomaly_ratio)
    predicted_dir = infer_direction(predicted_movement_pct)
    actual_dir = infer_direction(actual_movement_pct)

    direction_mismatch = (
        f" (predicted {predicted_dir}, actual {actual_dir})"
        if predicted_dir != actual_dir
        else ""
    )

    alert_msg = (
        f"{ticker}: residual {residual:.2f}% ({anomaly_ratio:.1f}x baseline) "
        f"predicted {predicted_movement_pct:+.2f}% vs actual {actual_movement_pct:+.2f}%"
        f"{direction_mismatch}"
    )

    return AnomalyAlert(
        ticker=ticker,
        prediction_date=prediction_date,
        predicted_movement_pct=predicted_movement_pct,
        actual_movement_pct=actual_movement_pct,
        residual_pct=residual,
        anomaly_ratio=anomaly_ratio,
        severity=severity,
        alert_message=alert_msg,
        confidence_score=confidence_score,
        predicted_direction=predicted_dir,
        actual_direction=actual_dir,
        metadata=metadata,
    )


def store_anomaly_alert(
    db_path: str, alert: AnomalyAlert
) -> None:
    """Store AnomalyAlert to SQLite database for auditing and analysis."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_movement_pct REAL,
            actual_movement_pct REAL,
            residual_pct REAL,
            anomaly_ratio REAL,
            severity TEXT,
            alert_message TEXT,
            confidence_score REAL,
            predicted_direction TEXT,
            actual_direction TEXT,
            metadata TEXT,
            flagged_at TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        INSERT INTO anomaly_alerts
        (ticker, prediction_date, predicted_movement_pct, actual_movement_pct,
         residual_pct, anomaly_ratio, severity, alert_message,
         confidence_score, predicted_direction, actual_direction,
         metadata, flagged_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert.ticker,
            alert.prediction_date.isoformat(),
            alert.predicted_movement_pct,
            alert.actual_movement_pct,
            alert.residual_pct,
            alert.anomaly_ratio,
            alert.severity,
            alert.alert_message,
            alert.confidence_score,
            alert.predicted_direction,
            alert.actual_direction,
            json.dumps(alert.metadata),
            alert.flagged_at.isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def load_recent_anomalies(
    db_path: str, limit: int = 50
) -> list[AnomalyAlert]:
    """Load recent anomaly alerts from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM anomaly_alerts
        ORDER BY flagged_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    alerts = []
    for row in rows:
        alert = AnomalyAlert(
            ticker=row["ticker"],
            prediction_date=datetime.fromisoformat(row["prediction_date"]),
            predicted_movement_pct=row["predicted_movement_pct"],
            actual_movement_pct=row["actual_movement_pct"],
            residual_pct=row["residual_pct"],
            anomaly_ratio=row["anomaly_ratio"],
            severity=row["severity"],
            alert_message=row["alert_message"],
            confidence_score=row["confidence_score"],
            predicted_direction=row["predicted_direction"],
            actual_direction=row["actual_direction"],
            metadata=json.loads(row["metadata"]) if row["metadata
