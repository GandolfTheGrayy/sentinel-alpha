"""
Anomaly Detection Engine for Sentinel Judge Pillar.

Detects when actual market moves exceed 2x the predicted residual and flags
them as AnomalyAlert events. Ingests predicted vs. actual price deltas,
computes z-scores and residual multiples, and persists alerts to SQLite
for post-mortem analysis and model recalibration.

Integrates with Judge's daily reconciliation flow: after resolver.py
reconciles predictions, this module runs to identify surprising market
behavior that may indicate model blind spots or external shocks.
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class AnomalyAlert:
    """
    Represents a single market anomaly event flagged by the detector.
    """
    alert_id: str
    ticker: str
    prediction_date: str
    actual_date: str
    predicted_move_pct: float
    actual_move_pct: float
    residual_pct: float
    residual_multiple: float
    z_score: float
    severity: str
    reason: str
    created_at: str

    def to_dict(self) -> dict:
        """Convert AnomalyAlert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize AnomalyAlert to JSON string."""
        return json.dumps(self.to_dict())


class AnomalyDetector:
    """
    Detects and flags market move anomalies in post-mortem analysis.
    """

    RESIDUAL_THRESHOLD_MULTIPLIER = 2.0
    Z_SCORE_THRESHOLD = 2.0
    DB_PATH = "sentinel_anomalies.db"

    def __init__(self, db_path: str = DB_PATH):
        """Initialize detector and ensure database schema exists."""
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create anomaly alerts table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                alert_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                actual_date TEXT NOT NULL,
                predicted_move_pct REAL NOT NULL,
                actual_move_pct REAL NOT NULL,
                residual_pct REAL NOT NULL,
                residual_multiple REAL NOT NULL,
                z_score REAL NOT NULL,
                severity TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(ticker, prediction_date, actual_date)
            )
        """)
        conn.commit()
        conn.close()

    def detect(
        self,
        ticker: str,
        predicted_move_pct: float,
        actual_move_pct: float,
        historical_residuals: Optional[List[float]] = None,
    ) -> Optional[AnomalyAlert]:
        """
        Detect if actual move is anomalous relative to prediction.

        Returns AnomalyAlert if residual exceeds 2x threshold, else None.
        """
        residual_pct = actual_move_pct - predicted_move_pct
        residual_multiple = (
            abs(residual_pct) / abs(predicted_move_pct)
            if predicted_move_pct != 0
            else abs(residual_pct)
        )

        z_score = 0.0
        if historical_residuals and len(historical_residuals) > 1:
            mean_residual = np.mean(historical_residuals)
            std_residual = np.std(historical_residuals)
            if std_residual > 0:
                z_score = (residual_pct - mean_residual) / std_residual

        is_anomalous = (
            residual_multiple >= self.RESIDUAL_THRESHOLD_MULTIPLIER
            or abs(z_score) >= self.Z_SCORE_THRESHOLD
        )

        if not is_anomalous:
            return None

        severity = self._compute_severity(residual_multiple, z_score)
        reason = self._compute_reason(
            predicted_move_pct, actual_move_pct, residual_multiple, z_score
        )

        alert = AnomalyAlert(
            alert_id=self._generate_alert_id(ticker),
            ticker=ticker,
            prediction_date=datetime.utcnow().isoformat(),
            actual_date=datetime.utcnow().isoformat(),
            predicted_move_pct=predicted_move_pct,
            actual_move_pct=actual_move_pct,
            residual_pct=residual_pct,
            residual_multiple=residual_multiple,
            z_score=z_score,
            severity=severity,
            reason=reason,
            created_at=datetime.utcnow().isoformat(),
        )

        return alert

    def persist_alert(self, alert: AnomalyAlert) -> bool:
        """Store AnomalyAlert in database; return True on success."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO anomaly_alerts
                (alert_id, ticker, prediction_date, actual_date,
                 predicted_move_pct, actual_move_pct, residual_pct,
                 residual_multiple, z_score, severity, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.ticker,
                    alert.prediction_date,
                    alert.actual_date,
                    alert.predicted_move_pct,
                    alert.actual_move_pct,
                    alert.residual_pct,
                    alert.residual_multiple,
                    alert.z_score,
                    alert.severity,
                    alert.reason,
                    alert.created_at,
                ),
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_alerts_by_ticker(self, ticker: str, limit: int = 100) -> List[AnomalyAlert]:
        """Retrieve recent anomaly alerts for a given ticker."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM anomaly_alerts
            WHERE ticker = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (ticker, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        alerts = [
            AnomalyAlert(
                alert_id=row["alert_id"],
                ticker=row["ticker"],
                prediction_date=row["prediction_date"],
                actual_date=row["actual_date"],
                predicted_move_pct=row["predicted_move_pct"],
                actual_move_pct=row["actual_move_pct"],
                residual_pct=row["residual_pct"],
                residual_multiple=row["residual_multiple"],
                z_score=row["z_score"],
                severity=row["severity"],
                reason=row["reason"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return alerts

    def get_all_alerts(self, limit: int = 500) -> List[AnomalyAlert]:
        """Retrieve all recent anomaly alerts across all tickers."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM anomaly_alerts
            ORDER BY created_at DESC
            LIMIT ?
