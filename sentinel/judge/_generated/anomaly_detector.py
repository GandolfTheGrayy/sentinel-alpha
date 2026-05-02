"""
Anomaly Detection Engine for Sentinel Sentiment Engine.

This module detects when actual market moves deviate significantly from predicted
moves, flagging statistical outliers that warrant post-mortem analysis. It compares
predicted price deltas against realized deltas and generates AnomalyAlert objects
when actual moves exceed 2x the predicted residual threshold.

Used by `sentinel/judge/postmortem.py` to identify prediction failures and retrain signals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3


@dataclass
class AnomalyAlert:
    """
    Immutable alert for a single anomalous market move.
    
    Attributes:
        ticker: Stock symbol (e.g., 'AAPL').
        prediction_date: Date on which prediction was made.
        predicted_delta_pct: Model's predicted price change (%).
        actual_delta_pct: Realized price change (%).
        residual_pct: Absolute error = |actual - predicted|.
        anomaly_severity: Ratio of actual_delta to predicted_delta (>2.0 = anomalous).
        flagged_at: Timestamp when anomaly was detected.
        explanation: Optional human-readable reason for the flag.
    """
    ticker: str
    prediction_date: str
    predicted_delta_pct: float
    actual_delta_pct: float
    residual_pct: float
    anomaly_severity: float
    flagged_at: datetime = field(default_factory=datetime.utcnow)
    explanation: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize AnomalyAlert to dict for JSON/DB storage."""
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date,
            "predicted_delta_pct": self.predicted_delta_pct,
            "actual_delta_pct": self.actual_delta_pct,
            "residual_pct": self.residual_pct,
            "anomaly_severity": self.anomaly_severity,
            "flagged_at": self.flagged_at.isoformat(),
            "explanation": self.explanation,
        }


class AnomalyDetector:
    """
    Detects statistical outliers in prediction vs. realized market moves.
    
    Thresholds:
      - anomaly_threshold: severity ratio (actual / predicted); default 2.0x.
      - min_prediction_magnitude: ignore predictions with |delta| < this; default 0.5%.
    """

    def __init__(
        self,
        anomaly_threshold: float = 2.0,
        min_prediction_magnitude: float = 0.5,
    ):
        """
        Initialize anomaly detector with thresholds.
        
        Args:
            anomaly_threshold: Severity ratio trigger (default 2.0x).
            min_prediction_magnitude: Ignore predictions with |delta| < this %.
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_prediction_magnitude = min_prediction_magnitude

    def detect(
        self,
        ticker: str,
        prediction_date: str,
        predicted_delta_pct: float,
        actual_delta_pct: float,
    ) -> Optional[AnomalyAlert]:
        """
        Compare prediction to actual move and return AnomalyAlert if threshold exceeded.
        
        Args:
            ticker: Stock symbol.
            prediction_date: ISO date string of prediction.
            predicted_delta_pct: Model's predicted % change.
            actual_delta_pct: Realized % change.
        
        Returns:
            AnomalyAlert if anomaly detected; None otherwise.
        """
        residual_pct = abs(actual_delta_pct - predicted_delta_pct)

        # Skip tiny predictions (noise floor).
        if abs(predicted_delta_pct) < self.min_prediction_magnitude:
            return None

        # Calculate severity as ratio of actual magnitude to predicted magnitude.
        if predicted_delta_pct == 0:
            severity = float("inf") if actual_delta_pct != 0 else 1.0
        else:
            severity = abs(actual_delta_pct) / abs(predicted_delta_pct)

        # Flag if severity exceeds threshold.
        if severity >= self.anomaly_threshold:
            explanation = self._explain_anomaly(
                ticker, predicted_delta_pct, actual_delta_pct, severity
            )
            return AnomalyAlert(
                ticker=ticker,
                prediction_date=prediction_date,
                predicted_delta_pct=predicted_delta_pct,
                actual_delta_pct=actual_delta_pct,
                residual_pct=residual_pct,
                anomaly_severity=severity,
                explanation=explanation,
            )
        return None

    def _explain_anomaly(
        self,
        ticker: str,
        predicted: float,
        actual: float,
        severity: float,
    ) -> str:
        """Generate human-readable explanation for anomaly."""
        direction_match = (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0)
        direction_word = "direction matched" if direction_match else "direction REVERSED"
        return (
            f"{ticker}: predicted {predicted:.2f}%, actual {actual:.2f}% "
            f"({direction_word}, severity={severity:.2f}x)"
        )

    def detect_batch(
        self, predictions: list[dict]
    ) -> list[AnomalyAlert]:
        """
        Scan multiple predictions and return all anomalies detected.
        
        Args:
            predictions: List of dicts with keys
              {ticker, prediction_date, predicted_delta_pct, actual_delta_pct}.
        
        Returns:
            List of AnomalyAlert objects (only anomalies).
        """
        alerts = []
        for pred in predictions:
            alert = self.detect(
                ticker=pred["ticker"],
                prediction_date=pred["prediction_date"],
                predicted_delta_pct=pred["predicted_delta_pct"],
                actual_delta_pct=pred["actual_delta_pct"],
            )
            if alert:
                alerts.append(alert)
        return alerts


class AnomalyStore:
    """
    Persistent SQLite storage for anomaly alerts.
    
    Schema: anomalies table with columns for ticker, dates, deltas, severity, explanation.
    """

    def __init__(self, db_path: str = "sentinel_anomalies.db"):
        """
        Initialize anomaly store, creating table if needed.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Create anomalies table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    predicted_delta_pct REAL NOT NULL,
                    actual_delta_pct REAL NOT NULL,
                    residual_pct REAL NOT NULL,
                    anomaly_severity REAL NOT NULL,
                    flagged_at TEXT NOT NULL,
                    explanation TEXT,
                    UNIQUE(ticker, prediction_date)
                )
                """
            )
            conn.commit()

    def save(self, alert: AnomalyAlert) -> bool:
        """
        Persist an AnomalyAlert to the database.
        
        Args:
            alert: AnomalyAlert instance to save.
        
        Returns:
            True if inserted; False if duplicate key.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO anomalies (
                        ticker, prediction_date, predicted_delta_pct,
                        actual_delta_pct, residual_pct, anomaly_severity,
