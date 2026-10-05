"""
Anomaly Detector — Sentinel Judge subsystem.

Detects when actual market moves deviate significantly from predicted residuals.
Generates AnomalyAlert dataclass when actual price change exceeds 2x the predicted
residual standard deviation. Feeds into Judge post-mortem and Discord notifications.

Integrates with:
  - sentinel/judge/predictor.py (predicted residuals, confidence bounds)
  - sentinel/judge/resolver.py (actual market outcomes)
  - sentinel/judge/notify.py (alert dispatch)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3
import json


@dataclass
class AnomalyAlert:
    """Single anomaly event: predicted vs. actual divergence exceeded threshold."""
    
    ticker: str
    prediction_date: str
    alert_date: str
    predicted_move_pct: float
    actual_move_pct: float
    residual_std: float
    threshold_multiple: float
    alert_magnitude: float
    alert_category: str
    confidence_score: Optional[float] = None
    notes: str = field(default_factory=str)
    raw_signals: dict = field(default_factory=dict)


class AnomalyDetector:
    """Detects market move anomalies exceeding 2x residual standard deviation."""
    
    def __init__(self, db_path: str = "sentinel.db", threshold_multiple: float = 2.0):
        """
        Initialize anomaly detector with database path and threshold multiplier.
        
        Args:
            db_path: Path to SQLite database storing predictions and outcomes.
            threshold_multiple: How many standard deviations triggers anomaly (default 2.0).
        """
        self.db_path = db_path
        self.threshold_multiple = threshold_multiple
        self._ensure_table()
    
    def _ensure_table(self) -> None:
        """Create anomaly_alerts table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                alert_date TEXT NOT NULL,
                predicted_move_pct REAL NOT NULL,
                actual_move_pct REAL NOT NULL,
                residual_std REAL NOT NULL,
                threshold_multiple REAL NOT NULL,
                alert_magnitude REAL NOT NULL,
                alert_category TEXT NOT NULL,
                confidence_score REAL,
                notes TEXT,
                raw_signals TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def check_anomaly(
        self,
        ticker: str,
        prediction_date: str,
        predicted_move_pct: float,
        actual_move_pct: float,
        residual_std: float,
        confidence_score: Optional[float] = None,
        raw_signals: Optional[dict] = None,
    ) -> Optional[AnomalyAlert]:
        """
        Check if actual move exceeds 2x predicted residual; return AnomalyAlert if triggered.
        
        Args:
            ticker: Stock symbol (e.g. "AAPL").
            prediction_date: ISO date of prediction (YYYY-MM-DD).
            predicted_move_pct: Model's predicted % change.
            actual_move_pct: Observed % change.
            residual_std: Standard deviation of residuals from model calibration.
            confidence_score: Model confidence [0, 1].
            raw_signals: Dict of contributing signals (sentiment, volume, etc.).
        
        Returns:
            AnomalyAlert if triggered, else None.
        """
        if residual_std <= 0:
            return None
        
        threshold = self.threshold_multiple * residual_std
        alert_magnitude = abs(actual_move_pct - predicted_move_pct)
        
        if alert_magnitude < threshold:
            return None
        
        alert_category = self._categorize_anomaly(
            predicted_move_pct, actual_move_pct, residual_std
        )
        
        alert = AnomalyAlert(
            ticker=ticker,
            prediction_date=prediction_date,
            alert_date=datetime.utcnow().isoformat(),
            predicted_move_pct=predicted_move_pct,
            actual_move_pct=actual_move_pct,
            residual_std=residual_std,
            threshold_multiple=self.threshold_multiple,
            alert_magnitude=alert_magnitude,
            alert_category=alert_category,
            confidence_score=confidence_score,
            notes="",
            raw_signals=raw_signals or {},
        )
        
        self._persist_alert(alert)
        return alert
    
    def _categorize_anomaly(
        self,
        predicted_move_pct: float,
        actual_move_pct: float,
        residual_std: float,
    ) -> str:
        """Categorize anomaly by direction and magnitude relative to residual."""
        move_diff = actual_move_pct - predicted_move_pct
        normalized = move_diff / residual_std if residual_std > 0 else 0
        
        if normalized > 3.0:
            return "EXTREME_UPSIDE"
        elif normalized < -3.0:
            return "EXTREME_DOWNSIDE"
        elif normalized > 2.0:
            return "STRONG_UPSIDE"
        elif normalized < -2.0:
            return "STRONG_DOWNSIDE"
        else:
            return "MODERATE"
    
    def _persist_alert(self, alert: AnomalyAlert) -> None:
        """Store AnomalyAlert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO anomaly_alerts (
                ticker, prediction_date, alert_date, predicted_move_pct,
                actual_move_pct, residual_std, threshold_multiple,
                alert_magnitude, alert_category, confidence_score, notes, raw_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.ticker,
            alert.prediction_date,
            alert.alert_date,
            alert.predicted_move_pct,
            alert.actual_move_pct,
            alert.residual_std,
            alert.threshold_multiple,
            alert.alert_magnitude,
            alert.alert_category,
            alert.confidence_score,
            alert.notes,
            json.dumps(alert.raw_signals),
        ))
        conn.commit()
        conn.close()
    
    def get_alerts_for_ticker(self, ticker: str, limit: int = 50) -> list[AnomalyAlert]:
        """Retrieve recent anomaly alerts for a given ticker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, prediction_date, alert_date, predicted_move_pct,
                   actual_move_pct, residual_std, threshold_multiple,
                   alert_magnitude, alert_category, confidence_score, notes, raw_signals
            FROM anomaly_alerts
            WHERE ticker = ?
            ORDER BY alert_date DESC
            LIMIT ?
        """, (ticker, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in rows:
            alert = AnomalyAlert(
                ticker=row[0],
                prediction_date=row[1],
                alert_date=row[2],
                predicted_move_pct=row[3],
                actual_move_pct=row[4],
                residual_std=row[5],
                threshold_multiple=row[6],
                alert_magnitude=row[7],
                alert_category=row[8],
                confidence_score=row[9],
                notes=row[10],
