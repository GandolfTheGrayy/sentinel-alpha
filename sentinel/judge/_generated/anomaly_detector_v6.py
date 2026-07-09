"""
Anomaly detection system for Sentinel Sentiment Engine.

This module compares actual market moves against predicted residuals,
flagging trades where reality diverged >2x from expectation. Generates
AnomalyAlert dataclasses for post-mortem analysis and model refinement.

Fits into judge pillar: runs after resolver confirms actual prices,
feeds flagged anomalies into daily postmortem for heuristic tuning.
"""

import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json


@dataclass
class AnomalyAlert:
    """Represents a detected market anomaly exceeding prediction confidence."""
    
    ticker: str
    date: str
    predicted_move_pct: float
    actual_move_pct: float
    residual_ratio: float
    severity: str  # "minor" (2-3x), "moderate" (3-5x), "severe" (>5x)
    predicted_direction: str  # "up", "down", "neutral"
    actual_direction: str  # "up", "down", "neutral"
    confidence_score: float
    news_events: list[str] = field(default_factory=list)
    flagged_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


class AnomalyDetector:
    """
    Detects market moves that exceed 2x predicted residual.
    Stores alerts in SQLite for daily postmortem aggregation.
    """
    
    def __init__(self, db_path: str = "sentinel/data/sentinel.db"):
        """Initialize detector and create alerts table if absent."""
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Create anomaly_alerts table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                predicted_move_pct REAL NOT NULL,
                actual_move_pct REAL NOT NULL,
                residual_ratio REAL NOT NULL,
                severity TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                actual_direction TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                news_events TEXT,
                flagged_at TEXT NOT NULL,
                notes TEXT,
                UNIQUE(ticker, date)
            )
        """)
        conn.commit()
        conn.close()
    
    def _classify_direction(self, move_pct: float) -> str:
        """Classify price move as up, down, or neutral."""
        if move_pct > 0.5:
            return "up"
        elif move_pct < -0.5:
            return "down"
        else:
            return "neutral"
    
    def _classify_severity(self, ratio: float) -> str:
        """Classify anomaly severity by residual ratio."""
        if ratio >= 5.0:
            return "severe"
        elif ratio >= 3.0:
            return "moderate"
        else:
            return "minor"
    
    def check_prediction(
        self,
        ticker: str,
        date: str,
        predicted_move_pct: float,
        actual_move_pct: float,
        confidence_score: float,
        news_events: Optional[list[str]] = None,
        notes: str = ""
    ) -> Optional[AnomalyAlert]:
        """
        Check if actual move exceeds 2x predicted residual; flag if so.
        
        Returns AnomalyAlert if anomaly detected, else None.
        """
        if confidence_score < 0.0 or confidence_score > 1.0:
            confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Compute residual (unsigned difference)
        residual = abs(actual_move_pct - predicted_move_pct)
        
        # Avoid division by zero; if predicted is ~0, use baseline threshold
        if abs(predicted_move_pct) < 0.01:
            threshold = 1.0  # Flag if actual move exceeds 1%
            ratio = residual / 1.0 if residual > 0 else 0.0
        else:
            threshold = 2.0 * abs(predicted_move_pct)
            ratio = residual / abs(predicted_move_pct) if predicted_move_pct != 0 else 0.0
        
        # Trigger anomaly if residual >= 2x predicted or exceeds absolute threshold
        if residual >= threshold and ratio >= 2.0:
            predicted_dir = self._classify_direction(predicted_move_pct)
            actual_dir = self._classify_direction(actual_move_pct)
            severity = self._classify_severity(ratio)
            
            alert = AnomalyAlert(
                ticker=ticker,
                date=date,
                predicted_move_pct=round(predicted_move_pct, 3),
                actual_move_pct=round(actual_move_pct, 3),
                residual_ratio=round(ratio, 2),
                severity=severity,
                predicted_direction=predicted_dir,
                actual_direction=actual_dir,
                confidence_score=round(confidence_score, 3),
                news_events=news_events or [],
                notes=notes
            )
            
            self._store_alert(alert)
            return alert
        
        return None
    
    def _store_alert(self, alert: AnomalyAlert) -> None:
        """Persist alert to SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO anomaly_alerts
                (ticker, date, predicted_move_pct, actual_move_pct, residual_ratio,
                 severity, predicted_direction, actual_direction, confidence_score,
                 news_events, flagged_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.ticker,
                alert.date,
                alert.predicted_move_pct,
                alert.actual_move_pct,
                alert.residual_ratio,
                alert.severity,
                alert.predicted_direction,
                alert.actual_direction,
                alert.confidence_score,
                json.dumps(alert.news_events),
                alert.flagged_at,
                alert.notes
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            # Duplicate key; update instead
            cursor.execute("""
                UPDATE anomaly_alerts
                SET predicted_move_pct = ?, actual_move_pct = ?, residual_ratio = ?,
                    severity = ?, predicted_direction = ?, actual_direction = ?,
                    confidence_score = ?, news_events = ?, flagged_at = ?, notes = ?
                WHERE ticker = ? AND date = ?
            """, (
                alert.predicted_move_pct,
                alert.actual_move_pct,
                alert.residual_ratio,
                alert.severity,
                alert.predicted_direction,
                alert.actual_direction,
                alert.confidence_score,
                json.dumps(alert.news_events),
                alert.flagged_at,
                alert.notes,
                alert.ticker,
                alert.date
            ))
            conn.commit()
        finally:
            conn.close()
    
    def fetch_alerts_since(self, days_ago: int = 7) -> list[AnomalyAlert]:
        """Retrieve all anomaly alerts from the past N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days_ago)).isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ticker, date
