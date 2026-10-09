"""
Calibration Logger for Sentinel Sentiment Engine.

This module provides persistent logging of prediction accuracy metrics and
heuristic calibration results. It appends CalibrationResult entries to a
JSONL file and computes rolling 7-day and 30-day accuracy windows to track
model drift and inform Judge refinement over time.

Used by sentinel/judge/postmortem.py to store daily post-mortems and by
sentinel/judge/resolver.py to inspect historical performance trends.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass
class CalibrationResult:
    """A single prediction accuracy record."""

    timestamp: str
    ticker: str
    predicted_direction: str
    actual_direction: str
    predicted_magnitude: float
    actual_magnitude: float
    confidence_score: float
    correct: bool
    absolute_error: float
    notes: Optional[str] = None


class CalibrationLogger:
    """Persistent JSONL logger for prediction accuracy and heuristic calibration."""

    def __init__(self, db_path: str = "sentinel_calibration.db") -> None:
        """Initialize logger with SQLite backing and optional JSONL export."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create calibration table if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    predicted_direction TEXT NOT NULL,
                    actual_direction TEXT NOT NULL,
                    predicted_magnitude REAL NOT NULL,
                    actual_magnitude REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    correct INTEGER NOT NULL,
                    absolute_error REAL NOT NULL,
                    notes TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp ON calibration_results(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ticker ON calibration_results(ticker)"
            )
            conn.commit()

    def append(self, result: CalibrationResult) -> None:
        """Write a single CalibrationResult to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO calibration_results (
                    timestamp, ticker, predicted_direction, actual_direction,
                    predicted_magnitude, actual_magnitude, confidence_score,
                    correct, absolute_error, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.timestamp,
                    result.ticker,
                    result.predicted_direction,
                    result.actual_direction,
                    result.predicted_magnitude,
                    result.actual_magnitude,
                    result.confidence_score,
                    int(result.correct),
                    result.absolute_error,
                    result.notes,
                ),
            )
            conn.commit()

    def batch_append(self, results: List[CalibrationResult]) -> None:
        """Write multiple CalibrationResult entries efficiently."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO calibration_results (
                    timestamp, ticker, predicted_direction, actual_direction,
                    predicted_magnitude, actual_magnitude, confidence_score,
                    correct, absolute_error, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.timestamp,
                        r.ticker,
                        r.predicted_direction,
                        r.actual_direction,
                        r.predicted_magnitude,
                        r.actual_magnitude,
                        r.confidence_score,
                        int(r.correct),
                        r.absolute_error,
                        r.notes,
                    )
                    for r in results
                ],
            )
            conn.commit()

    def rolling_accuracy(self, days: int = 7, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Compute accuracy metrics over last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT correct, confidence_score, absolute_error FROM calibration_results WHERE timestamp >= ?"
            params = [cutoff]
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            rows = conn.execute(query, params).fetchall()

        if not rows:
            return {
                "period_days": days,
                "ticker": ticker,
                "total_predictions": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "avg_absolute_error": 0.0,
            }

        correct_count = sum(1 for r in rows if r[0])
        avg_confidence = sum(r[1] for r in rows) / len(rows)
        avg_error = sum(r[2] for r in rows) / len(rows)

        return {
            "period_days": days,
            "ticker": ticker,
            "total_predictions": len(rows),
            "accuracy": correct_count / len(rows),
            "avg_confidence": avg_confidence,
            "avg_absolute_error": avg_error,
        }

    def export_jsonl(self, output_path: str) -> None:
        """Export all calibration records to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM calibration_results ORDER BY timestamp DESC"
            ).fetchall()

        with open(output_path, "w") as f:
            for row in rows:
                record = dict(row)
                record["correct"] = bool(record["correct"])
                f.write(json.dumps(record) + "\n")

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Retrieve all calibration records as dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM calibration_results ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_ticker_history(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent prediction history for a specific ticker."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM calibration_results WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def confidence_binned_accuracy(
        self, bin_width: float = 0.1, days: Optional[int] = None
    ) -> Dict[str, float]:
        """Compute accuracy within confidence score bins (e.g. 0.0–0.1, 0.1–0.2, etc.)."""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT correct, confidence_score FROM calibration_results"
            params = []
            if days:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
                query += " WHERE timestamp >= ?"
