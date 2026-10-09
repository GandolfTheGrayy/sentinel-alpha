"""
Calibration logger for Sentinel Sentiment Engine.

Appends CalibrationResult entries to a JSONL file and computes rolling 7-day
and 30-day accuracy metrics. Called by Judge post-mortem to track prediction
vs. actual market outcomes and heuristic drift over time.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


class CalibrationResult:
    """Encapsulates a single prediction-vs-actual comparison."""

    def __init__(
        self,
        ticker: str,
        predicted_direction: str,
        predicted_confidence: float,
        actual_direction: str,
        timestamp: datetime,
        notes: Optional[str] = None,
    ) -> None:
        """Initialize a calibration result entry."""
        self.ticker = ticker
        self.predicted_direction = predicted_direction
        self.predicted_confidence = predicted_confidence
        self.actual_direction = actual_direction
        self.timestamp = timestamp
        self.notes = notes or ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "ticker": self.ticker,
            "predicted_direction": self.predicted_direction,
            "predicted_confidence": self.predicted_confidence,
            "actual_direction": self.actual_direction,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            "correct": self.predicted_direction == self.actual_direction,
        }


class CalibrationLogger:
    """Logs calibration results and computes rolling accuracy metrics."""

    def __init__(self, db_path: str = "sentinel/data/calibration.db") -> None:
        """Initialize logger with SQLite backend for fast range queries."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create calibration table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                actual_direction TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                notes TEXT,
                correct INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON calibration_results(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ticker ON calibration_results(ticker)"
        )
        conn.commit()
        conn.close()

    def log_result(self, result: CalibrationResult) -> None:
        """Append a calibration result to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO calibration_results
            (ticker, predicted_direction, predicted_confidence, actual_direction, timestamp, notes, correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.ticker,
                result.predicted_direction,
                result.predicted_confidence,
                result.actual_direction,
                result.timestamp.isoformat(),
                result.notes,
                1 if result.predicted_direction == result.actual_direction else 0,
            ),
        )
        conn.commit()
        conn.close()

    def get_rolling_accuracy(
        self, days: int = 7, ticker: Optional[str] = None
    ) -> dict:
        """
        Compute rolling accuracy over last N days.

        Returns dict with keys: total, correct, accuracy_pct.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = (
            "SELECT COUNT(*) as total, SUM(correct) as correct "
            "FROM calibration_results WHERE timestamp >= ?"
        )
        params = [cutoff]

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()

        total, correct = row if row else (0, 0)
        correct = correct or 0
        accuracy_pct = (correct / total * 100) if total > 0 else 0

        return {
            "days": days,
            "total": total,
            "correct": int(correct),
            "accuracy_pct": round(accuracy_pct, 2),
            "ticker": ticker,
        }

    def get_accuracy_by_confidence(
        self, days: int = 7, confidence_bins: Optional[list] = None
    ) -> list:
        """
        Stratify accuracy by predicted confidence level.

        Returns list of dicts: {bin, total, correct, accuracy_pct}.
        """
        if confidence_bins is None:
            confidence_bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cursor.execute(
            "SELECT predicted_confidence, correct FROM calibration_results WHERE timestamp >= ? ORDER BY predicted_confidence",
            (cutoff,),
        )
        rows = cursor.fetchall()
        conn.close()

        results = []
        for i in range(len(confidence_bins) - 1):
            low, high = confidence_bins[i], confidence_bins[i + 1]
            bin_rows = [
                (conf, correct)
                for conf, correct in rows
                if low <= conf < high or (i == len(confidence_bins) - 2 and conf == high)
            ]
            total = len(bin_rows)
            correct = sum(1 for _, c in bin_rows if c)
            accuracy_pct = (correct / total * 100) if total > 0 else 0

            results.append(
                {
                    "bin": f"{low:.1f}–{high:.1f}",
                    "total": total,
                    "correct": correct,
                    "accuracy_pct": round(accuracy_pct, 2),
                }
            )

        return results

    def export_jsonl(self, output_path: str = "sentinel/data/calibration.jsonl") -> None:
        """Export all calibration results to JSONL."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, predicted_direction, predicted_confidence, actual_direction, timestamp, notes, correct FROM calibration_results ORDER BY timestamp"
        )
        rows = cursor.fetchall()
        conn.close()

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for (
                ticker,
                pred_dir,
                pred_conf,
                actual_dir,
                timestamp,
                notes,
                correct,
            ) in rows:
                entry = {
                    "ticker": ticker,
                    "predicted_direction": pred_dir,
                    "predicted_confidence": pred_conf,
                    "actual_direction": actual_dir,
                    "timestamp": timestamp,
                    "notes": notes,
                    "correct": bool(correct),
                }
                f.write(json.dumps(entry) + "\n")

    def get_summary_stats(self, days: int = 7) -> dict:
        """
        Get comprehensive summary statistics over last N days.

        Returns dict with keys: accuracy_7d, accuracy_30d, total_predictions,
        avg_confidence, by_ticker.
        """
