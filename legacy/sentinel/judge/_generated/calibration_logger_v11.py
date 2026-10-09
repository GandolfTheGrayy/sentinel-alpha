"""
Calibration Logger for Sentinel Sentiment Engine.

This module maintains a persistent JSONL log of CalibrationResult entries
(predicted vs. actual market moves) and computes rolling 7-day and 30-day
accuracy metrics. Used by the Judge post-mortem pipeline to refine heuristics
and detect systematic prediction drift.

Integrates with sentinel/judge/postmortem.py to persist daily predictions
and enable retrospective performance analysis via scripts/weekly_retro.py.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple


@dataclass
class CalibrationResult:
    """A single prediction outcome: predicted direction vs. actual price move."""

    ticker: str
    date: str
    predicted_direction: str
    predicted_confidence: float
    actual_direction: str
    actual_return_pct: float
    correct: bool
    model_version: str = "sonnet-4-6"


class CalibrationLogger:
    """Appends CalibrationResult entries to JSONL and computes rolling accuracy."""

    def __init__(self, log_path: Optional[Path] = None, db_path: Optional[Path] = None) -> None:
        """Initialize logger with JSONL and SQLite paths."""
        self.log_path = log_path or Path("sentinel/data/calibration.jsonl")
        self.db_path = db_path or Path("sentinel/data/calibration.db")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite table for fast rolling-window queries."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                actual_direction TEXT NOT NULL,
                actual_return_pct REAL NOT NULL,
                correct INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
            """
        )
        conn.commit()
        conn.close()

    def append(self, result: CalibrationResult) -> None:
        """Write a CalibrationResult to JSONL and SQLite."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO calibration
                (ticker, date, predicted_direction, predicted_confidence,
                 actual_direction, actual_return_pct, correct, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.ticker,
                    result.date,
                    result.predicted_direction,
                    result.predicted_confidence,
                    result.actual_direction,
                    result.actual_return_pct,
                    int(result.correct),
                    result.model_version,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        finally:
            conn.close()

    def rolling_accuracy(
        self, days: int = 7, ticker: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute rolling accuracy over last N days, optionally filtered by ticker."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        query = """
            SELECT COUNT(*) as total, SUM(correct) as correct_count
            FROM calibration
            WHERE date >= ?
        """
        params = [cutoff]

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        cur.execute(query, params)
        total, correct_count = cur.fetchone()
        conn.close()

        if total == 0:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        return {
            "accuracy": correct_count / total if total > 0 else 0.0,
            "total": total,
            "correct": correct_count or 0,
        }

    def per_ticker_accuracy(self, days: int = 30) -> Dict[str, Dict[str, float]]:
        """Compute per-ticker rolling accuracy over last N days."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cur.execute(
            """
            SELECT ticker, COUNT(*) as total, SUM(correct) as correct_count
            FROM calibration
            WHERE date >= ?
            GROUP BY ticker
            ORDER BY ticker
            """,
            [cutoff],
        )

        result = {}
        for ticker, total, correct_count in cur.fetchall():
            result[ticker] = {
                "accuracy": correct_count / total if total > 0 else 0.0,
                "total": total,
                "correct": correct_count or 0,
            }

        conn.close()
        return result

    def confidence_calibration(
        self, bins: int = 5, days: int = 30
    ) -> List[Dict[str, float]]:
        """Compute accuracy stratified by predicted confidence bins (calibration curve)."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cur.execute(
            """
            SELECT predicted_confidence, correct
            FROM calibration
            WHERE date >= ?
            ORDER BY predicted_confidence
            """,
            [cutoff],
        )

        rows = cur.fetchall()
        conn.close()

        if not rows:
            return []

        bin_size = len(rows) // bins
        result = []

        for i in range(bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < bins - 1 else len(rows)
            bin_data = rows[start_idx:end_idx]

            if not bin_data:
                continue

            avg_confidence = sum(row[0] for row in bin_data) / len(bin_data)
            correct_count = sum(row[1] for row in bin_data)
            total = len(bin_data)

            result.append(
                {
                    "bin": i,
                    "avg_confidence": avg_confidence,
                    "accuracy": correct_count / total if total > 0 else 0.0,
                    "count": total,
                }
            )

        return result

    def recent_predictions(self, limit: int = 20) -> List[Dict]:
        """Fetch most recent CalibrationResult entries for review."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            SELECT ticker, date, predicted_direction, predicted_confidence,
                   actual_direction, actual_return_pct, correct, model_version
            FROM calibration
            ORDER BY date DESC
            LIMIT ?
            """,
            [limit],
        )

        result = []
        for row in cur.fetchall():
            result.append(
                {
                    "ticker": row[0],
                    "date": row[1],
                    "predicted_direction": row[2],
                    "predicted_confidence": row[3],
                    "actual_direction": row[4],
                    "actual_return_pct": row[5],
