"""
Calibration Logger for Sentinel Sentiment Engine.

This module manages persistent logging of prediction calibration results
(predicted vs. actual market moves) and computes rolling accuracy metrics
(7-day and 30-day windows) to drive heuristic refinement in the Judge pillar.

Appends CalibrationResult entries to a JSONL file and exposes functions
to query rolling performance by ticker, signal type, and confidence band.
Used by sentinel/judge/judge.py post-mortem flow and by scripts/weekly_retro.py
for retrospective analysis.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics


@dataclass
class CalibrationResult:
    """A single prediction vs. outcome record."""
    timestamp: str
    ticker: str
    predicted_direction: str
    predicted_confidence: float
    actual_direction: str
    signal_type: str
    reasoning: str


DB_PATH = Path("sentinel/data/calibration.db")


def init_calibration_db() -> None:
    """Initialize SQLite table for calibration records if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            reasoning TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def log_calibration_result(result: CalibrationResult) -> None:
    """Append a CalibrationResult to the SQLite database."""
    init_calibration_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO calibration (
            timestamp, ticker, predicted_direction, predicted_confidence,
            actual_direction, signal_type, reasoning
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.timestamp,
            result.ticker,
            result.predicted_direction,
            result.predicted_confidence,
            result.actual_direction,
            result.signal_type,
            result.reasoning,
        ),
    )
    conn.commit()
    conn.close()


def get_rolling_accuracy(
    days: int = 7, ticker: Optional[str] = None
) -> dict[str, Any]:
    """Compute rolling accuracy metrics over the last N days."""
    init_calibration_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    if ticker:
        cursor.execute(
            """
            SELECT predicted_direction, actual_direction, predicted_confidence
            FROM calibration
            WHERE timestamp >= ? AND ticker = ?
            ORDER BY timestamp DESC
            """,
            (cutoff, ticker),
        )
    else:
        cursor.execute(
            """
            SELECT predicted_direction, actual_direction, predicted_confidence
            FROM calibration
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        )

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            "days": days,
            "ticker": ticker,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": None,
            "mean_confidence": None,
        }

    correct = sum(1 for pred, actual, _ in rows if pred == actual)
    confidences = [conf for _, _, conf in rows]

    return {
        "days": days,
        "ticker": ticker,
        "total_predictions": len(rows),
        "correct_predictions": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
    }


def get_calibration_by_confidence_band(
    days: int = 7, ticker: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    """Group rolling accuracy by predicted confidence bands (0-25%, 25-50%, 50-75%, 75-100%)."""
    init_calibration_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

    if ticker:
        cursor.execute(
            """
            SELECT predicted_direction, actual_direction, predicted_confidence
            FROM calibration
            WHERE timestamp >= ? AND ticker = ?
            ORDER BY timestamp DESC
            """,
            (cutoff, ticker),
        )
    else:
        cursor.execute(
            """
            SELECT predicted_direction, actual_direction, predicted_confidence
            FROM calibration
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (cutoff,),
        )

    rows = cursor.fetchall()
    conn.close()

    bands = {
        "0-25": {"total": 0, "correct": 0, "accuracy": None},
        "25-50": {"total": 0, "correct": 0, "accuracy": None},
        "50-75": {"total": 0, "correct": 0, "accuracy": None},
        "75-100": {"total": 0, "correct": 0, "accuracy": None},
    }

    for pred, actual, conf in rows:
        if conf < 0.25:
            band = "0-25"
        elif conf < 0.50:
            band = "25-50"
        elif conf < 0.75:
            band = "50-75"
        else:
            band = "75-100"

        bands[band]["total"] += 1
        if pred == actual:
            bands[band]["correct"] += 1

    for band in bands:
        if bands[band]["total"] > 0:
            bands[band]["accuracy"] = (
                bands[band]["correct"] / bands[band]["total"]
            )

    return bands


def get_all_calibration_records(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch the most recent N calibration records from the database."""
    init_calibration_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, ticker, predicted_direction, predicted_confidence,
               actual_direction, signal_type, reasoning
        FROM calibration
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "timestamp": row[0],
            "ticker": row[1],
            "predicted_direction": row[2],
            "predicted_confidence": row[3],
            "actual_direction": row[4],
            "signal_type": row[5],
            "reasoning": row[6],
        }
        for row in rows
    ]


def export_calibration_jsonl(output_path: str) -> None:
    """Export all calibration records to a JSONL file for external analysis."""
    init_calibration_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT timestamp, ticker, predicted_direction, predicted_confidence,
               actual_direction, signal_type, reasoning
        FROM calibration
        ORDER BY timestamp DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    with open(output_path, "w") as f:
        for row in rows:
            record = {
                "timestamp": row[0],
                "ticker": row[1],
                "predicted_direction": row[2],
                "
