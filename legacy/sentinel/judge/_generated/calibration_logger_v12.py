"""
Calibration Logger — Heuristic Update & Rolling Accuracy Metrics

This module provides persistent storage and retrieval of CalibrationResult entries
(predicted vs. actual market moves) and computes rolling 7-day and 30-day accuracy
metrics to drive Judge heuristic refinement. Appends to a JSONL file for immutability
and enables post-mortem analysis and confidence scoring recalibration.

Integrated into: sentinel/judge/postmortem.py (accuracy trending),
sentinel/judge/predictor.py (confidence weighting feedback loop).
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TypedDict
import numpy as np


class CalibrationResult(TypedDict):
    """Schema for a single calibration entry."""
    timestamp: str
    ticker: str
    predicted_direction: str
    actual_direction: str
    predicted_confidence: float
    prediction_date: str
    resolution_date: str
    correct: bool
    feature_tags: list[str]


class RollingAccuracyMetrics(TypedDict):
    """Rolling accuracy aggregates over a time window."""
    window_days: int
    start_date: str
    end_date: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    by_confidence_bin: dict[str, dict]


def append_calibration_result(
    result: CalibrationResult,
    db_path: str = "sentinel_calibration.db"
) -> None:
    """Append a CalibrationResult entry to the persistent SQLite database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS calibration_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            actual_direction TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            prediction_date TEXT NOT NULL,
            resolution_date TEXT NOT NULL,
            correct BOOLEAN NOT NULL,
            feature_tags TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        INSERT INTO calibration_log (
            timestamp, ticker, predicted_direction, actual_direction,
            predicted_confidence, prediction_date, resolution_date,
            correct, feature_tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result["timestamp"],
            result["ticker"],
            result["predicted_direction"],
            result["actual_direction"],
            result["predicted_confidence"],
            result["prediction_date"],
            result["resolution_date"],
            result["correct"],
            json.dumps(result["feature_tags"]),
        ),
    )
    conn.commit()
    conn.close()


def compute_rolling_accuracy(
    window_days: int = 7,
    db_path: str = "sentinel_calibration.db"
) -> Optional[RollingAccuracyMetrics]:
    """Compute accuracy metrics over the past N days, binned by confidence level."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    cutoff_str = cutoff.isoformat()
    
    cur.execute(
        """
        SELECT predicted_confidence, correct FROM calibration_log
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        """,
        (cutoff_str,),
    )
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return None
    
    total = len(rows)
    correct = sum(1 for _, is_correct in rows if is_correct)
    
    # Bin by confidence: [0.5-0.6), [0.6-0.7), [0.7-0.8), [0.8-0.9), [0.9-1.0]
    confidence_bins = {
        "50-60": {"count": 0, "correct": 0},
        "60-70": {"count": 0, "correct": 0},
        "70-80": {"count": 0, "correct": 0},
        "80-90": {"count": 0, "correct": 0},
        "90-100": {"count": 0, "correct": 0},
    }
    
    for conf, is_correct in rows:
        if 0.50 <= conf < 0.60:
            key = "50-60"
        elif 0.60 <= conf < 0.70:
            key = "60-70"
        elif 0.70 <= conf < 0.80:
            key = "70-80"
        elif 0.80 <= conf < 0.90:
            key = "80-90"
        else:
            key = "90-100"
        
        confidence_bins[key]["count"] += 1
        if is_correct:
            confidence_bins[key]["correct"] += 1
    
    # Compute accuracy per bin
    for bin_data in confidence_bins.values():
        if bin_data["count"] > 0:
            bin_data["accuracy"] = bin_data["correct"] / bin_data["count"]
        else:
            bin_data["accuracy"] = None
    
    return RollingAccuracyMetrics(
        window_days=window_days,
        start_date=cutoff_str,
        end_date=datetime.utcnow().isoformat(),
        total_predictions=total,
        correct_predictions=correct,
        accuracy=correct / total if total > 0 else 0.0,
        by_confidence_bin=confidence_bins,
    )


def compute_accuracy_by_ticker(
    window_days: int = 30,
    db_path: str = "sentinel_calibration.db"
) -> dict[str, dict]:
    """Compute per-ticker accuracy over the past N days."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(days=window_days)
    cutoff_str = cutoff.isoformat()
    
    cur.execute(
        """
        SELECT ticker, COUNT(*) as total, SUM(correct) as correct
        FROM calibration_log
        WHERE timestamp >= ?
        GROUP BY ticker
        ORDER BY ticker ASC
        """,
        (cutoff_str,),
    )
    rows = cur.fetchall()
    conn.close()
    
    result = {}
    for ticker, total, correct_count in rows:
        result[ticker] = {
            "total_predictions": total,
            "correct_predictions": int(correct_count or 0),
            "accuracy": (correct_count / total) if total > 0 else 0.0,
        }
    
    return result


def get_recent_calibration_entries(
    limit: int = 100,
    db_path: str = "sentinel_calibration.db"
) -> list[CalibrationResult]:
    """Retrieve the most recent N calibration entries from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute(
        """
        SELECT timestamp, ticker, predicted_direction, actual_direction,
               predicted_confidence, prediction_date, resolution_date,
               correct, feature_tags
        FROM calibration_log
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append(
            CalibrationResult(
                timestamp=row[0],
                ticker=row[1],
                predicted_direction=row[2],
                actual_direction=row[3],
                predicted_confidence=row[4],
                prediction_date=row[5],
                resolution_date=row[6],
                correct=bool(row[7]),
                feature_tags=json.loads(row[8]),
            )
        )
    
    return results


def accuracy_trend
