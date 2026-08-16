"""
Calibration Logger for Sentinel Sentiment Engine.

This module provides persistent heuristic logging and rolling accuracy metrics.
It appends CalibrationResult entries to a JSONL file (one JSON object per line)
and computes rolling 7-day and 30-day accuracy windows. Used by Judge's
post-mortem pipeline to track prediction quality and detect drift in model
performance over time.

Integration point: Called by sentinel/judge/postmortem.py after each daily
prediction cycle to log actual vs. predicted outcomes and update rolling stats.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Optional


class CalibrationResult(NamedTuple):
    """A single prediction outcome logged for heuristic calibration."""
    timestamp: str  # ISO 8601
    ticker: str
    predicted_direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    predicted_confidence: float  # [0, 1]
    actual_direction: str  # 'UP', 'DOWN', 'NEUTRAL'
    actual_price_change_pct: float
    correct: bool
    reasoning_tags: list[str]  # e.g., ["sentiment_strong", "low_volume"]


class RollingAccuracy(NamedTuple):
    """Rolling accuracy metrics over a time window."""
    window_days: int
    start_date: str
    end_date: str
    total_predictions: int
    correct_predictions: int
    accuracy: float  # [0, 1]
    by_direction: dict[str, dict[str, int]]  # {"UP": {"total": 10, "correct": 8}, ...}


def _ensure_db(db_path: Path) -> None:
    """Create calibration SQLite schema if it does not exist."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            predicted_direction TEXT NOT NULL,
            predicted_confidence REAL NOT NULL,
            actual_direction TEXT NOT NULL,
            actual_price_change_pct REAL NOT NULL,
            correct INTEGER NOT NULL,
            reasoning_tags TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_calibration_result(
    result: CalibrationResult,
    log_file: Path = Path("sentinel/data/calibration.jsonl"),
    db_path: Path = Path("sentinel/data/calibration.db"),
) -> None:
    """
    Append a CalibrationResult to JSONL log file and SQLite database.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSONL
    with open(log_file, "a") as f:
        f.write(json.dumps({
            "timestamp": result.timestamp,
            "ticker": result.ticker,
            "predicted_direction": result.predicted_direction,
            "predicted_confidence": result.predicted_confidence,
            "actual_direction": result.actual_direction,
            "actual_price_change_pct": result.actual_price_change_pct,
            "correct": result.correct,
            "reasoning_tags": result.reasoning_tags,
        }) + "\n")
    
    # Write to SQLite
    _ensure_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO calibration
        (timestamp, ticker, predicted_direction, predicted_confidence,
         actual_direction, actual_price_change_pct, correct, reasoning_tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.timestamp,
        result.ticker,
        result.predicted_direction,
        result.predicted_confidence,
        result.actual_direction,
        result.actual_price_change_pct,
        1 if result.correct else 0,
        json.dumps(result.reasoning_tags),
    ))
    conn.commit()
    conn.close()


def compute_rolling_accuracy(
    window_days: int = 7,
    db_path: Path = Path("sentinel/data/calibration.db"),
    end_date: Optional[str] = None,
) -> RollingAccuracy:
    """
    Compute rolling accuracy over the past N days from the calibration database.
    
    Returns a RollingAccuracy namedtuple with overall and per-direction stats.
    """
    _ensure_db(db_path)
    
    if end_date is None:
        end_date = datetime.utcnow().isoformat()
    
    end_dt = datetime.fromisoformat(end_date)
    start_dt = end_dt - timedelta(days=window_days)
    start_date = start_dt.isoformat()
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Overall stats
    cursor.execute("""
        SELECT COUNT(*), SUM(correct) FROM calibration
        WHERE timestamp >= ? AND timestamp <= ?
    """, (start_date, end_date))
    row = cursor.fetchone()
    total = row[0] if row[0] else 0
    correct = row[1] if row[1] else 0
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-direction stats
    by_direction = {}
    for direction in ["UP", "DOWN", "NEUTRAL"]:
        cursor.execute("""
            SELECT COUNT(*), SUM(correct) FROM calibration
            WHERE timestamp >= ? AND timestamp <= ?
            AND predicted_direction = ?
        """, (start_date, end_date, direction))
        row = cursor.fetchone()
        dir_total = row[0] if row[0] else 0
        dir_correct = row[1] if row[1] else 0
        by_direction[direction] = {
            "total": dir_total,
            "correct": dir_correct,
            "accuracy": dir_correct / dir_total if dir_total > 0 else 0.0,
        }
    
    conn.close()
    
    return RollingAccuracy(
        window_days=window_days,
        start_date=start_date,
        end_date=end_date,
        total_predictions=total,
        correct_predictions=correct,
        accuracy=accuracy,
        by_direction=by_direction,
    )


def load_calibration_log(log_file: Path = Path("sentinel/data/calibration.jsonl")) -> list[CalibrationResult]:
    """
    Load all CalibrationResult entries from JSONL file.
    
    Returns a list of CalibrationResult namedtuples.
    """
    results = []
    if not log_file.exists():
        return results
    
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                results.append(CalibrationResult(
                    timestamp=obj["timestamp"],
                    ticker=obj["ticker"],
                    predicted_direction=obj["predicted_direction"],
                    predicted_confidence=obj["predicted_confidence"],
                    actual_direction=obj["actual_direction"],
                    actual_price_change_pct=obj["actual_price_change_pct"],
                    correct=obj["correct"],
                    reasoning_tags=obj["reasoning_tags"],
                ))
    
    return results


def get_accuracy_by_tag(
    tag: str,
    db_path: Path = Path("sentinel/data/calibration.db"),
) -> dict[str, int | float]:
    """
    Compute accuracy for predictions tagged with a specific reasoning tag.
    
    Returns dict with keys: total, correct, accuracy.
    """
    _ensure_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*), SUM(correct) FROM calibration
        WHERE reasoning_tags LIKE ?
    """, (
