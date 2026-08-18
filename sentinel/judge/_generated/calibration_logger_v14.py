"""
Calibration Logger — Sentinel's heuristic update & accuracy tracking system.

This module maintains a JSONL ledger of CalibrationResult entries, recording
each prediction's outcome vs. actual market move. It computes rolling 7-day
and 30-day accuracy metrics to inform Judge's heuristic refinement loop.

Integrates with Judge's post-mortem flow: after resolver confirms actual moves,
calibration_logger appends the result and emits rolling stats for strategy tuning.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationResult:
    """Single prediction outcome: what was predicted, what actually happened."""
    
    ticker: str
    prediction_date: str  # ISO 8601
    prediction_direction: str  # "up", "down", "hold"
    prediction_confidence: float  # 0.0–1.0
    actual_direction: str  # "up", "down", "hold" or "unknown"
    actual_price_move_pct: float  # e.g., 2.5 for +2.5%
    was_correct: bool  # prediction_direction == actual_direction
    resolution_date: str  # ISO 8601, when outcome confirmed
    strategy_name: str  # e.g., "baseline_momentum", "linguist_fusion"
    notes: str = ""  # Optional free-form context


def ensure_calibration_db(db_path: Path = Path("sentinel_data/calibration.db")) -> sqlite3.Connection:
    """Create or open calibration SQLite DB with CalibrationResult schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calibration_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            prediction_direction TEXT NOT NULL,
            prediction_confidence REAL NOT NULL,
            actual_direction TEXT,
            actual_price_move_pct REAL,
            was_correct BOOLEAN,
            resolution_date TEXT,
            strategy_name TEXT NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rolling_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at TEXT NOT NULL,
            window_days INTEGER,
            strategy_name TEXT,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            avg_confidence REAL
        )
    """)
    conn.commit()
    return conn


def log_calibration_result(
    result: CalibrationResult,
    db_path: Path = Path("sentinel_data/calibration.db")
) -> int:
    """Append CalibrationResult to DB; return inserted row ID."""
    conn = ensure_calibration_db(db_path)
    cursor = conn.cursor()
    data = asdict(result)
    cols = ", ".join(data.keys())
    placeholders = ", ".join(["?"] * len(data))
    cursor.execute(
        f"INSERT INTO calibration_results ({cols}) VALUES ({placeholders})",
        tuple(data.values())
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def compute_rolling_metrics(
    window_days: int = 7,
    strategy_name: Optional[str] = None,
    db_path: Path = Path("sentinel_data/calibration.db")
) -> dict:
    """Compute accuracy & confidence metrics over last N days; return dict with stats."""
    conn = ensure_calibration_db(db_path)
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=window_days)).isoformat()
    
    if strategy_name:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(prediction_confidence) as avg_conf
            FROM calibration_results
            WHERE resolution_date >= ? AND strategy_name = ?
        """, (cutoff, strategy_name))
    else:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                AVG(prediction_confidence) as avg_conf
            FROM calibration_results
            WHERE resolution_date >= ?
        """, (cutoff,))
    
    row = cursor.fetchone()
    total = row[0] or 0
    correct = row[1] or 0
    avg_conf = row[2] or 0.0
    accuracy = (correct / total) if total > 0 else 0.0
    
    metrics = {
        "window_days": window_days,
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "computed_at": datetime.utcnow().isoformat(),
    }
    
    if strategy_name:
        metrics["strategy_name"] = strategy_name
    
    conn.close()
    return metrics


def write_rolling_metrics_checkpoint(
    window_days_list: list[int] = None,
    strategy_name: Optional[str] = None,
    db_path: Path = Path("sentinel_data/calibration.db")
) -> None:
    """Compute & persist rolling metrics for multiple windows to DB."""
    if window_days_list is None:
        window_days_list = [7, 30]
    
    conn = ensure_calibration_db(db_path)
    cursor = conn.cursor()
    
    now = datetime.utcnow().isoformat()
    
    for days in window_days_list:
        metrics = compute_rolling_metrics(window_days=days, strategy_name=strategy_name, db_path=db_path)
        cursor.execute("""
            INSERT INTO rolling_metrics (computed_at, window_days, strategy_name, total_predictions, correct_predictions, accuracy, avg_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            now,
            metrics["window_days"],
            strategy_name,
            metrics["total_predictions"],
            metrics["correct_predictions"],
            metrics["accuracy"],
            metrics["avg_confidence"]
        ))
    
    conn.commit()
    conn.close()


def export_calibration_jsonl(
    output_path: Path = Path("sentinel_data/calibration.jsonl"),
    db_path: Path = Path("sentinel_data/calibration.db")
) -> None:
    """Export all CalibrationResult rows to JSONL for analysis/audit."""
    conn = ensure_calibration_db(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, prediction_date, prediction_direction, prediction_confidence,
               actual_direction, actual_price_move_pct, was_correct, resolution_date,
               strategy_name, notes
        FROM calibration_results
        ORDER BY created_at ASC
    """)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in cursor.fetchall():
            entry = {
                "ticker": row[0],
                "prediction_date": row[1],
                "prediction_direction": row[2],
                "prediction_confidence": row[3],
                "actual_direction": row[4],
                "actual_price_move_pct": row[5],
                "was_correct": bool(row[6]),
                "resolution_date": row[7],
                "strategy_name": row[8],
                "notes": row[9] or "",
            }
            f.write(json.dumps(entry) + "\n")
    
    conn
