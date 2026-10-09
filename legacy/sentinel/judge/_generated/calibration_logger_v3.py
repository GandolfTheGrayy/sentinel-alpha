"""
Calibration Logger for Sentinel Sentiment Engine.

Logs CalibrationResult entries (predicted vs. actual market moves) to a JSONL file
and computes rolling 7-day and 30-day accuracy metrics. Feeds heuristic refinement
in the Judge pillar's post-mortem workflow.

This module provides:
  - append_calibration(): Write prediction outcomes to disk.
  - compute_rolling_accuracy(): Calculate windowed accuracy over recent history.
  - load_calibration_history(): Retrieve all logged calibrations for analysis.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationResult:
    """Result of a single prediction vs. actual outcome."""

    timestamp: str  # ISO 8601 datetime
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "HOLD"
    predicted_confidence: float  # 0.0 to 1.0
    actual_direction: str  # "UP", "DOWN", "HOLD"
    actual_price_change_pct: float  # percent change
    correct: bool  # True if predicted_direction matches actual_direction
    notes: Optional[str] = None


def append_calibration(
    result: CalibrationResult, log_path: str = "data/calibration_log.jsonl"
) -> None:
    """Append a CalibrationResult entry to the JSONL calibration log.

    Args:
        result: CalibrationResult dataclass instance to log.
        log_path: Path to JSONL file (created if missing).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(asdict(result)) + "\n")


def load_calibration_history(
    log_path: str = "data/calibration_log.jsonl",
) -> list[CalibrationResult]:
    """Load all calibration entries from JSONL history.

    Args:
        log_path: Path to JSONL calibration log.

    Returns:
        List of CalibrationResult instances, oldest first.
    """
    if not os.path.exists(log_path):
        return []

    results = []
    with open(log_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(CalibrationResult(**data))
    return results


def compute_rolling_accuracy(
    log_path: str = "data/calibration_log.jsonl",
    days: int = 7,
    min_samples: int = 1,
) -> dict:
    """Compute rolling accuracy metrics over a time window.

    Args:
        log_path: Path to JSONL calibration log.
        days: Window size in days (e.g., 7 for weekly, 30 for monthly).
        min_samples: Minimum predictions required to return a result.

    Returns:
        Dict with keys:
          - 'window_days': int, the requested window
          - 'cutoff_time': str, ISO 8601 cutoff (now - days)
          - 'total_predictions': int, count in window
          - 'correct_predictions': int, count where correct==True
          - 'accuracy_pct': float, 0-100 or None if < min_samples
          - 'ticker_breakdown': dict mapping ticker -> {'total': int, 'correct': int, 'pct': float}
    """
    history = load_calibration_history(log_path)
    cutoff = datetime.utcnow() - timedelta(days=days)
    cutoff_str = cutoff.isoformat()

    in_window = [
        r for r in history if datetime.fromisoformat(r.timestamp) >= cutoff
    ]

    total = len(in_window)
    correct = sum(1 for r in in_window if r.correct)

    ticker_stats = {}
    for result in in_window:
        if result.ticker not in ticker_stats:
            ticker_stats[result.ticker] = {"total": 0, "correct": 0}
        ticker_stats[result.ticker]["total"] += 1
        if result.correct:
            ticker_stats[result.ticker]["correct"] += 1

    for ticker in ticker_stats:
        t = ticker_stats[ticker]["total"]
        c = ticker_stats[ticker]["correct"]
        ticker_stats[ticker]["pct"] = round(100 * c / t, 2) if t > 0 else 0.0

    accuracy_pct = None
    if total >= min_samples:
        accuracy_pct = round(100 * correct / total, 2)

    return {
        "window_days": days,
        "cutoff_time": cutoff_str,
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy_pct": accuracy_pct,
        "ticker_breakdown": ticker_stats,
    }


def compute_all_windows(
    log_path: str = "data/calibration_log.jsonl",
) -> dict:
    """Compute accuracy metrics for both 7-day and 30-day rolling windows.

    Args:
        log_path: Path to JSONL calibration log.

    Returns:
        Dict with keys 'window_7d' and 'window_30d', each containing full accuracy report.
    """
    return {
        "window_7d": compute_rolling_accuracy(log_path, days=7),
        "window_30d": compute_rolling_accuracy(log_path, days=30),
    }
