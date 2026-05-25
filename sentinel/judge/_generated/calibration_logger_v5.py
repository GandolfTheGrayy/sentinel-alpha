"""
Calibration logger for Sentinel Judge pillar.

Appends CalibrationResult entries to a JSONL file and computes rolling 7-day
and 30-day accuracy metrics. Used by the Judge post-mortem flow to track
prediction vs. actual market moves and refine heuristics over time.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TypedDict


class CalibrationResult(TypedDict, total=False):
    """Schema for a single calibration event."""
    timestamp: str
    ticker: str
    predicted_direction: str
    predicted_confidence: float
    actual_direction: str
    actual_pct_move: float
    correct: bool
    notes: str


class RollingAccuracy(TypedDict):
    """Rolling accuracy metrics."""
    window_days: int
    start_date: str
    end_date: str
    total_predictions: int
    correct_predictions: int
    accuracy_pct: float


def ensure_calibration_file(log_path: str) -> str:
    """Create calibration JSONL file if it does not exist."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).touch(exist_ok=True)
    return log_path


def append_calibration_result(
    log_path: str,
    result: CalibrationResult,
) -> None:
    """Append a single CalibrationResult to the JSONL log."""
    ensure_calibration_file(log_path)
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_calibration_results(log_path: str) -> list[CalibrationResult]:
    """Load all CalibrationResult entries from JSONL file."""
    if not Path(log_path).exists():
        return []
    results = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_rolling_accuracy(
    log_path: str,
    window_days: int = 7,
    as_of_date: Optional[str] = None,
) -> RollingAccuracy:
    """
    Compute rolling accuracy over a window of days.
    
    Args:
        log_path: Path to JSONL calibration log.
        window_days: Number of days to include in rolling window.
        as_of_date: ISO date string; defaults to today.
    
    Returns:
        RollingAccuracy dict with metrics.
    """
    if as_of_date is None:
        as_of_date = datetime.utcnow().date().isoformat()
    
    as_of_dt = datetime.fromisoformat(as_of_date)
    start_dt = as_of_dt - timedelta(days=window_days)
    
    results = load_calibration_results(log_path)
    
    window_results = [
        r for r in results
        if start_dt.date().isoformat() <= r.get("timestamp", "")[:10] <= as_of_date
    ]
    
    total = len(window_results)
    correct = sum(1 for r in window_results if r.get("correct", False))
    accuracy_pct = (correct / total * 100) if total > 0 else 0.0
    
    return RollingAccuracy(
        window_days=window_days,
        start_date=start_dt.date().isoformat(),
        end_date=as_of_date,
        total_predictions=total,
        correct_predictions=correct,
        accuracy_pct=accuracy_pct,
    )


def compute_all_rolling_accuracies(
    log_path: str,
    as_of_date: Optional[str] = None,
) -> dict[int, RollingAccuracy]:
    """
    Compute rolling accuracies for standard windows (7, 30 days).
    
    Returns:
        Dict mapping window_days to RollingAccuracy.
    """
    return {
        window: compute_rolling_accuracy(log_path, window, as_of_date)
        for window in [7, 30]
    }


def get_accuracy_by_ticker(
    log_path: str,
    ticker: Optional[str] = None,
) -> dict[str, dict]:
    """
    Compute per-ticker accuracy metrics.
    
    Args:
        log_path: Path to JSONL calibration log.
        ticker: If provided, return metrics for only that ticker.
    
    Returns:
        Dict mapping ticker to {total, correct, accuracy_pct}.
    """
    results = load_calibration_results(log_path)
    ticker_stats = {}
    
    for r in results:
        t = r.get("ticker", "UNKNOWN")
        if ticker and t != ticker:
            continue
        
        if t not in ticker_stats:
            ticker_stats[t] = {"total": 0, "correct": 0}
        
        ticker_stats[t]["total"] += 1
        if r.get("correct", False):
            ticker_stats[t]["correct"] += 1
    
    for t in ticker_stats:
        total = ticker_stats[t]["total"]
        correct = ticker_stats[t]["correct"]
        ticker_stats[t]["accuracy_pct"] = (correct / total * 100) if total > 0 else 0.0
    
    return ticker_stats


def format_calibration_report(
    log_path: str,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Format a human-readable calibration report.
    
    Returns:
        Multi-line report string.
    """
    if as_of_date is None:
        as_of_date = datetime.utcnow().date().isoformat()
    
    rolling = compute_all_rolling_accuracies(log_path, as_of_date)
    ticker_acc = get_accuracy_by_ticker(log_path)
    
    lines = [
        "=" * 60,
        f"Calibration Report (as of {as_of_date})",
        "=" * 60,
        "",
        "Rolling Accuracy:",
    ]
    
    for window_days in sorted(rolling.keys()):
        m = rolling[window_days]
        lines.append(
            f"  {m['window_days']:2d}-day: {m['correct_predictions']:3d}/{m['total_predictions']:3d} "
            f"({m['accuracy_pct']:6.2f}%) [{m['start_date']} to {m['end_date']}]"
        )
    
    lines.extend(["", "Per-Ticker Accuracy:"])
    for ticker in sorted(ticker_acc.keys()):
        stats = ticker_acc[ticker]
        lines.append(
            f"  {ticker:8s}: {stats['correct']:3d}/{stats['total']:3d} "
            f"({stats['accuracy_pct']:6.2f}%)"
        )
    
    lines.append("=" * 60)
    return "\n".join(lines)
