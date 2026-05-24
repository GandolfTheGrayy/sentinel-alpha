"""
Calibration Logger for Sentinel Sentiment Engine.

This module manages persistent logging of CalibrationResult entries (predicted vs. actual
market moves) to a JSONL file and computes rolling 7-day and 30-day accuracy metrics.
Used by Judge post-mortems to track prediction quality over time and feed into heuristic
refinement loops.

Role in Sentinel: Judge pillar — post-mortem instrumentation that underpins confidence
score weighting and anomaly detection in future prediction cycles.
"""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


@dataclass
class CalibrationResult:
    """Single prediction outcome: ticker, prediction, actual move, timestamp, confidence."""

    ticker: str
    predicted_direction: str  # "up", "down", "neutral"
    actual_direction: str  # "up", "down", "neutral"
    predicted_confidence: float  # 0.0–1.0
    actual_price_change_pct: float
    predicted_price_target: Optional[float]
    actual_price_close: float
    timestamp: str  # ISO 8601
    model_version: str  # e.g., "claude-sonnet-4-6"
    notes: str = ""


class CalibrationLogger:
    """Appends CalibrationResult entries to JSONL and computes rolling accuracy metrics."""

    def __init__(self, log_path: Path | str = "sentinel_calibration.jsonl"):
        """Initialize logger with JSONL file path."""
        self.log_path = Path(log_path)
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create JSONL file if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.touch()

    def append(self, result: CalibrationResult) -> None:
        """Append a CalibrationResult entry as JSON line."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def _load_entries(
        self, days_back: Optional[int] = None
    ) -> list[CalibrationResult]:
        """Load all entries (or last N days) from JSONL file."""
        entries = []
        if not self.log_path.exists():
            return entries

        cutoff = None
        if days_back:
            cutoff = datetime.utcnow() - timedelta(days=days_back)

        with open(self.log_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    result = CalibrationResult(**data)
                    if cutoff:
                        entry_time = datetime.fromisoformat(result.timestamp)
                        if entry_time < cutoff:
                            continue
                    entries.append(result)
                except (json.JSONDecodeError, TypeError):
                    pass

        return entries

    def accuracy_7d(self) -> dict[str, float]:
        """Compute 7-day rolling accuracy metrics: overall, by direction, by confidence band."""
        return self._compute_accuracy(days_back=7)

    def accuracy_30d(self) -> dict[str, float]:
        """Compute 30-day rolling accuracy metrics: overall, by direction, by confidence band."""
        return self._compute_accuracy(days_back=30)

    def accuracy_all_time(self) -> dict[str, float]:
        """Compute all-time accuracy metrics across entire log."""
        return self._compute_accuracy(days_back=None)

    def _compute_accuracy(self, days_back: Optional[int] = None) -> dict[str, float]:
        """Compute accuracy metrics for a time window."""
        entries = self._load_entries(days_back=days_back)
        if not entries:
            return {
                "total_predictions": 0,
                "correct": 0,
                "overall_accuracy": 0.0,
                "up_accuracy": 0.0,
                "down_accuracy": 0.0,
                "neutral_accuracy": 0.0,
                "high_confidence_accuracy": 0.0,
                "low_confidence_accuracy": 0.0,
                "avg_predicted_confidence": 0.0,
            }

        total = len(entries)
        correct = sum(
            1 for e in entries if e.predicted_direction == e.actual_direction
        )

        # By direction
        up_total = sum(1 for e in entries if e.predicted_direction == "up")
        up_correct = sum(
            1
            for e in entries
            if e.predicted_direction == "up" and e.actual_direction == "up"
        )

        down_total = sum(1 for e in entries if e.predicted_direction == "down")
        down_correct = sum(
            1
            for e in entries
            if e.predicted_direction == "down" and e.actual_direction == "down"
        )

        neutral_total = sum(1 for e in entries if e.predicted_direction == "neutral")
        neutral_correct = sum(
            1
            for e in entries
            if e.predicted_direction == "neutral"
            and e.actual_direction == "neutral"
        )

        # By confidence band
        high_conf = [e for e in entries if e.predicted_confidence >= 0.7]
        high_correct = sum(
            1
            for e in high_conf
            if e.predicted_direction == e.actual_direction
        )

        low_conf = [e for e in entries if e.predicted_confidence < 0.7]
        low_correct = sum(
            1 for e in low_conf if e.predicted_direction == e.actual_direction
        )

        avg_conf = sum(e.predicted_confidence for e in entries) / total if total else 0

        return {
            "total_predictions": total,
            "correct": correct,
            "overall_accuracy": correct / total if total else 0.0,
            "up_accuracy": up_correct / up_total if up_total else 0.0,
            "down_accuracy": down_correct / down_total if down_total else 0.0,
            "neutral_accuracy": neutral_correct / neutral_total
            if neutral_total
            else 0.0,
            "high_confidence_accuracy": high_correct / len(high_conf)
            if high_conf
            else 0.0,
            "low_confidence_accuracy": low_correct / len(low_conf)
            if low_conf
            else 0.0,
            "avg_predicted_confidence": avg_conf,
        }

    def get_entries_in_range(
        self, start_date: str, end_date: str
    ) -> list[CalibrationResult]:
        """Fetch entries within ISO 8601 date range (inclusive)."""
        entries = self._load_entries(days_back=None)
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)

        return [
            e
            for e in entries
            if start <= datetime.fromisoformat(e.timestamp) <= end
        ]

    def export_csv(self, output_path: Path | str) -> None:
        """Export all entries as CSV for external analysis."""
        import csv

        entries = self._load_entries(days_back=None)
        if not entries:
            return

        output_path = Path(output_path)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(entries[0]).keys())
            writer.writeheader()
            for entry in entries:
                writer.writerow(asdict(entry))

    def clear_old_entries(self, days_old: int) -> int:
        """Remove entries older than N days; return count deleted."""
        entries = self._load_entries(days_back=None)
        cutoff = datetime.utcnow() - timedelta(days=days_old)
