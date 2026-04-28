"""
Heuristic Update Logger for Sentinel Judge Agent.

Appends CalibrationResult entries to a JSONL file and computes rolling 7-day
and 30-day accuracy metrics. Used by Judge to track prediction vs. actual
market moves and refine heuristics over time.
"""

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple


@dataclass
class CalibrationResult:
    """
    Single prediction calibration record.
    
    Fields:
      timestamp: ISO 8601 datetime when prediction was made.
      ticker: Stock symbol (e.g., 'AAPL').
      predicted_direction: 'UP', 'DOWN', or 'NEUTRAL'.
      predicted_magnitude: Float, expected % move (e.g., 2.5 for +2.5%).
      confidence_score: Float in [0, 1], model confidence.
      actual_direction: 'UP', 'DOWN', or 'NEUTRAL' (filled post-market close).
      actual_magnitude: Float, realized % move (filled post-market close).
      correct: Optional bool, True if predicted_direction matches actual_direction.
      mape_error: Optional float, mean absolute percentage error on magnitude.
      heuristic_tag: Optional str, name of heuristic rule that drove prediction.
    """
    timestamp: str
    ticker: str
    predicted_direction: str
    predicted_magnitude: float
    confidence_score: float
    actual_direction: Optional[str] = None
    actual_magnitude: Optional[float] = None
    correct: Optional[bool] = None
    mape_error: Optional[float] = None
    heuristic_tag: Optional[str] = None


class HeuristicLogger:
    """
    Manages JSONL log of CalibrationResult entries and computes rolling accuracy.
    """

    def __init__(self, log_path: str = "calibration_log.jsonl") -> None:
        """Initialize logger with file path."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.touch()

    def append_result(self, result: CalibrationResult) -> None:
        """Append a single CalibrationResult as JSON line."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def append_results(self, results: List[CalibrationResult]) -> None:
        """Append multiple CalibrationResult entries."""
        with open(self.log_path, "a") as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")

    def load_all_results(self) -> List[CalibrationResult]:
        """Load all CalibrationResult entries from JSONL file."""
        results = []
        if not self.log_path.exists():
            return results
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(CalibrationResult(**data))
        return results

    def update_result(self, index: int, actual_direction: str, actual_magnitude: float) -> None:
        """
        Update a result at a given index with actual market move.
        Recomputes correct and mape_error, then rewrites JSONL file.
        """
        results = self.load_all_results()
        if 0 <= index < len(results):
            result = results[index]
            result.actual_direction = actual_direction
            result.actual_magnitude = actual_magnitude
            result.correct = result.predicted_direction == actual_direction
            if result.predicted_magnitude != 0:
                result.mape_error = abs(
                    (result.actual_magnitude - result.predicted_magnitude)
                    / result.predicted_magnitude
                )
            else:
                result.mape_error = abs(result.actual_magnitude)
            results[index] = result
            self._rewrite_all(results)

    def _rewrite_all(self, results: List[CalibrationResult]) -> None:
        """Rewrite entire JSONL file (internal helper)."""
        with open(self.log_path, "w") as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")

    def compute_rolling_accuracy(
        self, days: int = 7
    ) -> Dict[str, float]:
        """
        Compute accuracy metrics over last N days.
        Returns dict with keys: total_count, correct_count, direction_accuracy, avg_confidence.
        """
        results = self.load_all_results()
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [
            r for r in results
            if r.correct is not None
            and datetime.fromisoformat(r.timestamp) >= cutoff
        ]

        if not recent:
            return {
                "total_count": 0,
                "correct_count": 0,
                "direction_accuracy": 0.0,
                "avg_confidence": 0.0,
                "days": days,
            }

        correct_count = sum(1 for r in recent if r.correct)
        direction_accuracy = correct_count / len(recent) if recent else 0.0
        avg_confidence = sum(r.confidence_score for r in recent) / len(recent) if recent else 0.0

        return {
            "total_count": len(recent),
            "correct_count": correct_count,
            "direction_accuracy": round(direction_accuracy, 4),
            "avg_confidence": round(avg_confidence, 4),
            "days": days,
        }

    def compute_accuracy_by_heuristic(self) -> Dict[str, Dict[str, float]]:
        """
        Compute accuracy metrics grouped by heuristic_tag.
        Returns dict[heuristic_tag] -> {total, correct, accuracy}.
        """
        results = self.load_all_results()
        by_heuristic: Dict[str, List[CalibrationResult]] = {}

        for r in results:
            if r.correct is not None:
                tag = r.heuristic_tag or "unknown"
                if tag not in by_heuristic:
                    by_heuristic[tag] = []
                by_heuristic[tag].append(r)

        metrics = {}
        for tag, tag_results in by_heuristic.items():
            correct_count = sum(1 for r in tag_results if r.correct)
            accuracy = correct_count / len(tag_results) if tag_results else 0.0
            metrics[tag] = {
                "total": len(tag_results),
                "correct": correct_count,
                "accuracy": round(accuracy, 4),
            }

        return metrics

    def compute_magnitude_error_stats(
        self, days: int = 30
    ) -> Dict[str, float]:
        """
        Compute mean absolute percentage error (MAPE) stats over last N days.
        Returns dict with keys: mean_mape, median_mape, count.
        """
        results = self.load_all_results()
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [
            r for r in results
            if r.mape_error is not None
            and datetime.fromisoformat(r.timestamp) >= cutoff
        ]

        if not recent:
            return {"mean_mape": 0.0, "median_mape": 0.0, "count": 0, "days": days}

        errors = sorted([r.mape_error for r in recent])
        mean_mape = sum(errors) / len(errors)
        median_mape = errors[len(errors) // 2]

        return {
            "mean_mape": round(mean_mape, 4),
            "median_
