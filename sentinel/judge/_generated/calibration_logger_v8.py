"""
Sentinel Calibration Logger — Heuristic Update & Rolling Accuracy Metrics

Appends CalibrationResult entries to a JSONL file tracking daily post-mortem outcomes.
Computes rolling 7-day and 30-day accuracy metrics to detect heuristic drift and
guide Judge refinement. Integrates with resolver.py outcomes.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple, Optional, List, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CalibrationResult:
    """Single daily calibration record: prediction vs. actual outcome."""
    timestamp: str  # ISO 8601
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "HOLD"
    predicted_confidence: float  # [0.0, 1.0]
    actual_direction: str  # "UP", "DOWN", "HOLD"
    actual_price_change_pct: float
    correct: bool  # True if predicted_direction matched actual_direction
    strategy_used: str  # "baseline_momentum", "baseline_contrarian", "baseline_ensemble", "claude"
    notes: Optional[str] = None


class CalibrationLogger:
    """Appends daily predictions to JSONL, computes rolling accuracy metrics."""

    def __init__(self, log_path: str = "sentinel/data/calibration.jsonl") -> None:
        """Initialize logger with JSONL file path."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append_result(self, result: CalibrationResult) -> None:
        """Append a single calibration result to JSONL file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def append_results_batch(self, results: List[CalibrationResult]) -> None:
        """Append multiple calibration results to JSONL file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + "\n")

    def read_all_results(self) -> List[CalibrationResult]:
        """Read all calibration results from JSONL file."""
        results = []
        if not self.log_path.exists():
            return results
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(CalibrationResult(**data))
        return results

    def compute_accuracy_window(
        self, results: List[CalibrationResult], days: int
    ) -> Dict[str, Any]:
        """Compute accuracy metrics for last N days across all tickers."""
        if not results:
            return {
                "window_days": days,
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "by_strategy": {},
                "by_ticker": {},
                "by_direction": {},
            }

        cutoff = datetime.utcnow() - timedelta(days=days)
        windowed = [
            r for r in results if datetime.fromisoformat(r.timestamp) >= cutoff
        ]

        total = len(windowed)
        correct = sum(1 for r in windowed if r.correct)
        accuracy = (correct / total * 100) if total > 0 else 0.0

        # Break down by strategy
        by_strategy = {}
        for strategy in set(r.strategy_used for r in windowed):
            strat_results = [r for r in windowed if r.strategy_used == strategy]
            strat_correct = sum(1 for r in strat_results if r.correct)
            by_strategy[strategy] = {
                "count": len(strat_results),
                "correct": strat_correct,
                "accuracy": (
                    strat_correct / len(strat_results) * 100
                    if len(strat_results) > 0
                    else 0.0
                ),
            }

        # Break down by ticker
        by_ticker = {}
        for ticker in set(r.ticker for r in windowed):
            tick_results = [r for r in windowed if r.ticker == ticker]
            tick_correct = sum(1 for r in tick_results if r.correct)
            by_ticker[ticker] = {
                "count": len(tick_results),
                "correct": tick_correct,
                "accuracy": (
                    tick_correct / len(tick_results) * 100
                    if len(tick_results) > 0
                    else 0.0
                ),
            }

        # Break down by predicted direction
        by_direction = {}
        for direction in {"UP", "DOWN", "HOLD"}:
            dir_results = [r for r in windowed if r.predicted_direction == direction]
            if dir_results:
                dir_correct = sum(1 for r in dir_results if r.correct)
                by_direction[direction] = {
                    "count": len(dir_results),
                    "correct": dir_correct,
                    "accuracy": (
                        dir_correct / len(dir_results) * 100
                        if len(dir_results) > 0
                        else 0.0
                    ),
                }

        return {
            "window_days": days,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 2),
            "by_strategy": by_strategy,
            "by_ticker": by_ticker,
            "by_direction": by_direction,
        }

    def rolling_metrics(self) -> Dict[str, Any]:
        """Compute 7-day and 30-day rolling accuracy metrics."""
        all_results = self.read_all_results()
        return {
            "as_of": datetime.utcnow().isoformat(),
            "7_day": self.compute_accuracy_window(all_results, 7),
            "30_day": self.compute_accuracy_window(all_results, 30),
            "all_time": self.compute_accuracy_window(all_results, 999999),
        }

    def get_strategy_ranking(self) -> List[Dict[str, Any]]:
        """Rank strategies by 30-day accuracy."""
        metrics = self.rolling_metrics()
        strategies = metrics["30_day"]["by_strategy"]
        ranked = sorted(
            [
                {
                    "strategy": name,
                    "accuracy": data["accuracy"],
                    "count": data["count"],
                }
                for name, data in strategies.items()
            ],
            key=lambda x: x["accuracy"],
            reverse=True,
        )
        return ranked

    def flag_underperforming_strategies(self, threshold: float = 40.0) -> List[str]:
        """Return list of strategies below accuracy threshold in 30-day window."""
        metrics = self.rolling_metrics()
        strategies = metrics["30_day"]["by_strategy"]
        underperforming = [
            name
            for name, data in strategies.items()
            if data["accuracy"] < threshold and data["count"] >= 5
        ]
        return underperforming

    def export_metrics_json(self, output_path: str = "sentinel/data/metrics.json") -> None:
        """Export rolling metrics and strategy rankings to JSON file."""
        metrics = self.rolling_metrics()
        rankings = self.get_strategy_ranking()
        output_data = {
            "metrics": metrics,
            "strategy_rankings": rankings,
            "exported_at": datetime.utcnow().isoformat(),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)


def create
