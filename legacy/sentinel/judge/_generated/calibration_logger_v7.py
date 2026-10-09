"""
Calibration logger for Sentinel Sentiment Engine.

This module manages persistent logging of prediction vs. actual market outcomes
(CalibrationResult entries) to a JSONL file, and computes rolling accuracy metrics
(7-day and 30-day windows). Used by Judge to track heuristic refinement and
anomaly detection over time.

Appends to sentinel/data/calibration.jsonl and exposes rolling accuracy queries
for postmortem analysis and confidence calibration.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class CalibrationResult:
    """
    Represents a single prediction outcome: ticker, predicted direction,
    actual direction, confidence, and timestamp.
    """

    def __init__(
        self,
        ticker: str,
        predicted_direction: str,
        actual_direction: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
    ):
        """Initialize a CalibrationResult."""
        self.ticker = ticker
        self.predicted_direction = predicted_direction  # "up", "down", "neutral"
        self.actual_direction = actual_direction  # "up", "down", "neutral"
        self.confidence = confidence  # float [0, 1]
        self.timestamp = timestamp or datetime.utcnow()

    def is_correct(self) -> bool:
        """Return True if predicted direction matches actual direction."""
        return self.predicted_direction == self.actual_direction

    def to_dict(self) -> dict:
        """Serialize CalibrationResult to dict."""
        return {
            "ticker": self.ticker,
            "predicted_direction": self.predicted_direction,
            "actual_direction": self.actual_direction,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "correct": self.is_correct(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        """Deserialize CalibrationResult from dict."""
        result = cls(
            ticker=data["ticker"],
            predicted_direction=data["predicted_direction"],
            actual_direction=data["actual_direction"],
            confidence=data["confidence"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
        return result


class CalibrationLogger:
    """
    Appends CalibrationResult entries to a JSONL file and computes rolling
    accuracy metrics over 7-day and 30-day windows.
    """

    def __init__(self, log_path: str = "sentinel/data/calibration.jsonl"):
        """Initialize the logger with a target JSONL file path."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: CalibrationResult) -> None:
        """Append a CalibrationResult to the JSONL log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    def read_all(self) -> list[CalibrationResult]:
        """Read all CalibrationResult entries from the JSONL log."""
        results = []
        if not self.log_path.exists():
            return results
        with open(self.log_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    results.append(CalibrationResult.from_dict(data))
        return results

    def accuracy_in_window(
        self, days: int, ticker: Optional[str] = None
    ) -> dict:
        """
        Compute accuracy metrics over a rolling window of the last N days.

        Returns dict with keys: total_count, correct_count, accuracy, avg_confidence.
        If ticker is provided, filter to that ticker only.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        results = self.read_all()
        filtered = [
            r for r in results
            if r.timestamp >= cutoff and (ticker is None or r.ticker == ticker)
        ]

        if not filtered:
            return {
                "window_days": days,
                "ticker": ticker,
                "total_count": 0,
                "correct_count": 0,
                "accuracy": None,
                "avg_confidence": None,
            }

        correct_count = sum(1 for r in filtered if r.is_correct())
        total_count = len(filtered)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        avg_confidence = (
            sum(r.confidence for r in filtered) / total_count
            if total_count > 0
            else 0.0
        )

        return {
            "window_days": days,
            "ticker": ticker,
            "total_count": total_count,
            "correct_count": correct_count,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
        }

    def accuracy_7d(self, ticker: Optional[str] = None) -> dict:
        """Compute 7-day rolling accuracy."""
        return self.accuracy_in_window(7, ticker)

    def accuracy_30d(self, ticker: Optional[str] = None) -> dict:
        """Compute 30-day rolling accuracy."""
        return self.accuracy_in_window(30, ticker)

    def accuracy_by_ticker(self, days: int) -> dict[str, dict]:
        """
        Compute accuracy metrics per ticker over the last N days.

        Returns dict mapping ticker -> accuracy metrics.
        """
        results = self.read_all()
        cutoff = datetime.utcnow() - timedelta(days=days)
        filtered = [r for r in results if r.timestamp >= cutoff]

        ticker_results = {}
        for result in filtered:
            if result.ticker not in ticker_results:
                ticker_results[result.ticker] = []
            ticker_results[result.ticker].append(result)

        metrics = {}
        for ticker, ticker_list in ticker_results.items():
            correct_count = sum(1 for r in ticker_list if r.is_correct())
            total_count = len(ticker_list)
            accuracy = (
                correct_count / total_count if total_count > 0 else 0.0
            )
            avg_confidence = (
                sum(r.confidence for r in ticker_list) / total_count
                if total_count > 0
                else 0.0
            )
            metrics[ticker] = {
                "total_count": total_count,
                "correct_count": correct_count,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
            }

        return metrics

    def recent_predictions(self, limit: int = 10) -> list[dict]:
        """Return the most recent N predictions as dicts."""
        results = self.read_all()
        return [r.to_dict() for r in results[-limit:]]


if __name__ == "__main__":
    logger = CalibrationLogger()

    sample_results = [
        CalibrationResult("AAPL", "up", "up", 0.85),
        CalibrationResult("TSLA", "down", "up", 0.72),
        CalibrationResult("AAPL", "up", "up", 0.90),
        CalibrationResult("GOOGL", "neutral", "down", 0.65),
    ]

    for result in sample_results:
        logger.append(result)

    print("7-day accuracy (all tickers):", logger.accuracy_7d())
    print("30-day accuracy (all tickers):", logger.accuracy_30d())
    print("7-day accuracy by ticker:", logger.accuracy_by_ticker(7))
    print("Recent predictions:", logger.recent_predictions(5))
