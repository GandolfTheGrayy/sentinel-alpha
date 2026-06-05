"""
Calibration Logger — Sentinel's heuristic update and accuracy tracking system.

This module maintains a JSONL file of CalibrationResult entries, each recording
a prediction vs. actual outcome for a ticker. It computes rolling 7-day and 30-day
accuracy metrics, enabling the Judge to refine its prediction heuristics over time.

Integrated into sentinel/judge/resolver.py post-mortem flow.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class CalibrationResult:
    """A single prediction calibration record."""

    ticker: str
    prediction_date: str  # ISO format YYYY-MM-DD
    predicted_direction: str  # "up", "down", "neutral"
    predicted_magnitude: float  # percentage change, e.g., 2.5
    actual_direction: str  # "up", "down", "neutral"
    actual_magnitude: float  # percentage change
    confidence_score: float  # 0.0–1.0
    correct: bool  # prediction_direction == actual_direction
    mae: float  # |predicted_magnitude - actual_magnitude|
    notes: Optional[str] = None  # free-form reasoning


class CalibrationLogger:
    """Logs prediction outcomes and computes rolling accuracy metrics."""

    def __init__(self, log_path: str = "data/calibration.jsonl") -> None:
        """
        Initialize the logger with a JSONL file path.

        Args:
            log_path: Path to the JSONL calibration log file.
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: CalibrationResult) -> None:
        """Append a CalibrationResult to the JSONL log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(result)) + "\n")

    def load_records(self) -> list[CalibrationResult]:
        """Load all CalibrationResult records from the JSONL file."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(CalibrationResult(**data))
        return records

    def accuracy_by_ticker(self, days: int = 7) -> dict[str, float]:
        """
        Compute accuracy (% correct predictions) per ticker over the last N days.

        Args:
            days: Window size in days.

        Returns:
            Dictionary mapping ticker to accuracy (0.0–1.0).
        """
        records = self.load_records()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = [
            r
            for r in records
            if r.prediction_date >= cutoff
        ]

        if not filtered:
            return {}

        df = pd.DataFrame([(r.ticker, r.correct) for r in filtered], columns=["ticker", "correct"])
        return (df.groupby("ticker")["correct"].mean()).to_dict()

    def overall_accuracy(self, days: int = 7) -> float:
        """
        Compute overall accuracy across all tickers in the last N days.

        Args:
            days: Window size in days.

        Returns:
            Accuracy as a fraction (0.0–1.0).
        """
        records = self.load_records()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = [
            r
            for r in records
            if r.prediction_date >= cutoff
        ]

        if not filtered:
            return 0.0

        correct_count = sum(1 for r in filtered if r.correct)
        return correct_count / len(filtered)

    def mae_by_ticker(self, days: int = 7) -> dict[str, float]:
        """
        Compute mean absolute error (MAE) per ticker over the last N days.

        Args:
            days: Window size in days.

        Returns:
            Dictionary mapping ticker to MAE.
        """
        records = self.load_records()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = [
            r
            for r in records
            if r.prediction_date >= cutoff
        ]

        if not filtered:
            return {}

        df = pd.DataFrame([(r.ticker, r.mae) for r in filtered], columns=["ticker", "mae"])
        return (df.groupby("ticker")["mae"].mean()).to_dict()

    def rolling_metrics(self, days: int = 7) -> dict:
        """
        Compute a summary of rolling metrics over the last N days.

        Args:
            days: Window size in days.

        Returns:
            Dictionary with keys: overall_accuracy, ticker_accuracy, mae_by_ticker, sample_count.
        """
        records = self.load_records()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = [
            r
            for r in records
            if r.prediction_date >= cutoff
        ]

        if not filtered:
            return {
                "overall_accuracy": 0.0,
                "ticker_accuracy": {},
                "mae_by_ticker": {},
                "sample_count": 0,
            }

        overall_acc = sum(1 for r in filtered if r.correct) / len(filtered)
        ticker_acc = self.accuracy_by_ticker(days)
        mae = self.mae_by_ticker(days)

        return {
            "overall_accuracy": overall_acc,
            "ticker_accuracy": ticker_acc,
            "mae_by_ticker": mae,
            "sample_count": len(filtered),
        }

    def confidence_calibration(self, days: int = 7) -> dict[str, float]:
        """
        Compute accuracy bucketed by confidence score deciles.

        Args:
            days: Window size in days.

        Returns:
            Dictionary mapping confidence bucket (e.g., "0.0–0.1") to accuracy.
        """
        records = self.load_records()
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()[:10]
        filtered = [
            r
            for r in records
            if r.prediction_date >= cutoff
        ]

        if not filtered:
            return {}

        df = pd.DataFrame(
            [
                (r.confidence_score, r.correct)
                for r in filtered
            ],
            columns=["confidence", "correct"],
        )

        # Create decile buckets
        df["bucket"] = pd.cut(
            df["confidence"],
            bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=[
                "0.0–0.1",
                "0.1–0.2",
                "0.2–0.3",
                "0.3–0.4",
                "0.4–0.5",
                "0.5–0.6",
                "0.6–0.7",
                "0.7–0.8",
                "0.8–0.9",
                "0.9–1.0",
            ],
        )

        return (df.groupby("bucket")["correct"].mean()).to_dict()
</
