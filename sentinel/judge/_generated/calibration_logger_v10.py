"""
Calibration logger for Sentinel Sentiment Engine.

This module manages heuristic refinement by logging CalibrationResult entries
(predicted vs. actual market moves) to a JSONL file and computing rolling
7-day and 30-day accuracy metrics. Integrates with the daily post-mortem
workflow to track prediction quality over time and flag systematic biases.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class CalibrationResult:
    """Record of a single prediction vs. actual market outcome."""
    
    timestamp: str  # ISO 8601 datetime when prediction was made
    ticker: str
    predicted_direction: str  # "UP", "DOWN", "HOLD"
    predicted_confidence: float  # 0.0 to 1.0
    actual_direction: str  # "UP", "DOWN", "HOLD"
    actual_price_change: float  # percentage change over observation window
    correct: bool  # True if predicted_direction matched actual_direction
    reasoning: Optional[str] = None  # brief explanation for post-mortem review


class CalibrationLogger:
    """Logs prediction outcomes and computes rolling accuracy metrics."""
    
    def __init__(self, db_path: str = "sentinel_calibration.db") -> None:
        """Initialize calibration logger with SQLite backend.
        
        Args:
            db_path: Path to SQLite database for persistent storage.
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Create calibration table if it does not exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticker TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                actual_direction TEXT NOT NULL,
                actual_price_change REAL NOT NULL,
                correct INTEGER NOT NULL,
                reasoning TEXT,
                recorded_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def log_result(self, result: CalibrationResult) -> None:
        """Append a calibration result to the database.
        
        Args:
            result: CalibrationResult dataclass instance to persist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO calibration (
                timestamp, ticker, predicted_direction, predicted_confidence,
                actual_direction, actual_price_change, correct, reasoning, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.timestamp,
            result.ticker,
            result.predicted_direction,
            result.predicted_confidence,
            result.actual_direction,
            result.actual_price_change,
            int(result.correct),
            result.reasoning,
            datetime.utcnow().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def log_batch(self, results: list[CalibrationResult]) -> None:
        """Append multiple calibration results in one transaction.
        
        Args:
            results: List of CalibrationResult instances to persist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        recorded_at = datetime.utcnow().isoformat()
        for result in results:
            cursor.execute("""
                INSERT INTO calibration (
                    timestamp, ticker, predicted_direction, predicted_confidence,
                    actual_direction, actual_price_change, correct, reasoning, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp,
                result.ticker,
                result.predicted_direction,
                result.predicted_confidence,
                result.actual_direction,
                result.actual_price_change,
                int(result.correct),
                result.reasoning,
                recorded_at
            ))
        conn.commit()
        conn.close()
    
    def accuracy_7d(self) -> dict:
        """Compute 7-day rolling accuracy and confidence-stratified metrics.
        
        Returns:
            Dictionary with overall accuracy, per-direction accuracy, and
            confidence bucketing analysis for the past 7 days.
        """
        cutoff = datetime.utcnow() - timedelta(days=7)
        return self._compute_accuracy(cutoff.isoformat())
    
    def accuracy_30d(self) -> dict:
        """Compute 30-day rolling accuracy and confidence-stratified metrics.
        
        Returns:
            Dictionary with overall accuracy, per-direction accuracy, and
            confidence bucketing analysis for the past 30 days.
        """
        cutoff = datetime.utcnow() - timedelta(days=30)
        return self._compute_accuracy(cutoff.isoformat())
    
    def accuracy_by_ticker(self, days: int = 30) -> dict:
        """Compute per-ticker accuracy over the past N days.
        
        Args:
            days: Number of days to look back (default 30).
        
        Returns:
            Dictionary mapping ticker symbols to accuracy metrics.
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM calibration WHERE timestamp > ?",
            conn,
            params=(cutoff.isoformat(),)
        )
        conn.close()
        
        if df.empty:
            return {}
        
        results = {}
        for ticker in df["ticker"].unique():
            ticker_df = df[df["ticker"] == ticker]
            total = len(ticker_df)
            correct = ticker_df["correct"].sum()
            results[ticker] = {
                "total": int(total),
                "correct": int(correct),
                "accuracy": float(correct / total) if total > 0 else 0.0,
                "avg_confidence": float(ticker_df["predicted_confidence"].mean())
            }
        
        return results
    
    def _compute_accuracy(self, cutoff_iso: str) -> dict:
        """Helper: compute accuracy metrics for records after cutoff timestamp.
        
        Args:
            cutoff_iso: ISO 8601 timestamp threshold.
        
        Returns:
            Nested dict with overall, directional, and confidence-bucketed metrics.
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM calibration WHERE timestamp > ?",
            conn,
            params=(cutoff_iso,)
        )
        conn.close()
        
        if df.empty:
            return {
                "total_predictions": 0,
                "overall_accuracy": 0.0,
                "by_direction": {},
                "by_confidence_bucket": {}
            }
        
        total = len(df)
        correct = df["correct"].sum()
        overall_accuracy = float(correct / total) if total > 0 else 0.0
        
        # Per-direction breakdown
        by_direction = {}
        for direction in ["UP", "DOWN", "HOLD"]:
            dir_df = df[df["predicted_direction"] == direction]
            if len(dir_df) > 0:
                dir_correct = dir_df["correct"].sum()
                by_direction[direction] = {
                    "total": int(len(dir_df)),
                    "correct": int(dir_correct),
                    "accuracy": float(dir_correct / len(dir_df))
                }
        
        # Confidence bucketing (0-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0)
        by_confidence_bucket = {}
        buckets = [(0.0, 0.5), (0.
