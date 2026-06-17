"""
Sentinel Post-Mortem Report Generator — Judge pillar.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movement data via yfinance, compares predicted vs. actual outcomes, and writes
a markdown report to backtest_results/. Supports confidence scoring, heuristic
refinement hints, and anomaly flagging for model calibration.

Integrates with sentinel/judge/postmortem.py spine for rendering; this module
handles data aggregation, actual price fetching, and file I/O.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import yfinance as yf
import pandas as pd


def connect_predictions_db(db_path: str = "sentinel.db") -> sqlite3.Connection:
    """Open connection to predictions SQLite database."""
    return sqlite3.connect(db_path)


def fetch_yesterday_predictions(
    conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """
    Fetch all PredictionRecord entries from yesterday.
    
    Returns list of dicts with keys: ticker, predicted_direction, predicted_confidence,
    created_at, record_id.
    """
    cursor = conn.cursor()
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    cursor.execute(
        """
        SELECT id, ticker, predicted_direction, predicted_confidence, created_at
        FROM predictions
        WHERE DATE(created_at) = ?
        ORDER BY ticker, created_at DESC
        """,
        (yesterday.isoformat(),),
    )
    rows = cursor.fetchall()
    return [
        {
            "record_id": row[0],
            "ticker": row[1],
            "predicted_direction": row[2],
            "predicted_confidence": row[3],
            "created_at": row[4],
        }
        for row in rows
    ]


def fetch_actual_price_movement(
    ticker: str,
    reference_date: datetime,
    days_ahead: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Fetch actual price movement for ticker on/after reference_date.
    
    Returns dict with keys: open_price, close_price, high, low, volume, pct_change,
    actual_direction ('up', 'down', 'flat'), or None on failure.
    """
    try:
        ref_date_str = reference_date.strftime("%Y-%m-%d")
        end_date = (reference_date + timedelta(days=days_ahead + 5)).strftime("%Y-%m-%d")
        
        data = yf.download(
            ticker,
            start=ref_date_str,
            end=end_date,
            progress=False,
            quiet=True,
        )
        
        if data.empty:
            return None
        
        first_row = data.iloc[0]
        last_row = data.iloc[-1]
        
        open_price = float(first_row["Open"])
        close_price = float(last_row["Close"])
        high_price = float(data["High"].max())
        low_price = float(data["Low"].min())
        volume = int(last_row["Volume"])
        
        pct_change = ((close_price - open_price) / open_price) * 100
        actual_direction = "up" if pct_change > 0.5 else ("down" if pct_change < -0.5 else "flat")
        
        return {
            "open_price": open_price,
            "close_price": close_price,
            "high": high_price,
            "low": low_price,
            "volume": volume,
            "pct_change": pct_change,
            "actual_direction": actual_direction,
        }
    except Exception as e:
        print(f"[WARN] fetch_actual_price_movement({ticker}): {e}")
        return None


def evaluate_prediction_accuracy(
    predicted_direction: str,
    predicted_confidence: float,
    actual: Optional[Dict[str, Any]],
) -> Dict[str, Any]]:
    """
    Score prediction vs. actual outcome.
    
    Returns dict with keys: is_correct, accuracy_score, confidence_calibration,
    anomaly_flag.
    """
    if actual is None:
        return {
            "is_correct": None,
            "accuracy_score": 0.0,
            "confidence_calibration": "data_unavailable",
            "anomaly_flag": False,
        }
    
    actual_dir = actual["actual_direction"]
    is_correct = predicted_direction == actual_dir
    
    base_score = 1.0 if is_correct else 0.0
    accuracy_score = base_score * predicted_confidence
    
    if is_correct:
        confidence_calibration = "well_calibrated" if predicted_confidence > 0.7 else "under_confident"
    else:
        confidence_calibration = "over_confident" if predicted_confidence > 0.6 else "reasonably_cautious"
    
    anomaly_flag = abs(actual["pct_change"]) > 10.0
    
    return {
        "is_correct": is_correct,
        "accuracy_score": accuracy_score,
        "confidence_calibration": confidence_calibration,
        "anomaly_flag": anomaly_flag,
    }


def aggregate_daily_metrics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute daily aggregates from list of evaluation dicts.
    
    Returns dict with keys: total_predictions, correct_predictions, accuracy_rate,
    avg_confidence, avg_accuracy_score, anomalies_count.
    """
    if not evaluations:
        return {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy_rate": 0.0,
            "avg_confidence": 0.0,
            "avg_accuracy_score": 0.0,
            "anomalies_count": 0,
        }
    
    total = len(evaluations)
    correct = sum(1 for e in evaluations if e["is_correct"] is True)
    accuracy_rate = correct / total if total > 0 else 0.0
    
    confidences = [e["predicted_confidence"] for e in evaluations if "predicted_confidence" in e]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    accuracy_scores = [e["accuracy_score"] for e in evaluations]
    avg_accuracy_score = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0
    
    anomalies = sum(1 for e in evaluations if e.get("anomaly_flag", False))
    
    return {
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy_rate": accuracy_rate,
        "avg_confidence": avg_confidence,
        "avg_accuracy_score": avg_accuracy_score,
        "anomalies_count": anomalies,
    }


def generate_markdown_report(
    report_date: datetime,
    predictions: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    output_dir: str = "backtest_results",
) -> str:
    """
    Generate markdown post-mortem report and write to disk.
    
    Returns path to written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_date_str = report_date.strftime("%Y-%m-%d")
    filename = f"postmortem_{report_date_str}.md"
    filepath = os.path.join(output_dir, filename)
    
    lines = [
        f"# Sentinel Post-Mortem Report — {report_date_str}",
        "",
        "## Daily Metrics",
        "",
        f"- **Total Predictions**: {metrics['total_predictions']}",
