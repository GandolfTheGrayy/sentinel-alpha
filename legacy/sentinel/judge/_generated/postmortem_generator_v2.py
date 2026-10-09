"""
Post-mortem report generator for the Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movement data via yfinance, compares predicted vs. actual outcomes, and writes
a markdown report to backtest_results/. Used by the daily Judge post-mortem
cycle to calibrate heuristics and flag anomalies.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd


def get_db_path() -> str:
    """Return the path to the Sentinel SQLite database."""
    return os.getenv("SENTINEL_DB_PATH", "sentinel/data/sentinel.db")


def fetch_prediction_records(db_path: str, days_back: int = 1) -> list[dict]:
    """
    Fetch PredictionRecord entries from the last N days.
    
    Args:
        db_path: Path to SQLite database
        days_back: Number of days in the past to query (default 1 = yesterday)
    
    Returns:
        List of prediction records as dicts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    try:
        cursor.execute(
            """
            SELECT id, ticker, predicted_direction, predicted_confidence,
                   reasoning, created_at, tags
            FROM prediction_records
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (cutoff,)
        )
        rows = cursor.fetchall()
        records = [dict(row) for row in rows]
    finally:
        conn.close()
    
    return records


def fetch_actual_price_move(ticker: str, reference_date: str) -> Optional[dict]:
    """
    Fetch actual price movement for a ticker on/after reference_date.
    
    Args:
        ticker: Stock ticker symbol
        reference_date: ISO date string (prediction creation date)
    
    Returns:
        Dict with 'open', 'close', 'direction' keys, or None if unavailable
    """
    try:
        ref_dt = datetime.fromisoformat(reference_date)
        start_date = ref_dt.date()
        end_date = (ref_dt + timedelta(days=2)).date()
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return None
        
        if len(data) < 2:
            return None
        
        open_price = float(data.iloc[0]['Open'])
        close_price = float(data.iloc[-1]['Close'])
        direction = "up" if close_price > open_price else "down"
        
        return {
            "open": open_price,
            "close": close_price,
            "direction": direction,
            "pct_change": ((close_price - open_price) / open_price) * 100
        }
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None


def evaluate_prediction(
    predicted_direction: str,
    predicted_confidence: float,
    actual_move: Optional[dict]
) -> dict:
    """
    Compare predicted vs. actual direction and assign outcome.
    
    Args:
        predicted_direction: "up" or "down"
        predicted_confidence: Float 0-1
        actual_move: Dict with 'direction' key from fetch_actual_price_move
    
    Returns:
        Dict with 'correct', 'confidence_calibration', 'notes' keys
    """
    if not actual_move:
        return {
            "correct": None,
            "confidence_calibration": None,
            "notes": "No price data available"
        }
    
    correct = predicted_direction.lower() == actual_move["direction"].lower()
    
    calibration_notes = []
    if correct and predicted_confidence < 0.6:
        calibration_notes.append("Lucky call with low confidence")
    elif correct and predicted_confidence >= 0.9:
        calibration_notes.append("High-confidence correct prediction")
    elif not correct and predicted_confidence >= 0.8:
        calibration_notes.append("High-confidence miss — review reasoning")
    
    return {
        "correct": correct,
        "confidence_calibration": predicted_confidence,
        "notes": "; ".join(calibration_notes) if calibration_notes else "Neutral"
    }


def generate_markdown_report(
    records: list[dict],
    output_dir: str = "backtest_results"
) -> str:
    """
    Generate a markdown post-mortem report from prediction records.
    
    Args:
        records: List of prediction record dicts
        output_dir: Directory to write the markdown file
    
    Returns:
        Path to the generated markdown file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    report_path = os.path.join(output_dir, f"postmortem_{timestamp}.md")
    
    lines = [
        "# Sentinel Post-Mortem Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "## Summary",
        ""
    ]
    
    if not records:
        lines.append("No prediction records found for the period.")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        return report_path
    
    total = len(records)
    correct_count = 0
    incorrect_count = 0
    neutral_count = 0
    
    lines.append("## Detailed Results\n")
    
    for rec in records:
        ticker = rec["ticker"]
        predicted_direction = rec["predicted_direction"]
        predicted_confidence = rec["predicted_confidence"]
        created_at = rec["created_at"]
        reasoning = rec["reasoning"]
        
        actual_move = fetch_actual_price_move(ticker, created_at)
        evaluation = evaluate_prediction(
            predicted_direction,
            predicted_confidence,
            actual_move
        )
        
        if evaluation["correct"] is True:
            correct_count += 1
            outcome = "✓ CORRECT"
        elif evaluation["correct"] is False:
            incorrect_count += 1
            outcome = "✗ INCORRECT"
        else:
            neutral_count += 1
            outcome = "? NO DATA"
        
        lines.append(f"### {ticker} — {outcome}")
        lines.append(f"- **Prediction**: {predicted_direction.upper()} (confidence: {predicted_confidence:.2%})")
        lines.append(f"- **Created**: {created_at}")
        
        if actual_move:
            lines.append(f"- **Actual**: {actual_move['direction'].upper()} ({actual_move['pct_change']:+.2f}%)")
        else:
            lines.append(f"- **Actual**: No price data")
        
        lines.append(f"- **Calibration**: {evaluation['notes']}")
        lines.append(f"- **Reasoning**: {reasoning[:200]}...")
        lines.append("")
    
    win_rate = (correct_count / total * 100) if total > 0 else 0
    lines.insert(
        lines.index("## Detailed Results\n"),
        f"- **Total Predictions**: {total}\n"
        f"- **Correct**: {correct_count} ({win_rate:.1f}%)\n"
        f"- **Incorrect**: {incorrect_count}\n"
        f"- **No Data**: {neutral_count}\n"
    )
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    return report_path


def
