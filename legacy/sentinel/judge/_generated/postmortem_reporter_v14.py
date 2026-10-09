"""
Post-mortem Report Generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
data via yfinance, calculates prediction accuracy metrics, and generates
markdown reports in backtest_results/. Integrated into Judge pillar for
daily calibration and anomaly detection.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import yfinance as yf
import pandas as pd


def get_db_path() -> str:
    """Return the path to the Sentinel SQLite database."""
    return os.environ.get("SENTINEL_DB", "sentinel_predictions.db")


def fetch_prediction_records(db_path: str, days_back: int = 1) -> List[Dict[str, Any]]:
    """Fetch PredictionRecord rows from the past N days."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
    
    query = """
    SELECT id, ticker, predicted_direction, confidence, predicted_price,
           reasoning, created_at, baseline_strategy
    FROM predictions
    WHERE created_at >= ?
    ORDER BY created_at DESC
    """
    cursor.execute(query, (cutoff_date,))
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records


def fetch_actual_price_data(ticker: str, days_back: int = 2) -> Optional[Dict[str, float]]:
    """Fetch opening and closing prices for ticker from the past N days."""
    try:
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)
        
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            timeout=10
        )
        
        if data.empty:
            return None
        
        if len(data) < 2:
            return None
        
        yesterday_row = data.iloc[-2]
        today_row = data.iloc[-1]
        
        return {
            "yesterday_open": float(yesterday_row["Open"]),
            "yesterday_close": float(yesterday_row["Close"]),
            "today_open": float(today_row["Open"]),
            "today_close": float(today_row["Close"]),
        }
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return None


def calculate_accuracy(
    predicted_direction: str,
    actual_move: float
) -> Dict[str, Any]:
    """Calculate whether prediction was correct and by how much."""
    move_direction = "up" if actual_move > 0 else "down" if actual_move < 0 else "flat"
    
    is_correct = (
        (predicted_direction.lower() == "up" and actual_move > 0.001) or
        (predicted_direction.lower() == "down" and actual_move < -0.001) or
        (predicted_direction.lower() == "flat" and abs(actual_move) <= 0.001)
    )
    
    return {
        "is_correct": is_correct,
        "actual_direction": move_direction,
        "actual_percent_change": round(actual_move * 100, 2),
    }


def generate_postmortem_report(
    db_path: str,
    output_dir: str = "backtest_results/",
    days_back: int = 1
) -> Optional[str]:
    """
    Generate a markdown post-mortem report for predictions from the past N days.
    
    Returns the path to the generated report file, or None if no predictions found.
    """
    records = fetch_prediction_records(db_path, days_back=days_back)
    
    if not records:
        print("No prediction records found.")
        return None
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_lines = [
        "# Sentinel Post-Mortem Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        f"Period: Past {days_back} day(s)",
        "",
        "## Summary",
        "",
    ]
    
    correct_count = 0
    total_count = 0
    accuracy_by_strategy = {}
    accuracy_by_ticker = {}
    
    results = []
    
    for record in records:
        ticker = record["ticker"]
        predicted_direction = record["predicted_direction"]
        confidence = record["confidence"]
        predicted_price = record["predicted_price"]
        reasoning = record["reasoning"]
        baseline_strategy = record["baseline_strategy"] or "unknown"
        
        price_data = fetch_actual_price_data(ticker, days_back=2)
        
        if price_data is None:
            results.append({
                "ticker": ticker,
                "status": "no_data",
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "predicted_price": predicted_price,
            })
            continue
        
        yesterday_close = price_data["yesterday_close"]
        today_close = price_data["today_close"]
        actual_move = (today_close - yesterday_close) / yesterday_close if yesterday_close else 0
        
        accuracy = calculate_accuracy(predicted_direction, actual_move)
        
        results.append({
            "ticker": ticker,
            "status": "evaluated",
            "predicted_direction": predicted_direction,
            "confidence": confidence,
            "predicted_price": predicted_price,
            "actual_yesterday_close": yesterday_close,
            "actual_today_close": today_close,
            **accuracy,
            "baseline_strategy": baseline_strategy,
            "reasoning": reasoning,
        })
        
        if accuracy["is_correct"]:
            correct_count += 1
        total_count += 1
        
        if baseline_strategy not in accuracy_by_strategy:
            accuracy_by_strategy[baseline_strategy] = {"correct": 0, "total": 0}
        accuracy_by_strategy[baseline_strategy]["correct"] += int(accuracy["is_correct"])
        accuracy_by_strategy[baseline_strategy]["total"] += 1
        
        if ticker not in accuracy_by_ticker:
            accuracy_by_ticker[ticker] = {"correct": 0, "total": 0}
        accuracy_by_ticker[ticker]["correct"] += int(accuracy["is_correct"])
        accuracy_by_ticker[ticker]["total"] += 1
    
    if total_count > 0:
        overall_accuracy = (correct_count / total_count) * 100
    else:
        overall_accuracy = 0
    
    report_lines.append(f"**Overall Accuracy:** {correct_count}/{total_count} ({overall_accuracy:.1f}%)")
    report_lines.append("")
    
    if accuracy_by_strategy:
        report_lines.append("### By Strategy")
        report_lines.append("")
        for strategy, stats in sorted(accuracy_by_strategy.items()):
            pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report_lines.append(f"- **{strategy}**: {stats['correct']}/{stats['total']} ({pct:.1f}%)")
        report_lines.append("")
    
    if accuracy_by_ticker:
        report_lines.append("### By Ticker")
        report_lines.append("")
        for ticker, stats in sorted(accuracy_by_ticker.items()):
            pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report_lines.append(f"- **{ticker}**: {stats['correct']}/{stats['total']} ({pct:.1f}%)")
        report_lines.append("")
    
    report_lines.
