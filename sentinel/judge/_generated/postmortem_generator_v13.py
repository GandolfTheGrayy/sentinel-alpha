"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, calculates prediction accuracy metrics, and renders
a markdown report to backtest_results/. Serves the Judge pillar's daily
calibration workflow.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yfinance as yf
import pandas as pd


def get_yesterday_predictions(db_path: str) -> list[dict]:
    """Fetch all PredictionRecord entries from yesterday."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    cursor.execute(
        """
        SELECT ticker, predicted_direction, confidence, predicted_at, target_price
        FROM predictions
        WHERE DATE(predicted_at) = ?
        ORDER BY ticker
        """,
        (yesterday,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_actual_prices(ticker: str, date_str: str) -> Optional[dict]:
    """Fetch actual price data for ticker on given date (YYYY-MM-DD)."""
    try:
        data = yf.download(ticker, start=date_str, end=date_str, progress=False)
        if data.empty:
            return None
        row = data.iloc[0]
        return {
            "open": float(row["Open"]),
            "close": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
        }
    except Exception as e:
        print(f"Error fetching {ticker} on {date_str}: {e}")
        return None


def calculate_accuracy(
    predicted_direction: str, open_price: float, close_price: float
) -> bool:
    """Check if predicted direction matches actual movement."""
    actual_direction = "UP" if close_price > open_price else "DOWN"
    return predicted_direction.upper() == actual_direction


def generate_postmortem_report(
    db_path: str, output_dir: str = "backtest_results"
) -> str:
    """Generate markdown post-mortem report and write to file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    predictions = get_yesterday_predictions(db_path)
    if not predictions:
        report = "# Post-Mortem Report\n\nNo predictions found for yesterday.\n"
        return report
    
    yesterday_str = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    correct_count = 0
    total_count = 0
    results = []
    
    for pred in predictions:
        ticker = pred["ticker"]
        predicted_dir = pred["predicted_direction"]
        confidence = pred["confidence"]
        target_price = pred.get("target_price")
        
        actual = fetch_actual_prices(ticker, yesterday_str)
        if not actual:
            results.append(
                {
                    "ticker": ticker,
                    "predicted": predicted_dir,
                    "confidence": confidence,
                    "actual": "N/A",
                    "correct": None,
                    "target": target_price,
                }
            )
            continue
        
        is_correct = calculate_accuracy(
            predicted_dir, actual["open"], actual["close"]
        )
        if is_correct:
            correct_count += 1
        total_count += 1
        
        actual_direction = "UP" if actual["close"] > actual["open"] else "DOWN"
        move_pct = (
            100 * (actual["close"] - actual["open"]) / actual["open"]
        )
        
        results.append(
            {
                "ticker": ticker,
                "predicted": predicted_dir,
                "confidence": confidence,
                "actual": actual_direction,
                "correct": is_correct,
                "move_pct": move_pct,
                "open": actual["open"],
                "close": actual["close"],
                "target": target_price,
            }
        )
    
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    
    report = f"""# Sentinel Post-Mortem Report
**Date:** {yesterday_str}

## Summary
- **Total Predictions:** {total_count}
- **Correct:** {correct_count}
- **Accuracy:** {accuracy:.1f}%

## Detailed Results

| Ticker | Predicted | Confidence | Actual | Move % | Target | ✓ |
|--------|-----------|------------|--------|--------|--------|---|
"""
    
    for result in results:
        ticker = result["ticker"]
        predicted = result["predicted"]
        conf = f"{result['confidence']:.2f}" if result["confidence"] else "N/A"
        actual = result["actual"]
        move = f"{result.get('move_pct', 0):.2f}%" if "move_pct" in result else "N/A"
        target = f"{result['target']:.2f}" if result["target"] else "N/A"
        correct_mark = "✓" if result["correct"] is True else "✗" if result["correct"] is False else "?"
        
        report += f"| {ticker} | {predicted} | {conf} | {actual} | {move} | {target} | {correct_mark} |\n"
    
    report += "\n---\n*Generated by Sentinel Judge — Daily Calibration Workflow*\n"
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"postmortem_{yesterday_str}_{timestamp}.md"
    
    with open(output_file, "w") as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    db_path = os.getenv("SENTINEL_DB", "sentinel/data/sentinel.db")
    report = generate_postmortem_report(db_path)
    print(report)
