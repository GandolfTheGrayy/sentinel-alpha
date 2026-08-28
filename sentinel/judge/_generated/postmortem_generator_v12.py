"""
Sentinel Post-Mortem Report Generator.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, computes prediction accuracy metrics, and renders a
markdown report to backtest_results/. Integrates with Judge's daily calibration
loop to identify systematic prediction biases and anomalies.

Part of sentinel/judge/ pillar — produces human-readable retrospectives for
model refinement and performance tracking.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import yfinance as yf
import pandas as pd


def get_yesterday_predictions(db_path: str) -> List[Dict[str, Any]]:
    """Fetch all PredictionRecord rows from yesterday."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    query = """
    SELECT
        id, ticker, predicted_direction, predicted_confidence,
        prediction_date, created_at, reasoning
    FROM predictions
    WHERE DATE(prediction_date) = ?
    ORDER BY ticker, created_at DESC
    """
    cursor.execute(query, (yesterday,))
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return records


def fetch_actual_prices(ticker: str, start_date: str, end_date: str) -> Optional[Dict[str, float]]:
    """Fetch actual OHLCV data from yfinance for a ticker over a date range."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None
        
        open_price = float(data.iloc[0]["Open"])
        close_price = float(data.iloc[-1]["Close"])
        high = float(data["High"].max())
        low = float(data["Low"].min())
        
        return {
            "open": open_price,
            "close": close_price,
            "high": high,
            "low": low,
            "pct_change": ((close_price - open_price) / open_price) * 100 if open_price else 0.0,
        }
    except Exception as e:
        print(f"[WARN] Failed to fetch {ticker}: {e}")
        return None


def compute_accuracy(predicted_direction: str, actual_prices: Dict[str, float]) -> Dict[str, Any]:
    """Compute prediction accuracy against actual price movement."""
    if not actual_prices or actual_prices["pct_change"] == 0:
        return {
            "correct": False,
            "actual_direction": "flat",
            "accuracy_note": "Insufficient data or flat movement",
        }
    
    actual_direction = "up" if actual_prices["pct_change"] > 0 else "down"
    correct = (predicted_direction.lower() == actual_direction)
    
    return {
        "correct": correct,
        "actual_direction": actual_direction,
        "actual_pct_change": actual_prices["pct_change"],
        "accuracy_note": f"{'✓ Correct' if correct else '✗ Incorrect'}: predicted {predicted_direction}, actual {actual_direction} ({actual_prices['pct_change']:.2f}%)",
    }


def generate_postmortem_report(
    predictions: List[Dict[str, Any]],
    accuracies: List[Dict[str, Any]],
    output_dir: str = "backtest_results",
) -> str:
    """Generate a markdown post-mortem report and save it."""
    os.makedirs(output_dir, exist_ok=True)
    
    report_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    report_path = os.path.join(output_dir, f"postmortem_{report_date}.md")
    
    correct_count = sum(1 for a in accuracies if a.get("correct"))
    total_count = len(accuracies)
    accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0.0
    
    lines = [
        f"# Sentinel Post-Mortem — {report_date}",
        "",
        f"**Accuracy:** {correct_count}/{total_count} ({accuracy_pct:.1f}%)",
        "",
        "## Predictions vs. Actuals",
        "",
    ]
    
    for pred, acc in zip(predictions, accuracies):
        lines.extend([
            f"### {pred['ticker']}",
            f"- **Predicted:** {pred['predicted_direction'].upper()} (confidence: {pred['predicted_confidence']:.1%})",
            f"- **Actual:** {acc['actual_direction'].upper()} ({acc.get('actual_pct_change', 0):.2f}%)",
            f"- **Result:** {acc['accuracy_note']}",
            f"- **Reasoning:** {pred.get('reasoning', 'N/A')}",
            "",
        ])
    
    lines.extend([
        "## Summary",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "",
    ])
    
    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"[INFO] Post-mortem report saved to {report_path}")
    return report_path


def run_postmortem(db_path: str = "sentinel.db", output_dir: str = "backtest_results") -> str:
    """
    Execute full post-mortem pipeline: fetch predictions, fetch actuals, compute accuracy, render report.
    """
    print("[INFO] Starting post-mortem generation...")
    
    predictions = get_yesterday_predictions(db_path)
    if not predictions:
        print("[WARN] No predictions found for yesterday.")
        return ""
    
    print(f"[INFO] Found {len(predictions)} prediction(s) from yesterday.")
    
    accuracies = []
    for pred in predictions:
        ticker = pred["ticker"]
        pred_date = pred["prediction_date"]
        
        actual = fetch_actual_prices(
            ticker,
            pred_date,
            (datetime.strptime(pred_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        
        if actual:
            accuracy = compute_accuracy(pred["predicted_direction"], actual)
            accuracies.append(accuracy)
        else:
            accuracies.append({
                "correct": False,
                "actual_direction": "unknown",
                "accuracy_note": f"Failed to fetch price data for {ticker}",
            })
    
    report_path = generate_postmortem_report(predictions, accuracies, output_dir)
    return report_path


if __name__ == "__main__":
    run_postmortem()
