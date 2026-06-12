"""
Sentinel post-mortem report generator.

Reads yesterday's PredictionRecord from SQLite, fetches actual price data via yfinance,
compares predicted vs. actual movements, and writes a markdown report to backtest_results/.
Integrates with the Judge pillar to calibrate heuristics and flag anomalies.
"""

import sqlite3
import json
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
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    cursor.execute(
        """
        SELECT ticker, predicted_direction, predicted_magnitude, confidence_score,
               reasoning, created_at
        FROM prediction_record
        WHERE DATE(created_at) = ?
        """,
        (yesterday,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def fetch_actual_price_movement(ticker: str, date_str: str) -> Optional[dict]:
    """Fetch actual price data for a ticker on a given date."""
    try:
        data = yf.download(ticker, start=date_str, end=date_str, progress=False)
        if data.empty:
            return None
        
        close = data["Close"].iloc[0]
        open_price = data["Open"].iloc[0]
        high = data["High"].iloc[0]
        low = data["Low"].iloc[0]
        
        movement_pct = ((close - open_price) / open_price) * 100 if open_price > 0 else 0.0
        direction = "up" if movement_pct > 0 else "down" if movement_pct < 0 else "flat"
        
        return {
            "open": open_price,
            "close": close,
            "high": high,
            "low": low,
            "movement_pct": movement_pct,
            "direction": direction,
        }
    except Exception as e:
        return None


def evaluate_prediction(prediction: dict, actual: dict) -> dict:
    """Compare predicted vs. actual movement and score accuracy."""
    predicted_direction = prediction.get("predicted_direction", "unknown")
    predicted_magnitude = prediction.get("predicted_magnitude", 0.0)
    confidence = prediction.get("confidence_score", 0.0)
    
    actual_direction = actual.get("direction", "unknown")
    actual_magnitude = abs(actual.get("movement_pct", 0.0))
    
    direction_correct = predicted_direction == actual_direction
    magnitude_error = abs(predicted_magnitude - actual_magnitude)
    
    return {
        "direction_correct": direction_correct,
        "magnitude_error": magnitude_error,
        "confidence_used": confidence,
        "actual_magnitude": actual_magnitude,
    }


def generate_postmortem_report(
    db_path: str,
    output_dir: str = "backtest_results",
) -> str:
    """
    Generate a markdown post-mortem report for yesterday's predictions.
    
    Returns the path to the written report file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    predictions = get_yesterday_predictions(db_path)
    
    if not predictions:
        report_content = "# Post-Mortem Report\n\nNo predictions found for yesterday.\n"
    else:
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        report_lines = [
            "# Post-Mortem Report",
            f"\n**Date:** {yesterday}\n",
            f"**Predictions analyzed:** {len(predictions)}\n",
        ]
        
        results = []
        direction_accuracy = 0
        total_magnitude_error = 0.0
        
        for pred in predictions:
            ticker = pred.get("ticker", "UNKNOWN")
            actual = fetch_actual_price_movement(ticker, yesterday)
            
            if actual is None:
                report_lines.append(f"\n### {ticker}")
                report_lines.append("*No trading data found (holiday or delisted)*")
                continue
            
            evaluation = evaluate_prediction(pred, actual)
            results.append({
                "ticker": ticker,
                "evaluation": evaluation,
                "prediction": pred,
                "actual": actual,
            })
            
            if evaluation["direction_correct"]:
                direction_accuracy += 1
            total_magnitude_error += evaluation["magnitude_error"]
        
        if results:
            accuracy_pct = (direction_accuracy / len(results)) * 100
            avg_magnitude_error = total_magnitude_error / len(results)
            
            report_lines.append(f"\n## Summary Metrics\n")
            report_lines.append(f"- **Direction Accuracy:** {accuracy_pct:.1f}% ({direction_accuracy}/{len(results)})")
            report_lines.append(f"- **Avg Magnitude Error:** {avg_magnitude_error:.2f}%\n")
        
        report_lines.append("\n## Per-Ticker Results\n")
        
        for result in results:
            ticker = result["ticker"]
            eval_data = result["evaluation"]
            actual = result["actual"]
            pred = result["prediction"]
            
            direction_emoji = "✓" if eval_data["direction_correct"] else "✗"
            
            report_lines.append(f"\n### {ticker} {direction_emoji}")
            report_lines.append(f"- **Predicted:** {pred.get('predicted_direction', '?')} ({pred.get('predicted_magnitude', 0):.2f}%)")
            report_lines.append(f"- **Actual:** {actual['direction']} ({actual['movement_pct']:.2f}%)")
            report_lines.append(f"- **Magnitude Error:** {eval_data['magnitude_error']:.2f}%")
            report_lines.append(f"- **Confidence Used:** {eval_data['confidence_used']:.2f}")
            report_lines.append(f"- **Reasoning:** {pred.get('reasoning', 'N/A')}")
        
        report_content = "\n".join(report_lines)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    report_path = Path(output_dir) / f"postmortem_{timestamp}.md"
    
    with open(report_path, "w") as f:
        f.write(report_content)
    
    return str(report_path)


if __name__ == "__main__":
    db_path = "sentinel/data/sentinel.db"
    report_file = generate_postmortem_report(db_path)
    print(f"Post-mortem report written to: {report_file}")
