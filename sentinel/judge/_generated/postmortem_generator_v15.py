"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, compares predicted vs. actual outcomes, and writes
markdown reports to backtest_results/. Supports confidence calibration and
anomaly flagging for Judge's daily refinement loop.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd


def get_prediction_records(db_path: str, days_ago: int = 1) -> list[dict]:
    """Fetch all PredictionRecord entries from N days ago."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    target_date = (datetime.now() - timedelta(days=days_ago)).date()
    
    cursor.execute(
        """
        SELECT ticker, predicted_direction, confidence, 
               predicted_price, created_at, reasoning
        FROM predictions
        WHERE DATE(created_at) = ?
        ORDER BY ticker
        """,
        (str(target_date),)
    )
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records


def fetch_actual_price(ticker: str, target_date: str) -> Optional[dict]:
    """Fetch actual close price for ticker on target_date via yfinance."""
    try:
        # Fetch data for target date and one day prior for comparison
        data = yf.download(
            ticker,
            start=(datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"),
            end=(datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            quiet=True
        )
        
        if data.empty:
            return None
        
        # Find the target date row
        target_datetime = pd.to_datetime(target_date)
        if target_datetime in data.index:
            row = data.loc[target_datetime]
            return {
                "close": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]) if "Volume" in row else 0,
            }
        
        return None
    except Exception as e:
        print(f"Error fetching {ticker} for {target_date}: {e}")
        return None


def compute_accuracy(
    prediction: dict,
    actual: Optional[dict]
) -> dict:
    """Compare predicted direction/price to actual outcome."""
    if actual is None:
        return {
            "outcome": "no_data",
            "direction_correct": None,
            "price_error_pct": None,
            "confidence_calibrated": False,
        }
    
    actual_close = actual["close"]
    predicted_direction = prediction.get("predicted_direction", "neutral").lower()
    predicted_price = prediction.get("predicted_price")
    
    # Infer actual direction from close price change
    # (assumes we have prior day data; simplified here)
    actual_direction = "up" if actual_close > actual.get("open", actual_close) else "down"
    
    direction_correct = (predicted_direction == actual_direction)
    
    price_error_pct = None
    if predicted_price and predicted_price > 0:
        price_error_pct = abs(actual_close - predicted_price) / predicted_price * 100
    
    return {
        "outcome": "correct" if direction_correct else "incorrect",
        "direction_correct": direction_correct,
        "actual_close": actual_close,
        "price_error_pct": price_error_pct,
        "confidence_calibrated": prediction.get("confidence", 0) >= 0.7,
    }


def generate_postmortem_report(
    db_path: str,
    output_dir: str = "backtest_results",
    days_ago: int = 1,
) -> str:
    """Generate markdown post-mortem report; return report path."""
    records = get_prediction_records(db_path, days_ago=days_ago)
    
    if not records:
        print(f"No prediction records found from {days_ago} day(s) ago.")
        return ""
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    target_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    report_filename = f"postmortem_{target_date}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    # Build report
    lines = [
        f"# Sentinel Post-Mortem: {target_date}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]
    
    correct_count = 0
    total_count = 0
    high_confidence_correct = 0
    high_confidence_total = 0
    
    ticker_results = []
    
    for pred in records:
        ticker = pred["ticker"]
        actual = fetch_actual_price(ticker, target_date)
        accuracy = compute_accuracy(pred, actual)
        
        total_count += 1
        if accuracy["outcome"] == "correct":
            correct_count += 1
        
        conf = pred.get("confidence", 0)
        if conf >= 0.7:
            high_confidence_total += 1
            if accuracy["outcome"] == "correct":
                high_confidence_correct += 1
        
        ticker_results.append({
            "ticker": ticker,
            "prediction": pred.get("predicted_direction"),
            "confidence": conf,
            "actual_close": accuracy.get("actual_close"),
            "price_error_pct": accuracy.get("price_error_pct"),
            "outcome": accuracy["outcome"],
            "reasoning": pred.get("reasoning", "N/A"),
        })
    
    accuracy_pct = (correct_count / total_count * 100) if total_count > 0 else 0
    high_conf_accuracy = (
        (high_confidence_correct / high_confidence_total * 100)
        if high_confidence_total > 0
        else 0
    )
    
    lines.extend([
        f"- **Directional Accuracy:** {correct_count}/{total_count} ({accuracy_pct:.1f}%)",
        f"- **High-Confidence (≥0.7) Accuracy:** {high_confidence_correct}/{high_confidence_total} ({high_conf_accuracy:.1f}%)",
        "",
        "## Detailed Results",
        "",
    ])
    
    # Table header
    lines.append("| Ticker | Prediction | Confidence | Actual Close | Error % | Outcome | Reasoning |")
    lines.append("|--------|------------|------------|--------------|---------|---------|-----------|")
    
    for res in ticker_results:
        error_str = (
            f"{res['price_error_pct']:.2f}" if res['price_error_pct'] is not None else "—"
        )
        actual_str = f"{res['actual_close']:.2f}" if res['actual_close'] else "N/A"
        reasoning_short = (res["reasoning"][:60] + "...") if len(res["reasoning"]) > 60 else res["reasoning"]
        
        lines.append(
            f"| {res['ticker']} | {res['prediction']} | {res['confidence']:.2f} | "
            f"{actual_str} | {error_str}% | {res['outcome']} | {reasoning_short} |"
        )
    
    lines.extend([
        "",
        "## Anomalies & Flags",
