"""Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord from SQLite, fetches actual price data via yfinance,
computes prediction accuracy (directional, magnitude), and renders a markdown report
to backtest_results/. Integrates with Judge pillar for daily performance calibration.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import yfinance as yf
import pandas as pd


def get_yesterday_predictions(db_path: str) -> list[dict]:
    """Fetch all PredictionRecord rows from yesterday (created_at date matches yesterday)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    yesterday_str = yesterday.isoformat()
    
    cursor.execute(
        """
        SELECT id, ticker, predicted_direction, predicted_confidence, 
               predicted_price_target, created_at
        FROM prediction_records
        WHERE DATE(created_at) = ?
        ORDER BY ticker ASC
        """,
        (yesterday_str,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def fetch_actual_prices(ticker: str, trade_date: str) -> Optional[dict]:
    """Fetch actual OHLCV data for ticker on trade_date (next trading day after prediction)."""
    try:
        data = yf.download(ticker, start=trade_date, end=(
            datetime.strptime(trade_date, "%Y-%m-%d") + timedelta(days=5)
        ).strftime("%Y-%m-%d"), progress=False)
        
        if data.empty:
            return None
        
        first_row = data.iloc[0]
        return {
            "open": float(first_row["Open"]),
            "close": float(first_row["Close"]),
            "high": float(first_row["High"]),
            "low": float(first_row["Low"]),
            "volume": int(first_row["Volume"]),
            "date": data.index[0].strftime("%Y-%m-%d")
        }
    except Exception as e:
        print(f"Error fetching prices for {ticker}: {e}")
        return None


def compute_accuracy(prediction: dict, actual: dict) -> dict:
    """Compute directional correctness and magnitude error vs. predicted target."""
    if not actual:
        return {"status": "no_data", "directional": None, "magnitude_error_pct": None}
    
    actual_move = actual["close"] - actual["open"]
    predicted_direction = prediction["predicted_direction"]  # "up" or "down"
    
    directional_correct = (
        (predicted_direction == "up" and actual_move > 0) or
        (predicted_direction == "down" and actual_move < 0)
    )
    
    target = prediction.get("predicted_price_target")
    magnitude_error = None
    if target:
        magnitude_error = abs(actual["close"] - target) / target * 100
    
    return {
        "status": "ok",
        "directional": directional_correct,
        "actual_move_pct": (actual_move / actual["open"]) * 100,
        "magnitude_error_pct": magnitude_error
    }


def generate_markdown_report(
    predictions: list[dict],
    output_dir: str = "backtest_results"
) -> str:
    """Generate markdown post-mortem report and write to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_date = (datetime.utcnow() - timedelta(days=1)).date()
    report_path = os.path.join(output_dir, f"postmortem_{report_date}.md")
    
    lines = [
        f"# Sentinel Post-Mortem: {report_date}",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        "",
        "## Prediction Accuracy Summary",
        ""
    ]
    
    if not predictions:
        lines.extend([
            "No predictions found for yesterday.",
            ""
        ])
    else:
        correct_count = 0
        valid_count = 0
        magnitude_errors = []
        
        lines.append("| Ticker | Predicted | Confidence | Target | Actual Close | Directional | Magnitude Error |")
        lines.append("|--------|-----------|------------|--------|--------------|-------------|-----------------|")
        
        for pred in predictions:
            ticker = pred["ticker"]
            pred_date = pred["created_at"][:10]  # Extract date string
            next_trading_day = (datetime.strptime(pred_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            
            actual = fetch_actual_prices(ticker, next_trading_day)
            accuracy = compute_accuracy(pred, actual)
            
            actual_close = actual["close"] if actual else "N/A"
            directional_str = "✓" if accuracy["directional"] else "✗" if accuracy["directional"] is not None else "—"
            magnitude_str = f"{accuracy['magnitude_error_pct']:.2f}%" if accuracy["magnitude_error_pct"] else "—"
            
            lines.append(
                f"| {ticker} | {pred['predicted_direction'].upper()} | {pred['predicted_confidence']:.2%} | "
                f"{pred.get('predicted_price_target', 'N/A')} | {actual_close} | {directional_str} | {magnitude_str} |"
            )
            
            if accuracy["status"] == "ok":
                valid_count += 1
                if accuracy["directional"]:
                    correct_count += 1
                if accuracy["magnitude_error_pct"] is not None:
                    magnitude_errors.append(accuracy["magnitude_error_pct"])
        
        lines.extend([
            "",
            f"**Directional Accuracy:** {correct_count}/{valid_count} ({correct_count/valid_count*100:.1f}%)" if valid_count > 0 else "**Directional Accuracy:** N/A",
            ""
        ])
        
        if magnitude_errors:
            avg_error = sum(magnitude_errors) / len(magnitude_errors)
            lines.append(f"**Avg Magnitude Error:** {avg_error:.2f}%")
            lines.append("")
    
    lines.extend([
        "---",
        "*Sentinel Judge Module — Daily Post-Mortem*",
        ""
    ])
    
    report_content = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_content)
    
    return report_path


def run_postmortem(db_path: str = "sentinel.db", output_dir: str = "backtest_results") -> str:
    """Main entry point: fetch yesterday's predictions, compute accuracy, generate report."""
    predictions = get_yesterday_predictions(db_path)
    report_path = generate_markdown_report(predictions, output_dir)
    print(f"Post-mortem report written to: {report_path}")
    return report_path


if __name__ == "__main__":
    run_postmortem()
