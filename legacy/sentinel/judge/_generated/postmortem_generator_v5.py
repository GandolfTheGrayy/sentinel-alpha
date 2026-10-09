"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price data
via yfinance, calculates prediction accuracy metrics, and renders a markdown report
to backtest_results/. Integrates with Judge pillar to enable daily calibration loops.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import pandas as pd


def get_yesterday_predictions(db_path: str) -> List[Dict]:
    """Retrieve all PredictionRecord entries from yesterday."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    cursor.execute("""
        SELECT ticker, predicted_direction, predicted_confidence, 
               prediction_timestamp, predicted_move_pct
        FROM predictions
        WHERE DATE(prediction_timestamp) = ?
        ORDER BY ticker, prediction_timestamp DESC
    """, (yesterday,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def fetch_actual_price_data(ticker: str, date_str: str) -> Optional[Tuple[float, float, float]]:
    """Fetch open, close, and volume for ticker on given date. Returns (open, close, volume) or None."""
    try:
        data = yf.download(ticker, start=date_str, end=date_str, progress=False, quiet=True)
        if data.empty:
            return None
        row = data.iloc[0]
        return (float(row['Open']), float(row['Close']), float(row['Volume']))
    except Exception as e:
        print(f"Warning: Could not fetch {ticker} on {date_str}: {e}")
        return None


def calculate_direction_accuracy(predicted_direction: str, open_price: float, close_price: float) -> bool:
    """Check if actual direction matches prediction (UP/DOWN/FLAT)."""
    actual_move = close_price - open_price
    if abs(actual_move) < 0.01 * open_price:
        actual_direction = "FLAT"
    elif actual_move > 0:
        actual_direction = "UP"
    else:
        actual_direction = "DOWN"
    
    return predicted_direction == actual_direction


def calculate_move_accuracy(predicted_move_pct: float, actual_move_pct: float, tolerance_pct: float = 1.0) -> bool:
    """Check if actual move is within tolerance_pct of predicted move."""
    return abs(actual_move_pct - predicted_move_pct) <= tolerance_pct


def generate_postmortem_report(predictions: List[Dict], results_dir: str = "backtest_results") -> str:
    """Generate markdown post-mortem report from yesterday's predictions and actual prices."""
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    report_path = Path(results_dir) / f"postmortem_{yesterday}.md"
    
    report_lines = [
        f"# Sentinel Post-Mortem Report",
        f"**Date:** {yesterday}",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        ""
    ]
    
    if not predictions:
        report_lines.append("No predictions found for yesterday.")
        report_content = "\n".join(report_lines)
        with open(report_path, "w") as f:
            f.write(report_content)
        return str(report_path)
    
    direction_hits = 0
    move_hits = 0
    total_predictions = 0
    ticker_results = []
    
    for pred in predictions:
        ticker = pred['ticker']
        predicted_direction = pred['predicted_direction']
        predicted_confidence = pred['predicted_confidence']
        predicted_move_pct = pred['predicted_move_pct']
        
        price_data = fetch_actual_price_data(ticker, yesterday)
        if price_data is None:
            continue
        
        open_price, close_price, volume = price_data
        actual_move_pct = ((close_price - open_price) / open_price) * 100.0
        
        direction_correct = calculate_direction_accuracy(predicted_direction, open_price, close_price)
        move_correct = calculate_move_accuracy(predicted_move_pct, actual_move_pct, tolerance_pct=1.0)
        
        direction_hits += int(direction_correct)
        move_hits += int(move_correct)
        total_predictions += 1
        
        ticker_results.append({
            'ticker': ticker,
            'predicted_direction': predicted_direction,
            'predicted_confidence': predicted_confidence,
            'predicted_move_pct': predicted_move_pct,
            'actual_move_pct': actual_move_pct,
            'direction_correct': direction_correct,
            'move_correct': move_correct,
            'open': open_price,
            'close': close_price,
            'volume': volume
        })
    
    if total_predictions > 0:
        direction_accuracy = (direction_hits / total_predictions) * 100.0
        move_accuracy = (move_hits / total_predictions) * 100.0
    else:
        direction_accuracy = 0.0
        move_accuracy = 0.0
    
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|--------|-------|")
    report_lines.append(f"| Total Predictions | {total_predictions} |")
    report_lines.append(f"| Direction Accuracy | {direction_accuracy:.1f}% ({direction_hits}/{total_predictions}) |")
    report_lines.append(f"| Move Accuracy (±1%) | {move_accuracy:.1f}% ({move_hits}/{total_predictions}) |")
    report_lines.append("")
    
    report_lines.append("## Detailed Results")
    report_lines.append("")
    report_lines.append("| Ticker | Pred Direction | Confidence | Pred Move | Actual Move | Dir ✓ | Move ✓ |")
    report_lines.append("|--------|----------------|------------|-----------|-------------|-------|--------|")
    
    for result in ticker_results:
        dir_check = "✓" if result['direction_correct'] else "✗"
        move_check = "✓" if result['move_correct'] else "✗"
        report_lines.append(
            f"| {result['ticker']} | {result['predicted_direction']} | {result['predicted_confidence']:.2f} | "
            f"{result['predicted_move_pct']:+.2f}% | {result['actual_move_pct']:+.2f}% | {dir_check} | {move_check} |"
        )
    
    report_lines.append("")
    report_lines.append("## Confidence Distribution")
    report_lines.append("")
    
    if ticker_results:
        high_conf = [r for r in ticker_results if r['predicted_confidence'] >= 0.7]
        mid_conf = [r for r in ticker_results if 0.4 <= r['predicted_confidence'] < 0.7]
        low_conf = [r for r in ticker_results if r['predicted_confidence'] < 0.4]
        
        report_lines.append(f"- **High Confidence (≥0.7):** {len(high_conf)} predictions")
        if high_conf:
            high_acc = sum(1 for r in high_conf if r['direction_correct']) / len(high_conf) * 100.0
            report_lines.append(f"  - Direction accuracy: {high_acc:.1f}%")
