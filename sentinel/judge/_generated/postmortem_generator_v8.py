"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price data
via yfinance, compares predicted vs. actual movements, and writes a markdown report
to backtest_results/. Used by Judge to calibrate heuristics and flag anomalies.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd


def get_yesterday_predictions(db_path: str) -> List[Dict]:
    """Fetch all PredictionRecord entries from yesterday."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).date()
    
    cursor.execute(
        """
        SELECT ticker, predicted_direction, predicted_confidence, predicted_at
        FROM prediction_record
        WHERE DATE(predicted_at) = ?
        ORDER BY predicted_at DESC
        """,
        (yesterday.isoformat(),)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def fetch_actual_price_move(ticker: str, prediction_date: str) -> Optional[Tuple[float, float, str]]:
    """
    Fetch actual price movement for ticker from prediction_date to next trading day.
    
    Returns (open_price, close_price, direction) or None if data unavailable.
    """
    try:
        # Parse prediction date
        pred_dt = datetime.fromisoformat(prediction_date)
        pred_date = pred_dt.date()
        
        # Fetch next 3 trading days to ensure we get at least one close
        end_date = pred_date + timedelta(days=5)
        
        ticker_data = yf.download(
            ticker,
            start=pred_date.isoformat(),
            end=end_date.isoformat(),
            progress=False,
            quiet=True
        )
        
        if ticker_data.empty or len(ticker_data) < 2:
            return None
        
        # Get open from prediction date or next available
        open_price = float(ticker_data.iloc[0]["Open"])
        
        # Get close from next trading day
        close_price = float(ticker_data.iloc[1]["Close"])
        
        # Determine direction
        move_pct = ((close_price - open_price) / open_price) * 100
        direction = "UP" if move_pct > 0 else "DOWN" if move_pct < 0 else "FLAT"
        
        return (open_price, close_price, direction)
    
    except Exception as e:
        print(f"Error fetching price for {ticker}: {e}")
        return None


def calculate_accuracy(predictions: List[Dict]) -> Dict:
    """
    Calculate hit rate, false positives/negatives, and confidence calibration.
    
    Returns dict with accuracy metrics.
    """
    if not predictions:
        return {
            "total": 0,
            "hits": 0,
            "misses": 0,
            "accuracy_pct": 0.0,
            "avg_confidence_correct": 0.0,
            "avg_confidence_wrong": 0.0,
            "up_predictions": 0,
            "down_predictions": 0,
            "up_correct": 0,
            "down_correct": 0,
        }
    
    hits = 0
    misses = 0
    confidence_correct = []
    confidence_wrong = []
    up_pred = 0
    down_pred = 0
    up_correct = 0
    down_correct = 0
    
    for pred in predictions:
        ticker = pred["ticker"]
        predicted_dir = pred["predicted_direction"]
        confidence = float(pred["predicted_confidence"])
        predicted_at = pred["predicted_at"]
        
        result = fetch_actual_price_move(ticker, predicted_at)
        
        if result is None:
            misses += 1
            confidence_wrong.append(confidence)
            continue
        
        _, _, actual_dir = result
        
        if predicted_dir == "UP":
            up_pred += 1
        else:
            down_pred += 1
        
        if predicted_dir == actual_dir:
            hits += 1
            confidence_correct.append(confidence)
            if predicted_dir == "UP":
                up_correct += 1
            else:
                down_correct += 1
        else:
            misses += 1
            confidence_wrong.append(confidence)
    
    total = hits + misses
    accuracy_pct = (hits / total * 100) if total > 0 else 0.0
    
    avg_conf_correct = sum(confidence_correct) / len(confidence_correct) if confidence_correct else 0.0
    avg_conf_wrong = sum(confidence_wrong) / len(confidence_wrong) if confidence_wrong else 0.0
    
    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "accuracy_pct": round(accuracy_pct, 2),
        "avg_confidence_correct": round(avg_conf_correct, 3),
        "avg_confidence_wrong": round(avg_conf_wrong, 3),
        "up_predictions": up_pred,
        "down_predictions": down_pred,
        "up_correct": up_correct,
        "down_correct": down_correct,
    }


def generate_postmortem_markdown(
    predictions: List[Dict],
    metrics: Dict,
    report_date: str
) -> str:
    """
    Generate markdown report comparing predictions vs. actuals.
    
    Returns markdown string.
    """
    lines = [
        f"# Sentinel Post-Mortem Report",
        f"**Report Date:** {report_date}",
        f"**Predictions Analyzed:** {metrics['total']}",
        "",
        "## Summary",
        f"- **Accuracy:** {metrics['accuracy_pct']}% ({metrics['hits']}/{metrics['total']})",
        f"- **UP Predictions:** {metrics['up_predictions']} (correct: {metrics['up_correct']})",
        f"- **DOWN Predictions:** {metrics['down_predictions']} (correct: {metrics['down_correct']})",
        f"- **Avg Confidence (Correct):** {metrics['avg_confidence_correct']}",
        f"- **Avg Confidence (Wrong):** {metrics['avg_confidence_wrong']}",
        "",
        "## Predictions vs. Actuals",
        "",
        "| Ticker | Predicted | Actual | Confidence | Result |",
        "|--------|-----------|--------|------------|--------|",
    ]
    
    for pred in predictions:
        ticker = pred["ticker"]
        predicted_dir = pred["predicted_direction"]
        confidence = float(pred["predicted_confidence"])
        predicted_at = pred["predicted_at"]
        
        result = fetch_actual_price_move(ticker, predicted_at)
        
        if result is None:
            actual_dir = "N/A"
            match = "❌ Data Missing"
        else:
            _, _, actual_dir = result
            match = "✅ Hit" if predicted_dir == actual_dir else "❌ Miss"
        
        lines.append(
            f"| {ticker} | {predicted_dir} | {actual_dir} | {confidence:.3f} | {match} |"
        )
    
    lines.extend([
        "",
        "## Calibration Notes",
        f"- Confidence scores on correct predictions average {metrics['avg_confidence_correct']}",
        f"- Confidence scores on wrong predictions average {metrics['avg_confidence_wrong']}",
        "- If correct predictions are consistently lower confidence than wrong ones, recalibrate linguist thresholds.",
        "",
        f"*Generated: {datetime.utcnow().isoformat()}*"
