"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, computes error metrics, and renders a markdown report
to backtest_results/. Feeds into Judge's daily calibration loop.

Depends on: sentinel.judge.predictor (PredictionRecord schema),
sentinel.scout.live_prices (price fetching), yfinance, sqlite3, datetime.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import yfinance as yf
import json


class PredictionRecord:
    """Mirrors the schema from sentinel.judge.predictor for type safety."""
    def __init__(self, ticker: str, predicted_direction: str, confidence: float,
                 predicted_price: float, reasoning: str, prediction_date: str,
                 actual_price: Optional[float] = None, actual_direction: Optional[str] = None,
                 error_pct: Optional[float] = None, resolved_at: Optional[str] = None):
        self.ticker = ticker
        self.predicted_direction = predicted_direction
        self.confidence = confidence
        self.predicted_price = predicted_price
        self.reasoning = reasoning
        self.prediction_date = prediction_date
        self.actual_price = actual_price
        self.actual_direction = actual_direction
        self.error_pct = error_pct
        self.resolved_at = resolved_at


def _fetch_predictions_from_db(db_path: str, target_date: str) -> List[PredictionRecord]:
    """Fetch all PredictionRecord rows for a given date from predictions.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT ticker, predicted_direction, confidence, predicted_price, reasoning,
                   prediction_date, actual_price, actual_direction, error_pct, resolved_at
            FROM predictions
            WHERE prediction_date = ?
            ORDER BY ticker
        """, (target_date,))
        rows = cursor.fetchall()
    finally:
        conn.close()
    
    records = []
    for row in rows:
        rec = PredictionRecord(
            ticker=row['ticker'],
            predicted_direction=row['predicted_direction'],
            confidence=row['confidence'],
            predicted_price=row['predicted_price'],
            reasoning=row['reasoning'],
            prediction_date=row['prediction_date'],
            actual_price=row['actual_price'],
            actual_direction=row['actual_direction'],
            error_pct=row['error_pct'],
            resolved_at=row['resolved_at']
        )
        records.append(rec)
    
    return records


def _fetch_actual_price(ticker: str, target_date: str) -> Optional[float]:
    """Fetch closing price for ticker on target_date via yfinance."""
    try:
        # Parse target_date (YYYY-MM-DD format)
        date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        start = date_obj
        end = date_obj + timedelta(days=1)
        
        data = yf.download(ticker, start=start, end=end, progress=False, quiet=True)
        if data.empty:
            return None
        
        # Return the closing price
        return float(data['Close'].iloc[0])
    except Exception as e:
        print(f"[WARN] Failed to fetch price for {ticker} on {target_date}: {e}")
        return None


def _compute_direction(open_price: float, close_price: float) -> str:
    """Compute direction as 'UP' or 'DOWN' based on price movement."""
    if close_price >= open_price:
        return "UP"
    else:
        return "DOWN"


def _resolve_predictions(records: List[PredictionRecord], target_date: str) -> Tuple[List[PredictionRecord], int]:
    """
    Fetch actual prices and resolve predictions.
    Returns updated records and count of newly resolved.
    """
    resolved_count = 0
    
    for rec in records:
        if rec.actual_price is None:
            actual = _fetch_actual_price(rec.ticker, target_date)
            if actual is not None:
                rec.actual_price = actual
                rec.actual_direction = _compute_direction(rec.predicted_price, actual)
                rec.error_pct = abs(actual - rec.predicted_price) / rec.predicted_price * 100
                rec.resolved_at = datetime.now().isoformat()
                resolved_count += 1
    
    return records, resolved_count


def _compute_accuracy_metrics(records: List[PredictionRecord]) -> Dict[str, float]:
    """Compute direction accuracy, avg error %, and confidence calibration."""
    resolved = [r for r in records if r.actual_direction is not None]
    
    if not resolved:
        return {
            "direction_accuracy": 0.0,
            "avg_error_pct": 0.0,
            "total_predictions": 0,
            "resolved_predictions": 0,
            "avg_confidence": 0.0
        }
    
    correct = sum(1 for r in resolved if r.predicted_direction == r.actual_direction)
    accuracy = correct / len(resolved) * 100
    avg_error = sum(r.error_pct for r in resolved) / len(resolved)
    avg_conf = sum(r.confidence for r in resolved) / len(resolved)
    
    return {
        "direction_accuracy": round(accuracy, 2),
        "avg_error_pct": round(avg_error, 2),
        "total_predictions": len(records),
        "resolved_predictions": len(resolved),
        "avg_confidence": round(avg_conf, 3)
    }


def _render_markdown_report(records: List[PredictionRecord], target_date: str,
                            metrics: Dict[str, float]) -> str:
    """Render a markdown post-mortem report."""
    timestamp = datetime.now().isoformat()
    
    md = f"""# Sentinel Post-Mortem Report
**Generated:** {timestamp}
**Prediction Date:** {target_date}

## Summary Metrics
| Metric | Value |
|--------|-------|
| Direction Accuracy | {metrics['direction_accuracy']}% |
| Avg Price Error | {metrics['avg_error_pct']}% |
| Total Predictions | {metrics['total_predictions']} |
| Resolved Predictions | {metrics['resolved_predictions']} |
| Avg Confidence | {metrics['avg_confidence']} |

## Prediction Details

"""
    
    # Table of per-ticker results
    md += "| Ticker | Direction | Confidence | Predicted Price | Actual Price | Error % | Hit? |\n"
    md += "|--------|-----------|------------|-----------------|--------------|---------|------|\n"
    
    for rec in sorted(records, key=lambda r: r.ticker):
        actual_str = f"${rec.actual_price:.2f}" if rec.actual_price else "—"
        error_str = f"{rec.error_pct:.2f}%" if rec.error_pct else "—"
        hit_str = "✓" if rec.predicted_direction == rec.actual_direction and rec.actual_direction else "✗"
        
        md += (f"| {rec.ticker} | {rec.predicted_direction} | {rec.confidence:.3f} | "
               f"${rec.predicted_price:.2f} | {actual_str} | {error_str} | {hit_str} |\n")
    
    md += "\n## Reasoning Samples\n\n"
    
    for rec in records[:5]:
        md += f"### {rec.ticker}\n"
        md += f"**Direction:** {rec.predicted_direction} (conf={rec.confidence:.3f})\n\n"
        md += f"{rec.reasoning}\n\n"
