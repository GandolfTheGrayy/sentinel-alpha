"""
Sentinel Judge — Post-Mortem Report Generator.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price data
via yfinance, compares predicted vs. actual market moves, and writes markdown
reports to backtest_results/ for performance analysis and heuristic refinement.

This module enables daily calibration of the Sentinel system's predictive models
by quantifying prediction accuracy and identifying systematic biases.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yfinance as yf
import pandas as pd


@dataclass
class PredictionRecord:
    """Represents a single prediction made by Sentinel."""
    prediction_id: str
    ticker: str
    prediction_date: str
    predicted_direction: str  # "BULL", "BEAR", "NEUTRAL"
    predicted_confidence: float  # 0.0 to 1.0
    predicted_price_target: Optional[float]
    reasoning_summary: str


@dataclass
class ActualMarketData:
    """Represents actual market movement observed."""
    ticker: str
    date: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
    price_change_pct: float


def get_db_connection(db_path: str) -> sqlite3.Connection:
    """Open SQLite connection to predictions database."""
    return sqlite3.connect(db_path)


def fetch_yesterday_predictions(db_path: str) -> list[PredictionRecord]:
    """Fetch all prediction records from yesterday."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    cursor.execute("""
        SELECT prediction_id, ticker, prediction_date, predicted_direction,
               predicted_confidence, predicted_price_target, reasoning_summary
        FROM predictions
        WHERE prediction_date = ?
        ORDER BY ticker
    """, (yesterday,))
    
    rows = cursor.fetchall()
    conn.close()
    
    records = [
        PredictionRecord(
            prediction_id=row[0],
            ticker=row[1],
            prediction_date=row[2],
            predicted_direction=row[3],
            predicted_confidence=row[4],
            predicted_price_target=row[5],
            reasoning_summary=row[6]
        )
        for row in rows
    ]
    
    return records


def fetch_actual_price_data(ticker: str, date: str) -> Optional[ActualMarketData]:
    """Fetch actual OHLCV data for ticker on given date via yfinance."""
    try:
        data = yf.download(ticker, start=date, end=date, progress=False)
        if data.empty:
            return None
        
        row = data.iloc[0]
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        price_change_pct = ((close_price - open_price) / open_price) * 100
        
        return ActualMarketData(
            ticker=ticker,
            date=date,
            open_price=open_price,
            close_price=close_price,
            high_price=float(row["High"]),
            low_price=float(row["Low"]),
            volume=int(row["Volume"]),
            price_change_pct=price_change_pct
        )
    except Exception as e:
        print(f"Error fetching data for {ticker} on {date}: {e}")
        return None


def evaluate_prediction(
    prediction: PredictionRecord,
    actual: ActualMarketData
) -> dict:
    """Compare predicted direction vs. actual market move."""
    is_correct = False
    actual_direction = "NEUTRAL"
    
    if actual.price_change_pct > 0.5:
        actual_direction = "BULL"
    elif actual.price_change_pct < -0.5:
        actual_direction = "BEAR"
    
    if prediction.predicted_direction == actual_direction:
        is_correct = True
    
    return {
        "prediction_id": prediction.prediction_id,
        "ticker": prediction.ticker,
        "predicted_direction": prediction.predicted_direction,
        "actual_direction": actual_direction,
        "is_correct": is_correct,
        "confidence": prediction.predicted_confidence,
        "price_change_pct": actual.price_change_pct,
        "open_price": actual.open_price,
        "close_price": actual.close_price,
    }


def generate_postmortem_markdown(
    predictions: list[PredictionRecord],
    evaluations: list[dict],
    report_date: str
) -> str:
    """Generate markdown post-mortem report."""
    total_predictions = len(evaluations)
    correct_predictions = sum(1 for e in evaluations if e["is_correct"])
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    avg_confidence = sum(e["confidence"] for e in evaluations) / len(evaluations) if evaluations else 0
    
    markdown = f"""# Sentinel Post-Mortem Report
**Report Date:** {report_date}

## Executive Summary
- **Total Predictions:** {total_predictions}
- **Correct Predictions:** {correct_predictions}
- **Accuracy:** {accuracy:.1f}%
- **Average Confidence:** {avg_confidence:.2f}

---

## Detailed Results

| Ticker | Predicted | Actual | Correct | Confidence | Price Change % |
|--------|-----------|--------|---------|------------|----------------|
"""
    
    for eval_result in evaluations:
        correct_mark = "✓" if eval_result["is_correct"] else "✗"
        markdown += f"""| {eval_result['ticker']} | {eval_result['predicted_direction']} | {eval_result['actual_direction']} | {correct_mark} | {eval_result['confidence']:.2f} | {eval_result['price_change_pct']:+.2f}% |
"""
    
    markdown += f"""
---

## Analysis

### Correctness by Direction
"""
    
    bull_preds = [e for e in evaluations if e["predicted_direction"] == "BULL"]
    bear_preds = [e for e in evaluations if e["predicted_direction"] == "BEAR"]
    neutral_preds = [e for e in evaluations if e["predicted_direction"] == "NEUTRAL"]
    
    if bull_preds:
        bull_acc = sum(1 for e in bull_preds if e["is_correct"]) / len(bull_preds) * 100
        markdown += f"- **BULL Predictions:** {sum(1 for e in bull_preds if e['is_correct'])}/{len(bull_preds)} correct ({bull_acc:.1f}%)\n"
    
    if bear_preds:
        bear_acc = sum(1 for e in bear_preds if e["is_correct"]) / len(bear_preds) * 100
        markdown += f"- **BEAR Predictions:** {sum(1 for e in bear_preds if e['is_correct'])}/{len(bear_preds)} correct ({bear_acc:.1f}%)\n"
    
    if neutral_preds:
        neutral_acc = sum(1 for e in neutral_preds if e["is_correct"]) / len(neutral_preds) * 100
        markdown += f"- **NEUTRAL Predictions:** {sum(1 for e in neutral_preds if e['is_correct'])}/{len(neutral_preds)} correct ({neutral_acc:.1f}%)\n"
    
    markdown += "\n### Confidence Analysis\nHigh-confidence predictions (>0.75):\n"
    high_conf = [e for e in evaluations if e["confidence"] > 0.75]
    if high_conf:
        high_conf_acc = sum(1 for e in high
