"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, compares predicted vs. actual outcomes, and renders
a markdown report to backtest_results/. Integrates with Judge pillar for
daily performance calibration and anomaly detection.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import pandas as pd


def get_db_path() -> str:
    """Return the path to the Sentinel SQLite database."""
    db_dir = Path(__file__).parent.parent.parent / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "sentinel.db")


def fetch_prediction_records(db_path: str, target_date: Optional[str] = None) -> List[Dict]:
    """
    Fetch all PredictionRecord entries for a given date (default: yesterday).
    
    Returns list of dicts with keys: ticker, predicted_direction, confidence,
    created_at, prediction_date, rationale.
    """
    if target_date is None:
        target_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT ticker, predicted_direction, confidence, created_at, 
               prediction_date, rationale
        FROM predictions
        WHERE DATE(prediction_date) = ?
        ORDER BY ticker, created_at
        """,
        (target_date,)
    )
    
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records


def fetch_actual_price_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> Optional[Dict[str, float]]:
    """
    Fetch OHLCV data for ticker between start_date and end_date.
    
    Returns dict with keys: open, close, high, low, volume.
    Returns None if data fetch fails.
    """
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            ignore_tz=True
        )
        
        if data.empty:
            return None
        
        # Get first and last trading day
        first_row = data.iloc[0]
        last_row = data.iloc[-1]
        
        return {
            "open_price": float(first_row["Open"]),
            "close_price": float(last_row["Close"]),
            "high_price": float(data["High"].max()),
            "low_price": float(data["Low"].min()),
            "volume": int(last_row["Volume"]),
            "pct_change": ((last_row["Close"] - first_row["Open"]) / first_row["Open"]) * 100
        }
    except Exception as e:
        print(f"[WARN] Failed to fetch price data for {ticker}: {e}")
        return None


def compute_outcome(
    predicted_direction: str,
    actual_pct_change: float
) -> Tuple[bool, str]:
    """
    Determine if prediction was correct and classify outcome.
    
    Returns (is_correct, outcome_label) where outcome_label is one of:
    "HIT_UP", "HIT_DOWN", "HIT_FLAT", "MISS_UP", "MISS_DOWN", "MISS_FLAT"
    """
    # Classify actual direction (±0.5% tolerance for "flat")
    if actual_pct_change > 0.5:
        actual_direction = "UP"
    elif actual_pct_change < -0.5:
        actual_direction = "DOWN"
    else:
        actual_direction = "FLAT"
    
    # Normalize prediction direction
    pred_norm = predicted_direction.upper().strip()
    if pred_norm in ("UP", "BULLISH", "BUY"):
        pred_norm = "UP"
    elif pred_norm in ("DOWN", "BEARISH", "SELL"):
        pred_norm = "DOWN"
    else:
        pred_norm = "FLAT"
    
    is_correct = (pred_norm == actual_direction)
    outcome_label = f"{'HIT' if is_correct else 'MISS'}_{actual_direction}"
    
    return is_correct, outcome_label


def generate_postmortem_markdown(
    records: List[Dict],
    price_data: Dict[str, Dict],
    output_dir: str = "backtest_results"
) -> str:
    """
    Generate markdown report comparing predictions vs. actuals.
    
    Returns the markdown string and writes to backtest_results/{date}.md
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    report_date = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = Path(output_dir) / f"{report_date}.md"
    
    # Build markdown
    lines = [
        f"# Sentinel Post-Mortem Report",
        f"**Date:** {report_date}",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        "",
        "## Summary",
        ""
    ]
    
    if not records:
        lines.append("No predictions for this period.")
        markdown = "\n".join(lines)
        report_path.write_text(markdown)
        return markdown
    
    # Compute statistics
    outcomes = []
    hits = 0
    misses = 0
    total_confidence = 0.0
    
    details = []
    
    for rec in records:
        ticker = rec["ticker"]
        conf = float(rec["confidence"]) if rec["confidence"] else 0.0
        pred_dir = rec["predicted_direction"]
        rationale = rec["rationale"] or "(no rationale)"
        
        total_confidence += conf
        
        if ticker not in price_data:
            details.append({
                "ticker": ticker,
                "prediction": pred_dir,
                "confidence": conf,
                "outcome": "NO_DATA",
                "actual_pct": None,
                "hit": False,
                "rationale": rationale
            })
            misses += 1
            continue
        
        prices = price_data[ticker]
        actual_pct = prices["pct_change"]
        is_correct, outcome_label = compute_outcome(pred_dir, actual_pct)
        
        details.append({
            "ticker": ticker,
            "prediction": pred_dir,
            "confidence": conf,
            "outcome": outcome_label,
            "actual_pct": actual_pct,
            "hit": is_correct,
            "rationale": rationale
        })
        
        outcomes.append(outcome_label)
        if is_correct:
            hits += 1
        else:
            misses += 1
    
    total_preds = hits + misses
    hit_rate = (hits / total_preds * 100) if total_preds > 0 else 0
    avg_confidence = (total_confidence / len(records)) if records else 0
    
    lines.extend([
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Predictions | {total_preds} |",
        f"| Hits | {hits} |",
        f"| Misses | {misses} |",
        f"| Hit Rate | {hit_rate:.1f}% |",
        f"| Avg Confidence | {avg_confidence:.2f} |",
        "",
        "## Outcome Distribution",
        ""
    ])
    
    from collections import Counter
    outcome_counts = Counter(outcomes)
    for outcome, count in sorted(outcome_counts.items()):
        lines.append(
