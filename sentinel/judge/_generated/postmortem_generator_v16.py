"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price data,
computes prediction accuracy metrics, and writes markdown reports to backtest_results/.
Integrates with Judge's daily calibration loop to measure model drift and heuristic effectiveness.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def get_db_path() -> str:
    """Return path to Sentinel's SQLite database."""
    return os.getenv("SENTINEL_DB_PATH", "sentinel.db")


def fetch_prediction_records(db_path: str, days_ago: int = 1) -> list[dict]:
    """
    Fetch PredictionRecord entries from yesterday (or N days ago).
    
    Returns list of dicts: {ticker, predicted_direction, predicted_confidence,
                            predicted_at, target_date, notes}.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    target_date = (datetime.utcnow() - timedelta(days=days_ago)).date()
    
    cursor.execute(
        """
        SELECT ticker, predicted_direction, predicted_confidence,
               predicted_at, target_date, notes
        FROM prediction_records
        WHERE DATE(predicted_at) = ?
        ORDER BY ticker, predicted_at
        """,
        (target_date,)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def fetch_actual_price_data(
    ticker: str, start_date: str, end_date: str
) -> Optional[dict]:
    """
    Fetch OHLCV data for ticker using yfinance.
    
    Returns dict: {open, close, high, low, volume, pct_change} or None on failure.
    """
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            return None
        
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        pct_change = ((last_row["Close"] - first_row["Open"]) / first_row["Open"]) * 100
        
        return {
            "open": float(first_row["Open"]),
            "close": float(last_row["Close"]),
            "high": float(df["High"].max()),
            "low": float(df["Low"].min()),
            "volume": int(df["Volume"].sum()),
            "pct_change": pct_change,
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def compute_accuracy(
    predicted_direction: str, actual_pct_change: float
) -> bool:
    """
    Check if prediction direction matches actual outcome.
    
    predicted_direction: 'UP', 'DOWN', or 'NEUTRAL'
    actual_pct_change: percentage change (positive = up)
    """
    threshold = 0.5  # Treat ±0.5% as neutral
    
    if predicted_direction == "UP":
        return actual_pct_change > threshold
    elif predicted_direction == "DOWN":
        return actual_pct_change < -threshold
    else:  # NEUTRAL
        return -threshold <= actual_pct_change <= threshold


def generate_postmortem_markdown(
    records: list[dict], prices: dict[str, dict]
) -> str:
    """
    Generate markdown report of predictions vs. actuals.
    
    records: list of PredictionRecord dicts
    prices: dict mapping ticker -> price_data dict
    """
    lines = [
        f"# Sentinel Post-Mortem Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
    ]
    
    if not records:
        lines.append("No predictions found for this period.")
        return "\n".join(lines)
    
    accuracy_count = 0
    total_count = len(records)
    
    lines.extend([
        "## Prediction Results",
        "",
        "| Ticker | Direction | Confidence | Actual Change | Hit | Notes |",
        "|--------|-----------|------------|---------------|-----|-------|",
    ])
    
    for rec in records:
        ticker = rec["ticker"]
        predicted_dir = rec["predicted_direction"]
        confidence = rec["predicted_confidence"]
        target_date = rec["target_date"]
        notes = rec["notes"] or ""
        
        # Fetch actual price data for target_date (±1 day for market closures)
        price_data = prices.get(ticker)
        if not price_data:
            # Attempt to fetch if not already cached
            try:
                start = datetime.strptime(target_date, "%Y-%m-%d").date()
                end = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=2)).date()
                price_data = fetch_actual_price_data(ticker, str(start), str(end))
                if price_data:
                    prices[ticker] = price_data
            except Exception:
                price_data = None
        
        if price_data:
            pct_change = price_data["pct_change"]
            hit = compute_accuracy(predicted_dir, pct_change)
            accuracy_count += hit
            hit_str = "✓" if hit else "✗"
        else:
            pct_change = "N/A"
            hit_str = "?"
        
        lines.append(
            f"| {ticker} | {predicted_dir} | {confidence:.2f} | "
            f"{pct_change if isinstance(pct_change, str) else f'{pct_change:.2f}%'} | "
            f"{hit_str} | {notes} |"
        )
    
    lines.extend([
        "",
        "## Summary",
        f"- **Total Predictions**: {total_count}",
        f"- **Accurate**: {accuracy_count} ({(accuracy_count/total_count)*100:.1f}%)" if total_count > 0 else "- **Accurate**: 0 (N/A)",
        f"- **Report Date**: {datetime.utcnow().strftime('%Y-%m-%d')}",
    ])
    
    return "\n".join(lines)


def write_postmortem_report(
    markdown_content: str, output_dir: str = "backtest_results"
) -> str:
    """
    Write markdown report to backtest_results/ directory.
    
    Returns path to written file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"postmortem_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, "w") as f:
        f.write(markdown_content)
    
    return filepath


def run_daily_postmortem(
    db_path: Optional[str] = None,
    output_dir: str = "backtest_results",
    days_ago: int = 1,
) -> str:
    """
    Execute full post-mortem pipeline: fetch predictions, actuals, generate report.
    
    Returns path to generated markdown report.
    """
    if db_path is None:
        db_path = get_db_path()
    
    # Fetch predictions from yesterday
    records = fetch_prediction_records(db_path, days_ago=days_ago)
    
    # Pre-fetch price data for all tickers
    prices = {}
    for rec in records:
        ticker = rec["ticker"]
        if ticker not in prices:
            target_date = rec["target_date"]
            try:
                start = datetime.strptime(target_date, "%Y
