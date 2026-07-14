"""
Sentinel Post-Mortem Report Generator.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, compares predicted vs. actual outcomes, and generates
markdown reports in backtest_results/. Part of the Judge pillar's daily
calibration loop.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def get_db_connection(db_path: str = "sentinel.db") -> sqlite3.Connection:
    """Open connection to Sentinel's SQLite database."""
    return sqlite3.connect(db_path)


def fetch_prediction_records(
    conn: sqlite3.Connection, date_str: str
) -> list[dict]:
    """
    Fetch all PredictionRecord rows for a given date.

    Args:
        conn: SQLite connection object.
        date_str: ISO date string (YYYY-MM-DD) to query.

    Returns:
        List of dicts with keys: ticker, predicted_direction, predicted_confidence,
        predicted_target_price, created_at.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT ticker, predicted_direction, predicted_confidence,
               predicted_target_price, created_at
        FROM prediction_record
        WHERE DATE(created_at) = ?
        ORDER BY ticker, created_at
        """,
        (date_str,),
    )
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetch_actual_price_movement(
    ticker: str, date_str: str
) -> Optional[dict]:
    """
    Fetch opening and closing prices for a ticker on a given date via yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        date_str: ISO date string (YYYY-MM-DD).

    Returns:
        Dict with keys: open_price, close_price, high, low, volume.
        Returns None if data unavailable.
    """
    try:
        start = datetime.fromisoformat(date_str)
        end = start + timedelta(days=1)
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        row = data.iloc[0]
        return {
            "open_price": float(row["Open"]),
            "close_price": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"]),
        }
    except Exception as e:
        print(f"Error fetching price for {ticker} on {date_str}: {e}")
        return None


def compute_outcome(
    predicted_direction: str, open_price: float, close_price: float
) -> str:
    """
    Determine if prediction was correct based on actual price movement.

    Args:
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        open_price: Opening price on the day.
        close_price: Closing price on the day.

    Returns:
        "CORRECT", "INCORRECT", or "NEUTRAL_MATCH".
    """
    actual_move = "UP" if close_price > open_price else "DOWN"
    if actual_move == predicted_direction:
        return "CORRECT"
    elif predicted_direction == "NEUTRAL":
        return "NEUTRAL_MATCH"
    else:
        return "INCORRECT"


def generate_markdown_report(
    date_str: str, predictions: list[dict], actuals: dict
) -> str:
    """
    Generate a markdown post-mortem report comparing predictions vs. actuals.

    Args:
        date_str: ISO date string for the report period.
        predictions: List of prediction dicts (from DB).
        actuals: Dict mapping ticker -> actual price movement dict.

    Returns:
        Markdown string ready to write to file.
    """
    lines = [
        f"# Sentinel Post-Mortem Report — {date_str}",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]

    if not predictions:
        lines.append("No predictions found for this date.")
        return "\n".join(lines)

    correct_count = 0
    incorrect_count = 0
    no_data_count = 0
    records = []

    for pred in predictions:
        ticker = pred["ticker"]
        actual = actuals.get(ticker)

        if actual is None:
            no_data_count += 1
            outcome = "NO_DATA"
            pct_move = "N/A"
        else:
            outcome = compute_outcome(
                pred["predicted_direction"],
                actual["open_price"],
                actual["close_price"],
            )
            pct_move = (
                f"{100 * (actual['close_price'] - actual['open_price']) / actual['open_price']:.2f}%"
            )
            if outcome == "CORRECT":
                correct_count += 1
            elif outcome == "INCORRECT":
                incorrect_count += 1

        records.append(
            {
                "Ticker": ticker,
                "Prediction": pred["predicted_direction"],
                "Confidence": f"{pred['predicted_confidence']:.2f}",
                "Target": f"${pred['predicted_target_price']:.2f}"
                if pred["predicted_target_price"]
                else "N/A",
                "Outcome": outcome,
                "% Move": pct_move,
            }
        )

    total = len(predictions)
    accuracy = (
        100 * correct_count / (total - no_data_count)
        if (total - no_data_count) > 0
        else 0
    )

    lines.append(f"- **Total Predictions:** {total}")
    lines.append(f"- **Correct:** {correct_count}")
    lines.append(f"- **Incorrect:** {incorrect_count}")
    lines.append(f"- **No Data:** {no_data_count}")
    lines.append(f"- **Accuracy (excluding no-data):** {accuracy:.1f}%")
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| Ticker | Prediction | Confidence | Target | Outcome | % Move |"
    )
    lines.append("|--------|------------|------------|--------|---------|--------|")

    for record in records:
        lines.append(
            f"| {record['Ticker']} | {record['Prediction']} | "
            f"{record['Confidence']} | {record['Target']} | "
            f"{record['Outcome']} | {record['% Move']} |"
        )

    lines.append("")
    return "\n".join(lines)


def run_postmortem(date_str: Optional[str] = None) -> None:
    """
    Execute post-mortem workflow: fetch predictions, actuals, generate report.

    Args:
        date_str: ISO date string (YYYY-MM-DD). Defaults to yesterday.
    """
    if date_str is None:
        date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[PostMortem] Generating report for {date_str}...")

    conn = get_db_connection()
    try:
        predictions = fetch_prediction_records(conn, date_str)

        if not predictions:
            print(f"No predictions found for {date_str}.")
            return

        actuals = {}
        for pred in predictions:
            ticker = pred["ticker"]
            actual = fetch_actual_price_movement(ticker, date_str)
            if actual:
                actuals[ticker] = actual

        report = generate_markdown_report(date_str, predictions, actuals)

        backtest_dir = Path("backtest_results")
        backtest_dir.mkdir(exist_ok=True)
