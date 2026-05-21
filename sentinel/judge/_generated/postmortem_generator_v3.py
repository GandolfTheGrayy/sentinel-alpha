"""
Post-mortem report generator for Sentinel Sentiment Engine.

Reads yesterday's PredictionRecord entries from SQLite, fetches actual price
movements via yfinance, computes prediction accuracy metrics, and writes
markdown reports to backtest_results/. Integrates with Judge pillar for
daily performance calibration and heuristic refinement.
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def get_predictions_from_db(
    db_path: str, prediction_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch PredictionRecord entries from SQLite for a given date.

    Args:
        db_path: Path to Sentinel SQLite database.
        prediction_date: YYYY-MM-DD string; defaults to yesterday.

    Returns:
        DataFrame with columns: ticker, predicted_direction, confidence,
        predicted_date, created_at.
    """
    if prediction_date is None:
        prediction_date = (datetime.utcnow() - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

    conn = sqlite3.connect(db_path)
    query = """
    SELECT ticker, predicted_direction, confidence, predicted_date, created_at
    FROM prediction_records
    WHERE DATE(predicted_date) = ?
    ORDER BY created_at DESC
    """
    df = pd.read_sql_query(query, conn, params=(prediction_date,))
    conn.close()
    return df


def fetch_actual_prices(
    ticker: str, target_date: str, lookback_days: int = 5
) -> Optional[dict]:
    """
    Fetch actual price data for a ticker on and around target_date.

    Args:
        ticker: Stock ticker symbol.
        target_date: YYYY-MM-DD prediction date.
        lookback_days: Days before target_date to include.

    Returns:
        Dict with keys: open, close, high, low, volume, pct_change, or None
        if fetch fails.
    """
    try:
        target = datetime.strptime(target_date, "%Y-%m-%d")
        start = target - timedelta(days=lookback_days)

        data = yf.download(ticker, start=start.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return None

        target_row = data.loc[target.strftime("%Y-%m-%d") :]
        if target_row.empty:
            target_row = data.iloc[-1:]

        if target_row.empty:
            return None

        row = target_row.iloc[0]
        return {
            "open": float(row["Open"]),
            "close": float(row["Close"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"]),
            "pct_change": float(row["Close"] - data.iloc[-2]["Close"])
            / float(data.iloc[-2]["Close"])
            * 100,
        }
    except Exception:
        return None


def compute_accuracy(predicted_direction: str, actual_pct_change: float) -> bool:
    """
    Check if prediction direction matches actual price movement.

    Args:
        predicted_direction: "UP", "DOWN", or "NEUTRAL".
        actual_pct_change: Percentage change in price.

    Returns:
        True if prediction was correct, False otherwise.
    """
    if predicted_direction == "UP":
        return actual_pct_change > 0.5
    elif predicted_direction == "DOWN":
        return actual_pct_change < -0.5
    else:
        return -0.5 <= actual_pct_change <= 0.5


def generate_postmortem_report(
    predictions_df: pd.DataFrame, output_dir: str = "backtest_results"
) -> str:
    """
    Generate markdown post-mortem report from predictions and actual prices.

    Args:
        predictions_df: DataFrame of predictions (from get_predictions_from_db).
        output_dir: Directory to write markdown report.

    Returns:
        Path to generated markdown file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_date = datetime.utcnow().strftime("%Y-%m-%d")
    report_path = os.path.join(output_dir, f"postmortem_{report_date}.md")

    lines = [
        f"# Sentinel Post-Mortem Report",
        f"**Generated:** {datetime.utcnow().isoformat()}",
        f"**Prediction Date:** {report_date}",
        "",
        "## Summary",
        "",
    ]

    if predictions_df.empty:
        lines.append("No predictions found for this date.")
    else:
        results = []
        correct = 0
        total = 0

        for _, row in predictions_df.iterrows():
            ticker = row["ticker"]
            predicted_direction = row["predicted_direction"]
            confidence = row["confidence"]
            predicted_date = row["predicted_date"]

            actual = fetch_actual_prices(ticker, predicted_date)
            if actual is None:
                continue

            total += 1
            is_correct = compute_accuracy(
                predicted_direction, actual["pct_change"]
            )
            if is_correct:
                correct += 1

            results.append(
                {
                    "ticker": ticker,
                    "predicted": predicted_direction,
                    "confidence": confidence,
                    "actual_pct": actual["pct_change"],
                    "correct": is_correct,
                    "close": actual["close"],
                }
            )

        if total > 0:
            accuracy = (correct / total) * 100
            lines.append(f"- **Accuracy:** {accuracy:.1f}% ({correct}/{total})")
            lines.append(f"- **Total Predictions:** {total}")
            lines.append("")
            lines.append("## Detailed Results")
            lines.append("")
            lines.append("| Ticker | Predicted | Confidence | Actual % | Result |")
            lines.append("|--------|-----------|-----------|---------|--------|")

            for res in results:
                status = "✓ HIT" if res["correct"] else "✗ MISS"
                lines.append(
                    f"| {res['ticker']} | {res['predicted']} | {res['confidence']:.2f} | "
                    f"{res['actual_pct']:+.2f}% | {status} |"
                )

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


def main(db_path: str = "sentinel.db", output_dir: str = "backtest_results") -> None:
    """
    Main entry point: fetch predictions, compute actuals, generate report.

    Args:
        db_path: Path to Sentinel SQLite database.
        output_dir: Directory for markdown output.
    """
    predictions = get_predictions_from_db(db_path)
    report_path = generate_postmortem_report(predictions, output_dir)
    print(f"Post-mortem report written to: {report_path}")


if __name__ == "__main__":
    main()
