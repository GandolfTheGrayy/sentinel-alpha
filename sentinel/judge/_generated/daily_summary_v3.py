"""
Daily Summary Printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON output and prints a concise console summary
of Sentinel's performance, including prediction accuracy, win rate, top movers,
and anomalies flagged by the Judge pillar.

Integrates with sentinel/judge/postmortem.py output and provides human-friendly
reporting for stakeholders.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_latest_postmortem(postmortem_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the most recent post-mortem JSON file from the postmortem directory."""
    if not postmortem_dir.exists():
        return None

    postmortem_files = sorted(
        postmortem_dir.glob("postmortem_*.json"), reverse=True
    )
    if not postmortem_files:
        return None

    try:
        with open(postmortem_files[0], "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def format_accuracy_section(postmortem: Dict[str, Any]) -> str:
    """Format the accuracy and performance metrics section."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("SENTINEL DAILY PERFORMANCE SUMMARY")
    lines.append("=" * 70)

    # Timestamp
    if "timestamp" in postmortem:
        lines.append(f"Report Date: {postmortem['timestamp']}")
    else:
        lines.append(f"Report Date: {datetime.utcnow().isoformat()}Z")

    lines.append("")

    # Overall metrics
    metrics = postmortem.get("metrics", {})
    lines.append("OVERALL METRICS:")
    lines.append(
        f"  Total Predictions: {metrics.get('total_predictions', 'N/A')}"
    )
    lines.append(
        f"  Correct Predictions: {metrics.get('correct_predictions', 'N/A')}"
    )
    lines.append(f"  Win Rate: {metrics.get('win_rate', 'N/A')}%")
    lines.append(
        f"  Mean Absolute Error: {metrics.get('mean_absolute_error', 'N/A')}"
    )

    return "\n".join(lines)


def format_predictions_section(postmortem: Dict[str, Any]) -> str:
    """Format the per-ticker predictions and outcomes."""
    lines = []
    lines.append("\n" + "-" * 70)
    lines.append("PREDICTION OUTCOMES (Last 10)")
    lines.append("-" * 70)

    predictions = postmortem.get("predictions", [])
    for pred in predictions[-10:]:
        ticker = pred.get("ticker", "???")
        predicted_direction = pred.get("predicted_direction", "N/A")
        predicted_magnitude = pred.get("predicted_magnitude", "N/A")
        actual_direction = pred.get("actual_direction", "N/A")
        actual_change = pred.get("actual_change_pct", "N/A")
        confidence = pred.get("confidence_score", "N/A")
        correct = pred.get("correct", False)

        status = "✓ CORRECT" if correct else "✗ WRONG"
        lines.append(f"\n  {ticker} [{status}]")
        lines.append(
            f"    Predicted: {predicted_direction} ({predicted_magnitude})"
        )
        lines.append(
            f"    Actual: {actual_direction} ({actual_change}%)"
        )
        lines.append(f"    Confidence: {confidence}")

    return "\n".join(lines)


def format_anomalies_section(postmortem: Dict[str, Any]) -> str:
    """Format any anomalies or red flags detected by the Judge."""
    lines = []
    lines.append("\n" + "-" * 70)
    lines.append("ANOMALIES & FLAGS")
    lines.append("-" * 70)

    anomalies = postmortem.get("anomalies", [])
    if not anomalies:
        lines.append("  No anomalies detected.")
    else:
        for anomaly in anomalies:
            ticker = anomaly.get("ticker", "???")
            flag_type = anomaly.get("type", "UNKNOWN")
            description = anomaly.get("description", "")
            severity = anomaly.get("severity", "INFO")
            lines.append(f"\n  [{severity}] {ticker} — {flag_type}")
            lines.append(f"    {description}")

    return "\n".join(lines)


def format_top_movers_section(postmortem: Dict[str, Any]) -> str:
    """Format the top gainers and losers."""
    lines = []
    lines.append("\n" + "-" * 70)
    lines.append("TOP MOVERS")
    lines.append("-" * 70)

    top_gainers = postmortem.get("top_gainers", [])
    top_losers = postmortem.get("top_losers", [])

    lines.append("\n  Gainers:")
    if top_gainers:
        for item in top_gainers[:5]:
            ticker = item.get("ticker", "???")
            change = item.get("change_pct", "N/A")
            lines.append(f"    {ticker}: +{change}%")
    else:
        lines.append("    None")

    lines.append("\n  Losers:")
    if top_losers:
        for item in top_losers[:5]:
            ticker = item.get("ticker", "???")
            change = item.get("change_pct", "N/A")
            lines.append(f"    {ticker}: {change}%")
    else:
        lines.append("    None")

    return "\n".join(lines)


def format_summary_stats(postmortem: Dict[str, Any]) -> str:
    """Format final summary statistics and next steps."""
    lines = []
    lines.append("\n" + "-" * 70)
    lines.append("NEXT STEPS")
    lines.append("-" * 70)

    notes = postmortem.get("notes", [])
    if notes:
        for note in notes:
            lines.append(f"  • {note}")
    else:
        lines.append("  • Continue monitoring sentiment signals and market movements.")
        lines.append("  • Review heuristic calibration for improved accuracy.")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


def print_daily_summary(postmortem_path: Optional[Path] = None) -> None:
    """
    Load and print a formatted daily summary of Sentinel's performance.

    Args:
        postmortem_path: Optional explicit path to postmortem JSON file.
                        If None, searches default location.
    """
    if postmortem_path is None:
        postmortem_dir = Path("sentinel/judge/postmortem")
        postmortem = load_latest_postmortem(postmortem_dir)
    else:
        try:
            with open(postmortem_path, "r") as f:
                postmortem = json.load(f)
        except (json.JSONDecodeError, IOError, FileNotFoundError):
            postmortem = None

    if postmortem is None:
        print("\nNo post-mortem data available. Run the daily pipeline first.")
        sys.exit(1)

    # Render all sections
    output = ""
    output += format_accuracy_section(postmortem)
    output += format_predictions_section(postmortem)
    output += format_anomalies_section(postmortem)
    output += format_top_movers_section(postmortem)
    output += format_summary_stats(postmortem)

    print(output)


if __name__ == "__main__":
    postmortem_arg = sys.argv[1] if len(sys.argv) > 1 else None
