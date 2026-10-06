"""
Daily summary printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON/YAML from sentinel/judge/postmortem.py output,
parses performance metrics (predicted vs. actual moves, accuracy, anomalies),
and prints a concise, human-readable console summary for morning briefing.

Fits into Judge pillar: post-execution analysis and calibration feedback loop.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[dict[str, Any]]:
    """Load the most recent postmortem file from the postmortem directory."""
    postmortem_path = Path(postmortem_dir)
    if not postmortem_path.exists():
        return None
    
    postmortem_files = sorted(
        postmortem_path.glob("postmortem_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not postmortem_files:
        return None
    
    latest_file = postmortem_files[0]
    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def extract_summary_stats(postmortem: dict[str, Any]) -> dict[str, Any]:
    """Extract key performance stats from postmortem data."""
    predictions = postmortem.get("predictions", [])
    timestamp = postmortem.get("timestamp", "unknown")
    
    total_predictions = len(predictions)
    correct_predictions = sum(1 for p in predictions if p.get("correct", False))
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
    
    avg_confidence = sum(p.get("confidence", 0) for p in predictions) / total_predictions if total_predictions > 0 else 0.0
    
    anomalies = [p for p in predictions if p.get("anomaly", False)]
    
    return {
        "timestamp": timestamp,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions,
        "accuracy_pct": round(accuracy, 2),
        "avg_confidence": round(avg_confidence, 3),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def format_console_summary(stats: dict[str, Any]) -> str:
    """Format summary statistics into a concise console-friendly string."""
    lines = [
        "=" * 70,
        "SENTINEL DAILY SUMMARY",
        "=" * 70,
        f"Generated: {stats['timestamp']}",
        "",
        f"Predictions Evaluated:  {stats['total_predictions']}",
        f"Correct:                {stats['correct_predictions']} / {stats['total_predictions']}",
        f"Accuracy:               {stats['accuracy_pct']}%",
        f"Avg Confidence:         {stats['avg_confidence']}",
        "",
    ]
    
    if stats["anomaly_count"] > 0:
        lines.append(f"⚠️  ANOMALIES DETECTED: {stats['anomaly_count']}")
        for anomaly in stats["anomalies"][:5]:
            ticker = anomaly.get("ticker", "UNKNOWN")
            reason = anomaly.get("reason", "unknown reason")
            lines.append(f"    • {ticker}: {reason}")
        if stats["anomaly_count"] > 5:
            lines.append(f"    ... and {stats['anomaly_count'] - 5} more")
        lines.append("")
    else:
        lines.append("✓ No anomalies flagged")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def print_daily_summary(postmortem_dir: str = "sentinel/judge/postmortems") -> None:
    """Main entry point: load postmortem and print daily summary to console."""
    postmortem = load_latest_postmortem(postmortem_dir)
    
    if not postmortem:
        print("No postmortem data found. Sentinel has not yet run today.")
        return
    
    stats = extract_summary_stats(postmortem)
    summary = format_console_summary(stats)
    print(summary)


if __name__ == "__main__":
    print_daily_summary()
