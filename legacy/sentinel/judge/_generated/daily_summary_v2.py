"""
Daily Summary Printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON file and prints a concise console summary
of Sentinel's daily performance, including prediction accuracy, top movers,
and key insights. Integrates with the Judge pillar's post-mortem workflow.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[dict]:
    """Load the most recent post-mortem JSON file from the postmortem directory."""
    try:
        pm_path = Path(postmortem_dir)
        if not pm_path.exists():
            print(f"[WARN] Postmortem directory not found: {postmortem_dir}")
            return None

        postmortem_files = sorted(pm_path.glob("postmortem_*.json"), reverse=True)
        if not postmortem_files:
            print(f"[WARN] No postmortem files found in {postmortem_dir}")
            return None

        latest_file = postmortem_files[0]
        with open(latest_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load postmortem: {e}")
        return None


def format_accuracy_section(postmortem: dict) -> str:
    """Format accuracy metrics from postmortem data."""
    lines = []
    lines.append("╔══════════════════════════════════════════╗")
    lines.append("║         SENTINEL DAILY PERFORMANCE       ║")
    lines.append("╚══════════════════════════════════════════╝")
    lines.append("")

    timestamp = postmortem.get("timestamp", "unknown")
    lines.append(f"📅 Report Date: {timestamp}")
    lines.append("")

    total_predictions = postmortem.get("total_predictions", 0)
    correct_predictions = postmortem.get("correct_predictions", 0)
    accuracy = (
        (correct_predictions / total_predictions * 100)
        if total_predictions > 0
        else 0.0
    )

    lines.append(f"🎯 Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")

    overall_return = postmortem.get("overall_return", 0.0)
    lines.append(f"💰 Overall Return: {overall_return:+.2f}%")

    return "\n".join(lines)


def format_ticker_results(postmortem: dict) -> str:
    """Format per-ticker prediction results."""
    lines = []
    lines.append("")
    lines.append("┌─ Per-Ticker Results ─────────────────────┐")

    results = postmortem.get("results", {})
    if not results:
        lines.append("│ No ticker results available             │")
        lines.append("└──────────────────────────────────────────┘")
        return "\n".join(lines)

    for ticker, data in sorted(results.items()):
        predicted = data.get("predicted_direction", "unknown")
        actual = data.get("actual_direction", "unknown")
        correct = data.get("correct", False)
        price_change = data.get("price_change_pct", 0.0)

        status = "✓" if correct else "✗"
        lines.append(
            f"│ {status} {ticker:6s} | Pred: {predicted:4s} | "
            f"Actual: {actual:4s} | Δ {price_change:+.2f}%"
        )

    lines.append("└──────────────────────────────────────────┘")
    return "\n".join(lines)


def format_insights_section(postmortem: dict) -> str:
    """Format key insights and anomalies."""
    lines = []
    lines.append("")
    lines.append("┌─ Key Insights ────────────────────────────┐")

    insights = postmortem.get("insights", [])
    if not insights:
        lines.append("│ No special insights recorded             │")
    else:
        for insight in insights[:5]:  # Limit to 5 insights
            truncated = insight[: 40] if len(insight) > 40 else insight
            lines.append(f"│ • {truncated}")

    anomalies = postmortem.get("anomalies", [])
    if anomalies:
        lines.append("│                                          │")
        lines.append("│ 🚨 Anomalies Detected:                  │")
        for anomaly in anomalies[:3]:
            truncated = anomaly[: 35] if len(anomaly) > 35 else anomaly
            lines.append(f"│   ⚠ {truncated}")

    lines.append("└──────────────────────────────────────────┘")
    return "\n".join(lines)


def format_model_health(postmortem: dict) -> str:
    """Format model health and confidence metrics."""
    lines = []
    lines.append("")
    lines.append("┌─ Model Health ────────────────────────────┐")

    avg_confidence = postmortem.get("avg_confidence", 0.0)
    lines.append(f"│ Avg Confidence: {avg_confidence:.1%}")

    consistency = postmortem.get("consistency_score", 0.0)
    lines.append(f"│ Consistency:    {consistency:.1%}")

    rag_quality = postmortem.get("rag_quality_score", 0.0)
    lines.append(f"│ RAG Quality:    {rag_quality:.1%}")

    lines.append("└──────────────────────────────────────────┘")
    return "\n".join(lines)


def print_daily_summary(postmortem_dir: str = "sentinel/judge/postmortems") -> None:
    """Load and print a formatted daily summary from the latest postmortem."""
    postmortem = load_latest_postmortem(postmortem_dir)
    if not postmortem:
        print("[ERROR] No valid postmortem to summarize.")
        return

    output = []
    output.append(format_accuracy_section(postmortem))
    output.append(format_ticker_results(postmortem))
    output.append(format_model_health(postmortem))
    output.append(format_insights_section(postmortem))
    output.append("")
    output.append(f"Generated at {datetime.now().isoformat()}")

    summary = "\n".join(output)
    print(summary)


if __name__ == "__main__":
    print_daily_summary()
