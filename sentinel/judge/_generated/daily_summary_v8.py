"""
Daily summary printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON and renders a concise console summary of
Sentinel's prediction accuracy, signal quality, and performance metrics.
Integrated into the daily build pipeline as the final human-facing output.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[dict[str, Any]]:
    """Load the most recent post-mortem JSON file from disk."""
    pm_path = Path(postmortem_dir)
    if not pm_path.exists():
        return None
    
    postmortem_files = sorted(pm_path.glob("postmortem_*.json"), reverse=True)
    if not postmortem_files:
        return None
    
    with open(postmortem_files[0], "r") as f:
        return json.load(f)


def format_accuracy_line(correct: int, total: int) -> str:
    """Format accuracy as a percentage with color-coded emoji."""
    if total == 0:
        return "N/A (no predictions)"
    pct = (correct / total) * 100
    emoji = "🟢" if pct >= 60 else "🟡" if pct >= 40 else "🔴"
    return f"{emoji} {correct}/{total} ({pct:.1f}%)"


def format_signal_quality(pm_data: dict[str, Any]) -> str:
    """Extract and format average signal certainty and hesitation metrics."""
    signals = pm_data.get("signals", [])
    if not signals:
        return "No signals analyzed"
    
    certainties = [s.get("certainty_score", 0.5) for s in signals]
    avg_certainty = sum(certainties) / len(certainties) if certainties else 0.5
    
    hesitations = [s.get("hesitation_count", 0) for s in signals]
    avg_hesitation = sum(hesitations) / len(hesitations) if hesitations else 0
    
    return f"Certainty: {avg_certainty:.2f} | Avg Hesitations: {avg_hesitation:.1f}"


def print_daily_summary(postmortem_dir: str = "sentinel/judge/postmortems") -> None:
    """Load latest post-mortem and print concise summary to stdout."""
    pm_data = load_latest_postmortem(postmortem_dir)
    
    if pm_data is None:
        print("❌ No post-mortem found. Sentinel has not run yet.")
        return
    
    timestamp = pm_data.get("timestamp", "unknown")
    predictions = pm_data.get("predictions", [])
    correct = pm_data.get("correct_count", 0)
    total = len(predictions)
    
    print("\n" + "=" * 70)
    print(f"📊 SENTINEL DAILY SUMMARY — {timestamp}")
    print("=" * 70)
    
    print(f"\n✓ Accuracy: {format_accuracy_line(correct, total)}")
    print(f"✓ Signal Quality: {format_signal_quality(pm_data)}")
    
    if "anomalies" in pm_data and pm_data["anomalies"]:
        print(f"\n⚠️  Anomalies Detected: {len(pm_data['anomalies'])}")
        for anomaly in pm_data["anomalies"][:3]:
            print(f"   • {anomaly.get('ticker', '?')}: {anomaly.get('reason', 'unknown')}")
    
    if "top_movers" in pm_data and pm_data["top_movers"]:
        print(f"\n📈 Top Performers:")
        for ticker, score in pm_data["top_movers"][:3]:
            print(f"   • {ticker}: {score:.2f}")
    
    print(f"\n💾 Full post-mortem: {postmortem_dir}/postmortem_*.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    print_daily_summary()
