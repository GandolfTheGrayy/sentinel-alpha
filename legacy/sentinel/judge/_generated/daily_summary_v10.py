"""
Daily Summary Printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON file and renders a concise console summary
of Sentinel's prediction accuracy, signal confidence, and anomalies detected.
Integrates with Judge pillar's post-mortem workflow for performance visibility.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[Dict[str, Any]]:
    """Load the most recent post-mortem JSON file from the postmortem directory."""
    postmortem_path = Path(postmortem_dir)
    if not postmortem_path.exists():
        return None
    
    postmortem_files = sorted(postmortem_path.glob("postmortem_*.json"), reverse=True)
    if not postmortem_files:
        return None
    
    with open(postmortem_files[0], "r") as f:
        return json.load(f)


def calculate_accuracy(postmortem: Dict[str, Any]) -> float:
    """Calculate overall prediction accuracy from postmortem data."""
    predictions = postmortem.get("predictions", [])
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p.get("correct", False))
    return (correct / len(predictions)) * 100


def extract_top_signals(postmortem: Dict[str, Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Extract top confidence signals sorted by impact."""
    predictions = postmortem.get("predictions", [])
    signals = []
    
    for pred in predictions:
        for signal in pred.get("signals", []):
            signals.append({
                "ticker": pred.get("ticker"),
                "signal": signal.get("name"),
                "confidence": signal.get("confidence", 0.0),
                "impact": signal.get("impact", "neutral"),
            })
    
    signals.sort(key=lambda x: x["confidence"], reverse=True)
    return signals[:limit]


def extract_anomalies(postmortem: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract flagged anomalies from postmortem."""
    return postmortem.get("anomalies", [])


def format_accuracy_bar(accuracy: float, width: int = 20) -> str:
    """Format a simple text bar chart for accuracy visualization."""
    filled = int((accuracy / 100) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {accuracy:.1f}%"


def print_daily_summary(postmortem: Optional[Dict[str, Any]] = None) -> None:
    """Print a formatted console summary of the latest post-mortem."""
    if postmortem is None:
        postmortem = load_latest_postmortem()
    
    if postmortem is None:
        print("❌ No post-mortem data found. Run the pipeline first.")
        return
    
    timestamp = postmortem.get("timestamp", "unknown")
    print("\n" + "=" * 70)
    print(f"📊 SENTINEL DAILY SUMMARY — {timestamp}")
    print("=" * 70)
    
    # Accuracy section
    accuracy = calculate_accuracy(postmortem)
    total_predictions = len(postmortem.get("predictions", []))
    print(f"\n📈 ACCURACY")
    print(f"   {format_accuracy_bar(accuracy)}")
    print(f"   Predictions evaluated: {total_predictions}")
    
    # Top signals
    top_signals = extract_top_signals(postmortem, limit=5)
    if top_signals:
        print(f"\n⭐ TOP CONFIDENCE SIGNALS")
        for i, sig in enumerate(top_signals, 1):
            icon = "🟢" if sig["impact"] == "bullish" else "🔴" if sig["impact"] == "bearish" else "⚪"
            print(f"   {i}. {sig['ticker']:6s} | {sig['signal']:30s} | {icon} {sig['confidence']:.2f}")
    
    # Anomalies
    anomalies = extract_anomalies(postmortem)
    if anomalies:
        print(f"\n⚠️  ANOMALIES DETECTED ({len(anomalies)})")
        for anomaly in anomalies[:5]:
            print(f"   • {anomaly.get('description', 'Unknown anomaly')}")
    
    # Prediction summary by ticker
    predictions_by_ticker = {}
    for pred in postmortem.get("predictions", []):
        ticker = pred.get("ticker")
        if ticker not in predictions_by_ticker:
            predictions_by_ticker[ticker] = {"correct": 0, "total": 0}
        predictions_by_ticker[ticker]["total"] += 1
        if pred.get("correct", False):
            predictions_by_ticker[ticker]["correct"] += 1
    
    if predictions_by_ticker:
        print(f"\n📋 BY TICKER")
        for ticker in sorted(predictions_by_ticker.keys()):
            stats = predictions_by_ticker[ticker]
            pct = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"   {ticker:6s} | {stats['correct']:2d}/{stats['total']:2d} correct ({pct:5.1f}%)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_daily_summary()
