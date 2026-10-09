"""
Daily summary printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON output and renders a concise console summary
of Sentinel's performance: prediction accuracy, top movers, anomalies, and
heuristic adjustments. Designed for daily standup integration and monitoring dashboards.

This module bridges the Judge's post-mortem output (judge/postmortem.py) and
human-readable reporting, enabling quick performance triage without parsing JSON.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_latest_postmortem(postmortem_dir: Path) -> Optional[Dict[str, Any]]:
    """Load the most recent post-mortem JSON file from disk."""
    if not postmortem_dir.exists():
        return None
    
    postmortem_files = sorted(
        postmortem_dir.glob("postmortem_*.json"),
        reverse=True
    )
    
    if not postmortem_files:
        return None
    
    try:
        with open(postmortem_files[0], "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def format_accuracy_section(postmortem: Dict[str, Any]) -> str:
    """Format accuracy metrics (precision, recall, Sharpe) for console output."""
    metrics = postmortem.get("metrics", {})
    
    lines = ["┌─ ACCURACY METRICS ─────────────────────────────┐"]
    lines.append(f"│ Precision (↑↓ calls):    {metrics.get('precision', 'N/A'):<8} │")
    lines.append(f"│ Recall:                  {metrics.get('recall', 'N/A'):<8} │")
    lines.append(f"│ Sharpe Ratio:            {metrics.get('sharpe_ratio', 'N/A'):<8} │")
    lines.append(f"│ Win Rate:                {metrics.get('win_rate', 'N/A'):<8} │")
    lines.append("└────────────────────────────────────────────────┘")
    
    return "\n".join(lines)


def format_predictions_section(postmortem: Dict[str, Any]) -> str:
    """Format per-ticker predictions and outcomes."""
    predictions = postmortem.get("predictions", [])
    
    lines = ["┌─ PREDICTIONS ──────────────────────────────────┐"]
    
    if not predictions:
        lines.append("│ No predictions on record.                      │")
    else:
        for pred in predictions[:10]:  # Top 10
            ticker = pred.get("ticker", "?")
            direction = pred.get("direction", "?")
            confidence = pred.get("confidence", "?")
            actual_move = pred.get("actual_move", "?")
            correct = "✓" if pred.get("correct") else "✗"
            
            summary = f"{ticker:6} {direction:4} @{confidence}% → {actual_move:6} {correct}"
            lines.append(f"│ {summary:<45} │")
    
    lines.append("└────────────────────────────────────────────────┘")
    return "\n".join(lines)


def format_anomalies_section(postmortem: Dict[str, Any]) -> str:
    """Format detected anomalies and edge cases."""
    anomalies = postmortem.get("anomalies", [])
    
    lines = ["┌─ ANOMALIES ────────────────────────────────────┐"]
    
    if not anomalies:
        lines.append("│ No anomalies detected.                         │")
    else:
        for anom in anomalies[:5]:  # Top 5
            label = anom.get("label", "unknown")
            ticker = anom.get("ticker", "?")
            severity = anom.get("severity", "low")
            
            summary = f"{severity.upper():6} {ticker:6} {label}"
            lines.append(f"│ {summary:<45} │")
    
    lines.append("└────────────────────────────────────────────────┘")
    return "\n".join(lines)


def format_heuristics_section(postmortem: Dict[str, Any]) -> str:
    """Format heuristic adjustments applied by Judge."""
    adjustments = postmortem.get("heuristic_adjustments", [])
    
    lines = ["┌─ HEURISTIC ADJUSTMENTS ────────────────────────┐"]
    
    if not adjustments:
        lines.append("│ No adjustments applied.                        │")
    else:
        for adj in adjustments[:5]:  # Top 5
            rule = adj.get("rule", "unknown")
            delta = adj.get("confidence_delta", "?")
            
            summary = f"{rule:<35} Δ{delta}"
            lines.append(f"│ {summary:<45} │")
    
    lines.append("└────────────────────────────────────────────────┘")
    return "\n".join(lines)


def print_daily_summary(postmortem_path: Optional[str] = None) -> None:
    """Print a concise daily summary of Sentinel performance to console."""
    if postmortem_path:
        postmortem_dir = Path(postmortem_path)
    else:
        # Default to sentinel/judge/postmortems/
        postmortem_dir = Path(__file__).parent.parent / "postmortems"
    
    postmortem = load_latest_postmortem(postmortem_dir)
    
    if not postmortem:
        print("❌ No post-mortem found. Run pipeline.py first.", file=sys.stderr)
        sys.exit(1)
    
    # Header
    generated_at = postmortem.get("generated_at", "unknown")
    print("\n" + "=" * 49)
    print(f"  SENTINEL DAILY SUMMARY — {generated_at}")
    print("=" * 49 + "\n")
    
    # Sections
    print(format_accuracy_section(postmortem))
    print()
    print(format_predictions_section(postmortem))
    print()
    print(format_anomalies_section(postmortem))
    print()
    print(format_heuristics_section(postmortem))
    print()
    
    # Footer with metadata
    total_preds = len(postmortem.get("predictions", []))
    total_anomalies = len(postmortem.get("anomalies", []))
    print(f"  Total predictions: {total_preds} | Anomalies flagged: {total_anomalies}")
    print("=" * 49 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Print daily summary of Sentinel performance."
    )
    parser.add_argument(
        "--postmortem-dir",
        type=str,
        default=None,
        help="Path to postmortem directory (default: sentinel/judge/postmortems/)",
    )
    
    args = parser.parse_args()
    print_daily_summary(postmortem_path=args.postmortem_dir)
