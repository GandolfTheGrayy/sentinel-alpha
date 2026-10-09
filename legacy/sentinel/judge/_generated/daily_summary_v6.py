"""
Daily summary printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON output and prints a concise console summary
of Sentinel's performance metrics, prediction accuracy, and anomalies flagged.
Used by the pipeline orchestrator to surface daily results to operators.
"""

import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def load_latest_postmortem(postmortem_dir: Path) -> Optional[dict]:
    """Load the most recent post-mortem JSON file from disk."""
    if not postmortem_dir.exists():
        return None
    
    json_files = sorted(postmortem_dir.glob("postmortem_*.json"), reverse=True)
    if not json_files:
        return None
    
    with open(json_files[0], "r") as f:
        return json.load(f)


def format_performance_line(metric_name: str, value: float, unit: str = "%") -> str:
    """Format a single performance metric for display."""
    return f"  {metric_name:<30} {value:>7.2f}{unit}"


def print_daily_summary(postmortem_dir: Optional[Path] = None) -> None:
    """
    Print a concise console summary of Sentinel's latest performance.
    
    Reads post-mortem JSON and outputs metrics, predictions, and anomalies
    to stdout in human-readable tabular format.
    """
    if postmortem_dir is None:
        postmortem_dir = Path(__file__).parent.parent.parent / "data" / "postmortems"
    
    pm = load_latest_postmortem(postmortem_dir)
    
    if pm is None:
        print("❌ No post-mortem data found. Pipeline may not have run yet.")
        return
    
    pm_date = pm.get("date", "unknown")
    print(f"\n{'='*70}")
    print(f"  SENTINEL DAILY SUMMARY — {pm_date}")
    print(f"{'='*70}\n")
    
    # Performance metrics
    if "metrics" in pm:
        metrics = pm["metrics"]
        print("📊 PERFORMANCE METRICS")
        print("-" * 70)
        
        if "accuracy" in metrics:
            print(format_performance_line("Accuracy", metrics["accuracy"]))
        if "precision" in metrics:
            print(format_performance_line("Precision", metrics["precision"]))
        if "recall" in metrics:
            print(format_performance_line("Recall", metrics["recall"]))
        if "f1_score" in metrics:
            print(format_performance_line("F1 Score", metrics["f1_score"]))
        if "total_predictions" in metrics:
            print(f"  {'Total Predictions':<30} {metrics['total_predictions']:>7.0f}")
        if "correct_predictions" in metrics:
            print(f"  {'Correct Predictions':<30} {metrics['correct_predictions']:>7.0f}")
        
        print()
    
    # Predictions summary
    if "predictions" in pm:
        preds = pm["predictions"]
        if isinstance(preds, list) and preds:
            print("🎯 PREDICTION SUMMARY")
            print("-" * 70)
            print(f"  {'Ticker':<10} {'Direction':<12} {'Confidence':<12} {'Actual':<12} {'Status'}")
            print("-" * 70)
            
            for pred in preds[:10]:  # Show top 10
                ticker = pred.get("ticker", "N/A")[:10]
                direction = pred.get("direction", "HOLD")[:12]
                confidence = pred.get("confidence", 0.0)
                actual = pred.get("actual_move", "N/A")[:12]
                status = "✓" if pred.get("correct", False) else "✗"
                print(f"  {ticker:<10} {direction:<12} {confidence:>10.1f}% {actual:<12} {status}")
            
            if len(preds) > 10:
                print(f"  ... and {len(preds) - 10} more")
            print()
    
    # Anomalies
    if "anomalies" in pm and pm["anomalies"]:
        print("⚠️  FLAGGED ANOMALIES")
        print("-" * 70)
        for anomaly in pm["anomalies"][:5]:
            print(f"  • {anomaly}")
        if len(pm["anomalies"]) > 5:
            print(f"  ... and {len(pm['anomalies']) - 5} more")
        print()
    
    # Heuristic notes
    if "heuristic_notes" in pm and pm["heuristic_notes"]:
        print("💡 HEURISTIC REFINEMENTS")
        print("-" * 70)
        for note in pm["heuristic_notes"][:3]:
            print(f"  • {note}")
        print()
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    postmortem_path = None
    if len(sys.argv) > 1:
        postmortem_path = Path(sys.argv[1])
    
    print_daily_summary(postmortem_path)
