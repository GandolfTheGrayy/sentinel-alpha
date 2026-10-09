"""
Daily Summary Printer — Sentinel Judge pillar.

Reads the latest post-mortem JSON output and prints a concise,
human-readable console summary of Sentinel's performance metrics,
prediction accuracy, and anomalies flagged by the resolver.

Integrates with postmortem.py output and serves as the human-facing
daily digest for monitoring system health and decision quality.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortem") -> Optional[dict]:
    """Load the most recent post-mortem JSON file from disk."""
    postmortem_path = Path(postmortem_dir)
    if not postmortem_path.exists():
        return None
    
    json_files = sorted(postmortem_path.glob("postmortem_*.json"), reverse=True)
    if not json_files:
        return None
    
    try:
        with open(json_files[0], "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def print_summary_header(postmortem: dict) -> None:
    """Print the header section with date and overall metrics."""
    run_date = postmortem.get("run_date", "unknown")
    print("\n" + "="*70)
    print(f"  SENTINEL DAILY SUMMARY — {run_date}")
    print("="*70 + "\n")

def print_performance_stats(postmortem: dict) -> None:
    """Print accuracy, hit rate, and confidence metrics."""
    stats = postmortem.get("stats", {})
    
    total_predictions = stats.get("total_predictions", 0)
    correct_predictions = stats.get("correct_predictions", 0)
    accuracy = stats.get("accuracy", 0.0)
    
    print("📊 PERFORMANCE METRICS")
    print(f"  Total Predictions:    {total_predictions}")
    print(f"  Correct Predictions:  {correct_predictions}")
    print(f"  Accuracy:             {accuracy:.1%}")
    
    mean_confidence = stats.get("mean_confidence", 0.0)
    print(f"  Mean Confidence:      {mean_confidence:.2f}")
    
    avg_error_pct = stats.get("avg_absolute_error_pct", 0.0)
    print(f"  Avg Absolute Error:   {avg_error_pct:.2f}%\n")

def print_predictions(postmortem: dict) -> None:
    """Print individual prediction outcomes."""
    predictions = postmortem.get("predictions", [])
    
    if not predictions:
        print("⚠️  NO PREDICTIONS RECORDED\n")
        return
    
    print("🎯 PREDICTION OUTCOMES")
    for pred in predictions:
        ticker = pred.get("ticker", "???")
        predicted_direction = pred.get("predicted_direction", "?")
        actual_direction = pred.get("actual_direction", "?")
        confidence = pred.get("confidence", 0.0)
        error_pct = pred.get("absolute_error_pct", 0.0)
        
        match_icon = "✓" if predicted_direction == actual_direction else "✗"
        print(f"  {match_icon} {ticker}: pred={predicted_direction}, actual={actual_direction} "
              f"| conf={confidence:.2f} | err={error_pct:.2f}%")
    print()

def print_anomalies(postmortem: dict) -> None:
    """Print any anomalies flagged by the resolver."""
    anomalies = postmortem.get("anomalies", [])
    
    if not anomalies:
        print("✅ NO ANOMALIES DETECTED\n")
        return
    
    print("⚠️  ANOMALIES FLAGGED")
    for anomaly in anomalies:
        ticker = anomaly.get("ticker", "???")
        reason = anomaly.get("reason", "unknown")
        severity = anomaly.get("severity", "low")
        print(f"  [{severity.upper()}] {ticker}: {reason}")
    print()

def print_strategy_comparison(postmortem: dict) -> None:
    """Print baseline strategy vs. Sentinel prediction comparison."""
    baselines = postmortem.get("baseline_comparison", {})
    
    if not baselines:
        print("(No baseline comparison data)\n")
        return
    
    print("📈 BASELINE COMPARISON")
    sentinel_acc = baselines.get("sentinel_accuracy", 0.0)
    buy_hold_acc = baselines.get("buy_hold_accuracy", 0.0)
    mean_revert_acc = baselines.get("mean_reversion_accuracy", 0.0)
    
    print(f"  Sentinel:        {sentinel_acc:.1%}")
    print(f"  Buy & Hold:      {buy_hold_acc:.1%}")
    print(f"  Mean Reversion:  {mean_revert_acc:.1%}\n")

def print_footer() -> None:
    """Print closing remarks and next-run guidance."""
    print("="*70)
    print("  Run `sentinel/pipeline.py` to refresh predictions.")
    print("="*70 + "\n")

def print_daily_summary(postmortem_dir: str = "sentinel/judge/postmortem") -> int:
    """Load and print a concise daily summary of Sentinel performance."""
    postmortem = load_latest_postmortem(postmortem_dir)
    
    if postmortem is None:
        print("❌ No post-mortem data found. Run the pipeline first.")
        return 1
    
    print_summary_header(postmortem)
    print_performance_stats(postmortem)
    print_predictions(postmortem)
    print_anomalies(postmortem)
    print_strategy_comparison(postmortem)
    print_footer()
    
    return 0

if __name__ == "__main__":
    postmortem_dir = sys.argv[1] if len(sys.argv) > 1 else "sentinel/judge/postmortem"
    sys.exit(print_daily_summary(postmortem_dir))
