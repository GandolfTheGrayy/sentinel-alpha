"""
Daily summary printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON from the judge pillar and renders
a concise console summary of prediction accuracy, win/loss ratio,
top movers, and anomalies flagged by the resolver.

Integrates with judge/postmortem.py output and provides human-friendly
performance metrics for daily standup and trend tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def load_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[dict[str, Any]]:
    """Load the most recent post-mortem JSON file from disk."""
    postmortem_path = Path(postmortem_dir)
    if not postmortem_path.exists():
        return None
    
    json_files = sorted(postmortem_path.glob("*.json"), reverse=True)
    if not json_files:
        return None
    
    with open(json_files[0], "r") as f:
        return json.load(f)


def compute_accuracy_metrics(postmortem: dict[str, Any]) -> dict[str, float]:
    """Extract accuracy, precision, recall, and F1 from post-mortem."""
    predictions = postmortem.get("predictions", [])
    if not predictions:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    correct = sum(1 for p in predictions if p.get("correct", False))
    accuracy = correct / len(predictions) if predictions else 0.0
    
    tp = sum(1 for p in predictions if p.get("correct") and p.get("predicted_direction") in ["up", "bullish"])
    fp = sum(1 for p in predictions if not p.get("correct") and p.get("predicted_direction") in ["up", "bullish"])
    fn = sum(1 for p in predictions if not p.get("correct") and p.get("predicted_direction") not in ["up", "bullish"])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "accuracy": round(accuracy, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def get_top_performers(postmortem: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    """Extract top n correctly predicted tickers by confidence score."""
    predictions = postmortem.get("predictions", [])
    correct = [p for p in predictions if p.get("correct", False)]
    sorted_correct = sorted(correct, key=lambda x: x.get("confidence", 0), reverse=True)
    return sorted_correct[:n]


def get_top_misses(postmortem: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    """Extract top n incorrectly predicted tickers by confidence score."""
    predictions = postmortem.get("predictions", [])
    incorrect = [p for p in predictions if not p.get("correct", False)]
    sorted_incorrect = sorted(incorrect, key=lambda x: x.get("confidence", 0), reverse=True)
    return sorted_incorrect[:n]


def format_prediction_row(pred: dict[str, Any]) -> str:
    """Format a single prediction row for console output."""
    ticker = pred.get("ticker", "???")
    direction = pred.get("predicted_direction", "?")
    confidence = pred.get("confidence", 0.0)
    actual_move = pred.get("actual_move", "?")
    correct = "✓" if pred.get("correct", False) else "✗"
    
    return f"  {ticker:6} | {direction:8} | {confidence:.1%} | {actual_move:6} | {correct}"


def print_daily_summary(postmortem_dir: str = "sentinel/judge/postmortems") -> None:
    """Load and print formatted daily performance summary to console."""
    postmortem = load_latest_postmortem(postmortem_dir)
    
    if not postmortem:
        print("[Sentinel Daily Summary] No post-mortem found.")
        return
    
    run_date = postmortem.get("run_date", "unknown")
    metrics = compute_accuracy_metrics(postmortem)
    top_hits = get_top_performers(postmortem, n=3)
    top_misses = get_top_misses(postmortem, n=3)
    anomalies = postmortem.get("anomalies", [])
    
    print("\n" + "="*80)
    print(f"[Sentinel Daily Summary] {run_date}")
    print("="*80)
    
    print(f"\nAccuracy Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Recall:    {metrics['recall']:.1%}")
    print(f"  F1:        {metrics['f1']:.1%}")
    
    pred_count = len(postmortem.get("predictions", []))
    print(f"\nPredictions: {pred_count} tickers analyzed")
    
    if top_hits:
        print(f"\nTop Hits ({len(top_hits)}):")
        print(f"  {'Ticker':6} | {'Direction':8} | {'Confidence':11} | {'Actual':6} | {'Result'}")
        print(f"  {'-'*60}")
        for pred in top_hits:
            print(format_prediction_row(pred))
    
    if top_misses:
        print(f"\nTop Misses ({len(top_misses)}):")
        print(f"  {'Ticker':6} | {'Direction':8} | {'Confidence':11} | {'Actual':6} | {'Result'}")
        print(f"  {'-'*60}")
        for pred in top_misses:
            print(format_prediction_row(pred))
    
    if anomalies:
        print(f"\nAnomalies Flagged ({len(anomalies)}):")
        for anomaly in anomalies[:5]:
            anomaly_type = anomaly.get("type", "unknown")
            ticker = anomaly.get("ticker", "???")
            reason = anomaly.get("reason", "")
            print(f"  [{anomaly_type}] {ticker}: {reason}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_daily_summary()
