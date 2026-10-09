"""
Daily Summary Printer for Sentinel Sentiment Engine.

Reads the latest post-mortem JSON file and prints a concise console summary
of Sentinel's performance, including prediction accuracy, top movers, and
confidence distributions. Designed to be run once per day after post-mortem
generation to give stakeholders a quick read on model health.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


def find_latest_postmortem(postmortem_dir: str = "sentinel/judge/postmortems") -> Optional[str]:
    """Find the most recent post-mortem JSON file in the postmortem directory."""
    postmortem_path = Path(postmortem_dir)
    if not postmortem_path.exists():
        return None
    
    json_files = sorted(postmortem_path.glob("*.json"), reverse=True)
    return str(json_files[0]) if json_files else None


def load_postmortem(filepath: str) -> Dict[str, Any]:
    """Load and parse a post-mortem JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading post-mortem: {e}")
        return {}


def format_accuracy_section(postmortem: Dict[str, Any]) -> str:
    """Format and return the accuracy metrics section."""
    lines = []
    lines.append("=" * 60)
    lines.append("ACCURACY METRICS")
    lines.append("=" * 60)
    
    if "summary" in postmortem:
        summary = postmortem["summary"]
        total = summary.get("total_predictions", 0)
        correct = summary.get("correct_predictions", 0)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        lines.append(f"Total Predictions: {total}")
        lines.append(f"Correct: {correct}")
        lines.append(f"Accuracy: {accuracy:.1f}%")
        lines.append(f"Directional Wins: {summary.get('directional_wins', 0)}")
    
    return "\n".join(lines)


def format_confidence_distribution(postmortem: Dict[str, Any]) -> str:
    """Format and return the confidence score distribution section."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("CONFIDENCE DISTRIBUTION")
    lines.append("=" * 60)
    
    if "confidence_bins" in postmortem:
        bins = postmortem["confidence_bins"]
        for bin_label, count in bins.items():
            lines.append(f"{bin_label}: {count} predictions")
    
    return "\n".join(lines)


def format_top_movers(postmortem: Dict[str, Any]) -> str:
    """Format and return the top performers and worst performers section."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("TOP MOVERS & OUTLIERS")
    lines.append("=" * 60)
    
    if "predictions" in postmortem:
        predictions = postmortem["predictions"]
        
        # Sort by absolute magnitude of prediction error
        sorted_preds = sorted(
            predictions,
            key=lambda p: abs(p.get("error", 0)),
            reverse=True
        )
        
        # Top 3 biggest misses
        lines.append("\nTop 3 Biggest Misses:")
        for i, pred in enumerate(sorted_preds[:3], 1):
            ticker = pred.get("ticker", "UNKNOWN")
            predicted = pred.get("predicted_direction", "?")
            actual = pred.get("actual_direction", "?")
            error = pred.get("error", 0.0)
            confidence = pred.get("confidence", 0.0)
            lines.append(
                f"  {i}. {ticker}: predicted {predicted}, got {actual} "
                f"(error: {error:.2f}%, conf: {confidence:.1f}%)"
            )
        
        # Top 3 highest confidence hits
        high_conf_correct = [
            p for p in predictions
            if p.get("correct", False) and p.get("confidence", 0) >= 70
        ]
        high_conf_correct.sort(key=lambda p: p.get("confidence", 0), reverse=True)
        
        lines.append("\nTop 3 High-Confidence Hits:")
        for i, pred in enumerate(high_conf_correct[:3], 1):
            ticker = pred.get("ticker", "UNKNOWN")
            direction = pred.get("predicted_direction", "?")
            confidence = pred.get("confidence", 0.0)
            lines.append(f"  {i}. {ticker}: {direction} @ {confidence:.1f}% confidence")
    
    return "\n".join(lines)


def format_sector_breakdown(postmortem: Dict[str, Any]) -> str:
    """Format and return the sector-level performance breakdown."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("SECTOR BREAKDOWN")
    lines.append("=" * 60)
    
    if "sector_stats" in postmortem:
        sectors = postmortem["sector_stats"]
        for sector, stats in sorted(sectors.items()):
            accuracy = stats.get("accuracy", 0.0)
            count = stats.get("prediction_count", 0)
            lines.append(f"{sector}: {accuracy:.1f}% ({count} predictions)")
    else:
        lines.append("(No sector data available)")
    
    return "\n".join(lines)


def format_anomalies(postmortem: Dict[str, Any]) -> str:
    """Format and return detected anomalies and edge cases."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("ANOMALIES & ALERTS")
    lines.append("=" * 60)
    
    if "anomalies" in postmortem and postmortem["anomalies"]:
        for anomaly in postmortem["anomalies"]:
            ticker = anomaly.get("ticker", "UNKNOWN")
            alert_type = anomaly.get("type", "UNKNOWN")
            description = anomaly.get("description", "")
            lines.append(f"[{alert_type}] {ticker}: {description}")
    else:
        lines.append("No anomalies detected.")
    
    return "\n".join(lines)


def format_metadata(postmortem: Dict[str, Any]) -> str:
    """Format and return metadata about the post-mortem."""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("RUN METADATA")
    lines.append("=" * 60)
    
    if "metadata" in postmortem:
        meta = postmortem["metadata"]
        lines.append(f"Generated: {meta.get('generated_at', 'N/A')}")
        lines.append(f"Period: {meta.get('period', 'N/A')}")
        lines.append(f"Model: {meta.get('model_version', 'N/A')}")
    
    return "\n".join(lines)


def print_daily_summary(postmortem_filepath: Optional[str] = None) -> None:
    """
    Load the latest post-mortem and print a formatted daily summary to console.
    """
    if not postmortem_filepath:
        postmortem_filepath = find_latest_postmortem()
    
    if not postmortem_filepath:
        print("No post-mortem file found. Run the pipeline first.")
        return
    
    postmortem = load_postmortem(postmortem_filepath)
    if not postmortem:
        return
    
    # Print header
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " SENTINEL SENTIMENT ENGINE — DAILY
