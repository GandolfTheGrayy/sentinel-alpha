"""End-to-end Sentinel run: Scout -> Linguist -> Historian -> Judge.

Writes backtest_results/YYYY-MM-DD.md and docs/data.json (consumed by dashboard).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import anthropic

from sentinel.historian.rag_query import query
from sentinel.judge.postmortem import render
from sentinel.linguist.sample_score import score_text
from sentinel.scout.live_prices import fetch_summary

HEADLINE = (
    "Apple's quarterly outlook may be subject to ongoing supply chain headwinds, "
    "though leadership remains cautiously optimistic about the upcoming product cycle."
)


def run(headline: str = HEADLINE) -> dict:
    """Execute the full pipeline and persist artifacts. Return the run summary."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    today = date.today()
    client = anthropic.Anthropic()
    prices = fetch_summary()
    score = score_text(headline, client=client)
    matches = query(headline)
    report = render(today, prices, score, matches, headline)

    summary = {
        "date": today.isoformat(),
        "headline": headline,
        "watchlist": prices,
        "linguist": score,
        "historian": matches,
        "report": str(report).replace("\\", "/"),
    }
    Path("docs/data.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    history_path = Path("docs/history.json")
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history = [h for h in history if h.get("date") != today.isoformat()]
    history.append({"date": today.isoformat(), "linguist_score": score.get("score", -1), "n_matches": len(matches), "watchlist_avg_pct": round(sum(p.get("pct_change", 0) for p in prices if "pct_change" in p) / max(1, sum(1 for p in prices if "pct_change" in p)), 2)})
    history = history[-90:]
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Pipeline complete — {report}")
    return summary


if __name__ == "__main__":
    run()
