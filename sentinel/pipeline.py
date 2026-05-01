"""End-to-end Sentinel run: predict, resolve due predictions, write artifacts.

Outputs:
    docs/predictions.json — full prediction ledger (append/update)
    docs/data.json        — today's snapshot (consumed by dashboard)
    docs/history.json     — daily KPI rollup for trend chart
    backtest_results/YYYY-MM-DD.md — markdown post-mortem
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic

from sentinel.historian.rag_query import query
from sentinel.judge.postmortem import render
from sentinel.judge.predictor import predict
from sentinel.judge.resolver import resolve
from sentinel.linguist.sample_score import score_text
from sentinel.scout.live_prices import WATCHLIST, fetch_summary
from sentinel.scout.news import latest_headline

PREDICTIONS_PATH = Path("docs/predictions.json")
DATA_PATH = Path("docs/data.json")
HISTORY_PATH = Path("docs/history.json")
HORIZON_DAYS = 5


def _load(p: Path, default):
    """Load JSON from path or return default on missing/corrupt."""
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


def run() -> dict:
    """Predict for today's watchlist, resolve due predictions, persist artifacts."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    today = date.today()
    client = anthropic.Anthropic()

    prices = fetch_summary()
    price_by = {p["ticker"]: p for p in prices}

    predictions: list[dict] = _load(PREDICTIONS_PATH, [])

    resolved_today: list[dict] = []
    for pr in predictions:
        if pr.get("resolved"):
            continue
        if today.isoformat() >= pr.get("resolves_on", "9999-12-31"):
            resolve(pr)
            if pr.get("resolved") and "actual_direction" in pr:
                resolved_today.append(pr)

    new_today: list[dict] = []
    today_iso = today.isoformat()
    already_today = {p["ticker"] for p in predictions if p.get("made") == today_iso}
    for t in WATCHLIST:
        if t in already_today:
            continue
        pr = price_by.get(t)
        if not pr or "error" in pr:
            continue
        nh = latest_headline(t) or {"title": "", "publisher": ""}
        if not nh.get("title"):
            continue
        out = predict(t, pr["pct_change"], nh["title"], nh.get("publisher", ""), client=client)
        rec = {
            "id": f"{today_iso}-{t}",
            "made": today_iso,
            "ticker": t,
            "direction": out.get("direction", "neutral"),
            "magnitude_pct": out.get("magnitude_pct", 0.0),
            "confidence": out.get("confidence", -1),
            "rationale": out.get("rationale", ""),
            "headline": nh["title"],
            "publisher": nh.get("publisher", ""),
            "horizon_days": HORIZON_DAYS,
            "made_pct_5d_prior": pr["pct_change"],
            "price_at_prediction": pr["close"],
            "resolves_on": (today + timedelta(days=HORIZON_DAYS + 2)).isoformat(),
            "resolved": False,
        }
        predictions.append(rec)
        new_today.append(rec)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    closed = [p for p in predictions if p.get("resolved") and p.get("correct_direction") is not None]
    n = len(closed)
    hits = sum(1 for p in closed if p["correct_direction"])
    hit_rate = round(hits / n * 100, 1) if n else None
    last_30 = closed[-30:]
    rolling_30 = round(sum(1 for p in last_30 if p["correct_direction"]) / len(last_30) * 100, 1) if last_30 else None
    last_7 = closed[-7:]
    rolling_7 = round(sum(1 for p in last_7 if p["correct_direction"]) / len(last_7) * 100, 1) if last_7 else None

    sample_headline = new_today[0]["headline"] if new_today else "Markets remain mixed amid earnings season"
    score = score_text(sample_headline, client=client)
    matches = query(sample_headline)
    report = render(today, prices, score, matches, sample_headline)

    summary = {
        "date": today_iso,
        "headline": sample_headline,
        "watchlist": prices,
        "linguist": score,
        "historian": matches,
        "report": str(report).replace("\\", "/"),
        "predictions_today": new_today,
        "resolved_today": resolved_today,
        "open_predictions": [p for p in predictions if not p.get("resolved")],
        "accuracy": {
            "total_resolved": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "rolling_7": rolling_7,
            "rolling_30": rolling_30,
        },
    }
    DATA_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    history = _load(HISTORY_PATH, [])
    history = [h for h in history if h.get("date") != today_iso]
    valid_pcts = [p["pct_change"] for p in prices if "pct_change" in p]
    history.append({
        "date": today_iso,
        "linguist_score": score.get("score", -1),
        "watchlist_avg_pct": round(sum(valid_pcts) / len(valid_pcts), 2) if valid_pcts else 0,
        "predictions_made": len(new_today),
        "predictions_resolved": len(resolved_today),
        "hit_rate": hit_rate,
        "rolling_7": rolling_7,
    })
    HISTORY_PATH.write_text(json.dumps(history[-90:], indent=2), encoding="utf-8")

    print(f"Pipeline — {len(new_today)} predicted, {len(resolved_today)} resolved, hit-rate {hit_rate}%")
    return summary


if __name__ == "__main__":
    run()
