"""End-to-end Sentinel run: predict (Claude + baselines), resolve, notify, persist.

Per-ticker isolation: any single ticker's failure cannot abort the run.
Outputs:
    docs/predictions.json — full prediction ledger across all strategies
    docs/data.json        — today's snapshot (consumed by dashboard)
    docs/history.json     — daily KPI rollup for trend chart
    backtest_results/YYYY-MM-DD.md — markdown post-mortem
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

import anthropic

from sentinel.historian.rag_query import query
from sentinel.judge import baselines
from sentinel.judge.notify import maybe_alert
from sentinel.judge.postmortem import render
from sentinel.judge.predictor import predict
from sentinel.judge.resolver import resolve
from sentinel.linguist.sample_score import score_text
from sentinel.scout.live_prices import WATCHLIST, fetch_summary
from sentinel.scout.news import latest_headline
from sentinel.scout.sec_filings import latest_filing

PREDICTIONS_PATH = Path("docs/predictions.json")
DATA_PATH = Path("docs/data.json")
HISTORY_PATH = Path("docs/history.json")
HORIZON_DAYS = 5
ALL_STRATEGIES = ["claude", *baselines.STRATEGIES]

TICKER_NAMES: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "NVDA": ["nvidia", "nvda"],
    "GOOGL": ["google", "alphabet", "googl"],
    "META": ["meta platforms", "facebook", " meta "],
    "TSLA": ["tesla", "tsla"],
    "AMZN": ["amazon", "amzn"],
}


def _headline_belongs_to(headline: str, ticker: str) -> bool:
    """Return True if the headline mentions this ticker's company name."""
    h = f" {headline.lower()} "
    return any(name in h for name in TICKER_NAMES.get(ticker, [ticker.lower()]))


def _load(p: Path, default):
    """Load JSON or return default on missing/corrupt."""
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


def _strategy_stats(preds: list[dict], strategy: str) -> dict:
    """Compute hit-rate stats for a single strategy."""
    closed = [p for p in preds if p.get("strategy") == strategy and p.get("resolved") and p.get("correct_direction") is not None]
    n = len(closed)
    hits = sum(1 for p in closed if p["correct_direction"])
    last_30 = closed[-30:]
    last_7 = closed[-7:]
    return {
        "total_resolved": n,
        "hits": hits,
        "hit_rate": round(hits / n * 100, 1) if n else None,
        "rolling_30": round(sum(1 for p in last_30 if p["correct_direction"]) / len(last_30) * 100, 1) if last_30 else None,
        "rolling_7": round(sum(1 for p in last_7 if p["correct_direction"]) / len(last_7) * 100, 1) if last_7 else None,
    }


def _make_record(strategy: str, ticker: str, pred: dict, today_iso: str, pr: dict, headline: str, publisher: str, filing: dict | None, today: date) -> dict:
    """Wrap a prediction dict + context into a stored record."""
    return {
        "id": f"{today_iso}-{ticker}-{strategy}",
        "made": today_iso,
        "ticker": ticker,
        "strategy": strategy,
        "direction": pred.get("direction", "neutral"),
        "magnitude_pct": float(pred.get("magnitude_pct", 0.0)),
        "confidence": int(pred.get("confidence", -1)),
        "rationale": pred.get("rationale", ""),
        "headline": headline,
        "publisher": publisher,
        "filing": {"form": filing["form"], "filed": filing["filed"], "url": filing["url"]} if filing else None,
        "horizon_days": HORIZON_DAYS,
        "made_pct_5d_prior": pr["pct_change"],
        "price_at_prediction": pr["close"],
        "resolves_on": (today + timedelta(days=HORIZON_DAYS + 2)).isoformat(),
        "resolved": False,
    }


def run() -> dict:
    """Execute pipeline: per-ticker isolation, all strategies, persist + notify."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    today = date.today()
    today_iso = today.isoformat()
    client = anthropic.Anthropic()

    prices = fetch_summary()
    price_by = {p["ticker"]: p for p in prices}

    predictions: list[dict] = _load(PREDICTIONS_PATH, [])
    for _p in predictions:
        _p.setdefault("strategy", "claude")

    # ---- resolve due predictions ----
    resolved_today: list[dict] = []
    alerts_sent = 0
    for pr in predictions:
        if pr.get("resolved"):
            continue
        if today_iso < pr.get("resolves_on", "9999-12-31"):
            continue
        try:
            resolve(pr)
            if pr.get("resolved") and "actual_direction" in pr:
                resolved_today.append(pr)
                if maybe_alert(pr):
                    alerts_sent += 1
        except Exception:
            traceback.print_exc()

    # ---- make new predictions for today (per-ticker isolated) ----
    new_today: list[dict] = []
    already = {(p["ticker"], p.get("strategy", "claude")) for p in predictions if p.get("made") == today_iso}
    seen_headlines: dict[str, str] = {}
    for t in WATCHLIST:
        try:
            pr = price_by.get(t)
            if not pr or "error" in pr:
                print(f"  {t}: no price data, skipping", file=sys.stderr)
                continue
            nh = latest_headline(t) or {"title": "", "publisher": ""}
            title = nh.get("title", "")
            if title:
                if title in seen_headlines and not _headline_belongs_to(title, t):
                    print(f"  {t}: dropping shared headline (already used by {seen_headlines[title]}, no name match)")
                    nh, title = {"title": "", "publisher": ""}, ""
                elif title not in seen_headlines:
                    seen_headlines[title] = t
            filing = latest_filing(t)
            if filing:
                print(f"  {t}: SEC {filing['form']} filed {filing['filed']} ({len(filing.get('text',''))} chars)")
            else:
                print(f"  {t}: no recent SEC filing")
            ctx_text = title
            if filing and filing.get("text"):
                ctx_text = (ctx_text + " | " + filing["text"][:1500]).strip(" |")
            publisher = nh.get("publisher", "") or (filing["form"] + " filing" if filing else "")

            for strategy in ALL_STRATEGIES:
                if (t, strategy) in already:
                    continue
                try:
                    if strategy == "claude":
                        if not ctx_text:
                            continue
                        pred = predict(t, pr["pct_change"], ctx_text, publisher, client=client)
                    else:
                        pred = baselines.predict(strategy, t, pr["pct_change"])
                    rec = _make_record(strategy, t, pred, today_iso, pr, nh.get("title", ""), nh.get("publisher", ""), filing, today)
                    predictions.append(rec)
                    new_today.append(rec)
                except Exception:
                    print(f"  prediction failed: {t}/{strategy}", file=sys.stderr)
                    traceback.print_exc()
        except Exception:
            print(f"  ticker failed: {t}", file=sys.stderr)
            traceback.print_exc()

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    # ---- accuracy stats per strategy ----
    accuracy = {s: _strategy_stats(predictions, s) for s in ALL_STRATEGIES}

    # ---- linguist + historian on a sample headline (kept for trend continuity) ----
    sample_headline = next((p["headline"] for p in new_today if p.get("strategy") == "claude" and p.get("headline")), "Markets remain mixed")
    try:
        score = score_text(sample_headline, client=client)
    except Exception:
        score = {"score": -1, "reasoning": "linguist failed"}
    try:
        matches = query(sample_headline)
    except Exception:
        matches = []
    report = render(today, prices, score, matches, sample_headline)

    claude_today = [p for p in new_today if p.get("strategy") == "claude"]
    summary = {
        "date": today_iso,
        "headline": sample_headline,
        "watchlist": prices,
        "linguist": score,
        "historian": matches,
        "report": str(report).replace("\\", "/"),
        "predictions_today": claude_today,
        "resolved_today": [p for p in resolved_today if p.get("strategy") == "claude"],
        "open_predictions": [p for p in predictions if not p.get("resolved")],
        "accuracy": accuracy,
        "alerts_sent": alerts_sent,
    }
    DATA_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    history = _load(HISTORY_PATH, [])
    history = [h for h in history if h.get("date") != today_iso]
    valid_pcts = [p["pct_change"] for p in prices if "pct_change" in p]
    history.append({
        "date": today_iso,
        "linguist_score": score.get("score", -1),
        "watchlist_avg_pct": round(sum(valid_pcts) / len(valid_pcts), 2) if valid_pcts else 0,
        "predictions_made": len(claude_today),
        "predictions_resolved": len([p for p in resolved_today if p.get("strategy") == "claude"]),
        "hit_rate": accuracy["claude"]["hit_rate"],
        "rolling_7": accuracy["claude"]["rolling_7"],
        "baseline_momentum_rolling_7": accuracy["momentum"]["rolling_7"],
        "baseline_always_up_rolling_7": accuracy["always_up"]["rolling_7"],
    })
    HISTORY_PATH.write_text(json.dumps(history[-90:], indent=2), encoding="utf-8")

    print(f"Pipeline — {len(claude_today)} Claude predictions, {len(resolved_today)} resolved, alerts: {alerts_sent}")
    print(f"  hit-rates: " + " · ".join(f"{s}={accuracy[s]['hit_rate']}%" for s in ALL_STRATEGIES))
    return summary


if __name__ == "__main__":
    run()
