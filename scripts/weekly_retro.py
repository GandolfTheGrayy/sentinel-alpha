"""Weekly Sunday retrospective: ask Claude to analyze last 7 days of resolved predictions.

Writes backtest_results/weekly/YYYY-WNN.md with patterns, hits, misses, and
hypotheses about what's working / breaking.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import anthropic
import requests

from sentinel.storage import db


def _maybe_email(subject: str, body_md: str) -> None:
    """Send the retrospective via Resend if RESEND_API_KEY + RESEND_TO are set."""
    key = os.environ.get("RESEND_API_KEY")
    to = os.environ.get("RESEND_TO")
    sender = os.environ.get("RESEND_FROM", "Sentinel <onboarding@resend.dev>")
    if not key or not to:
        return
    html = "<pre style='font-family:ui-monospace,monospace;font-size:13px;line-height:1.6;white-space:pre-wrap;'>" + body_md.replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"from": sender, "to": [to], "subject": subject, "html": html},
            timeout=20,
        )
        r.raise_for_status()
        print(f"  emailed retrospective to {to}")
    except Exception as exc:
        print(f"  WARN: email failed ({exc})", file=sys.stderr)

PREDICTIONS_PATH = Path("docs/predictions.json")
WEEKLY_DIR = Path("backtest_results/weekly")
MODEL = "claude-sonnet-4-6"


PROMPT = """You are reviewing the past week of stock-direction predictions made by an
autonomous research system called Sentinel. Look at the data below and write a
concise retrospective that is useful to the engineer building the system.

Cover:
1. Hit rate this week (Claude vs the three baselines)
2. The single best HIT and what made it work
3. The worst MISS and what the system should have caught
4. Any pattern in headlines / filings / tickers that correlate with HITs
5. Any pattern that correlates with MISSes
6. One concrete suggestion for next week's calibration

Write 250-400 words. Use markdown headers. Be direct, no fluff.

Predictions resolved this week:
{resolved_json}

Predictions made this week (some not yet resolved):
{made_json}
"""


def main() -> int:
    """Generate this week's retrospective markdown."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    today = date.today()
    week_ago = today - timedelta(days=7)
    try:
        preds = db.get_recent_predictions(since_date=week_ago.isoformat(), limit=500)
    except Exception as exc:
        print(f"WARN: Supabase read failed ({exc}); falling back to JSON", file=sys.stderr)
        if not PREDICTIONS_PATH.exists():
            return 0
        preds = json.loads(PREDICTIONS_PATH.read_text(encoding="utf-8"))
    # normalize made field for both data sources
    for p in preds:
        p.setdefault("made", str(p.get("made_on") or p.get("made") or ""))
    resolved_week = [p for p in preds if p.get("resolved")]
    made_week = preds
    if not resolved_week and not made_week:
        print("nothing happened this week")
        return 0
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT.format(
            resolved_json=json.dumps(resolved_week, default=str)[:8000],
            made_json=json.dumps([{k: v for k, v in p.items() if k not in ("rationale", "headline")} for p in made_week], default=str)[:4000],
        )}],
    )
    body = msg.content[0].text.strip()
    iso_year, iso_week, _ = today.isocalendar()
    out = WEEKLY_DIR / f"{iso_year}-W{iso_week:02d}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    full = f"# Sentinel Weekly Retrospective — {iso_year}-W{iso_week:02d}\n\nGenerated {today.isoformat()}.\n\n{body}\n"
    out.write_text(full, encoding="utf-8")
    print(f"wrote {out}")
    _maybe_email(f"Sentinel · W{iso_week:02d} retrospective", full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
