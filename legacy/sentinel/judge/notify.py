"""Judge: Discord webhook notifier for high-conviction HITs and big MISSes."""
from __future__ import annotations

import os

import requests

HIT_CONF_THRESHOLD = 60
MISS_CONF_THRESHOLD = 60
MISS_MAGNITUDE_THRESHOLD = 5.0


def _post(content: str) -> None:
    """POST to DISCORD_WEBHOOK_URL if set; silently no-op otherwise."""
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": content[:1900]}, timeout=10)
    except Exception:
        pass


def maybe_alert(pred: dict) -> bool:
    """Send a Discord alert for a resolved Claude prediction if it crosses thresholds."""
    if pred.get("strategy") and pred["strategy"] != "claude":
        return False
    if not pred.get("resolved") or pred.get("correct_direction") is None:
        return False
    conf = pred.get("confidence", -1)
    actual = pred.get("actual_pct", 0.0)
    if pred["correct_direction"] and conf >= HIT_CONF_THRESHOLD:
        _post(f"🎯 **HIT** — {pred['ticker']} predicted **{pred['direction']}** @ conf {conf} → actual {actual:+.2f}% (made {pred['made']})")
        return True
    if not pred["correct_direction"] and conf >= MISS_CONF_THRESHOLD and abs(actual) >= MISS_MAGNITUDE_THRESHOLD:
        _post(f"❌ **BIG MISS** — {pred['ticker']} predicted **{pred['direction']}** @ conf {conf} → actual {actual:+.2f}% (made {pred['made']})")
        return True
    return False
