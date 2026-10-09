"""Judge: Claude-driven hold/sell decision for an open position based on current state + news."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import anthropic

MODEL = "claude-haiku-4-5-20251001"

PROMPT = """You manage an open LONG position in a mock portfolio. Decide HOLD or SELL.

Position: {shares:.4f} shares of {ticker} bought at ${entry:.2f}, now ${current:.2f} ({pnl_pct:+.2f}%, held {days_held} days)
Original thesis: {thesis}

Recent news on {ticker}:
{news_block}

Decide based on:
- Has the original thesis played out? If yes, take profit.
- Are there new clearly negative catalysts? If yes, exit.
- Default to HOLD unless something specific changed. Don't churn.
- Consider that exiting too early forfeits upside; exiting too late forfeits gains.

Return ONLY a JSON object:
{{"action": "hold"|"sell", "confidence": <int 0-100>, "reason": "<one sentence>"}}"""


def _days_held(entry_time: str | None) -> int:
    """Compute integer days between entry_time and now."""
    if not entry_time:
        return 0
    try:
        dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - dt
        return max(0, delta.days)
    except Exception:
        return 0


def _format_news(items: list[dict]) -> str:
    """Render recent news items for the prompt."""
    if not items:
        return "(none in last 7 days)"
    return "\n".join(f"- {n.get('headline', '')} ({n.get('source', '?')})" for n in items[:3] if n.get("headline"))


def should_exit(position: dict, current_price: float, recent_news: list[dict], client: anthropic.Anthropic | None = None) -> dict:
    """Return {action, confidence, reason} for an open position."""
    client = client or anthropic.Anthropic()
    entry = float(position.get("entry_price") or 0)
    if not entry:
        return {"action": "hold", "confidence": 0, "reason": "no entry price"}
    pnl_pct = (current_price - entry) / entry * 100.0
    try:
        r = client.messages.create(
            model=MODEL,
            max_tokens=180,
            messages=[{
                "role": "user",
                "content": PROMPT.format(
                    shares=float(position.get("shares") or 0),
                    ticker=position.get("ticker", "?"),
                    entry=entry,
                    current=current_price,
                    pnl_pct=pnl_pct,
                    days_held=_days_held(position.get("entry_time")),
                    thesis=(position.get("thesis") or "(none)")[:300],
                    news_block=_format_news(recent_news),
                ),
            }],
        )
    except Exception as exc:
        return {"action": "hold", "confidence": 0, "reason": f"llm error: {exc}"}
    raw = r.content[0].text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"action": "hold", "confidence": 0, "reason": "parse failed"}
    try:
        out = json.loads(m.group(0))
        if out.get("action") not in ("hold", "sell"):
            out["action"] = "hold"
        return out
    except json.JSONDecodeError:
        return {"action": "hold", "confidence": 0, "reason": "json error"}
