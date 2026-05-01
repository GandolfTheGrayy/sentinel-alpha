"""Judge: ask Claude for a directional/magnitude prediction on a ticker."""
from __future__ import annotations

import json
import re

import anthropic

PROMPT = """Predict {ticker}'s 5-trading-day price movement.

Recent 5d move: {recent_pct:+.2f}%
Latest headline: "{headline}" ({publisher})

Return ONLY a JSON object:
{{"direction": "up"|"down"|"neutral", "magnitude_pct": <float, absolute % expected>, "confidence": <int 0-100>, "rationale": "<one sentence>"}}"""


def predict(
    ticker: str,
    recent_pct: float,
    headline: str,
    publisher: str = "",
    client: anthropic.Anthropic | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """Return prediction dict {direction, magnitude_pct, confidence, rationale}."""
    client = client or anthropic.Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=220,
        messages=[{
            "role": "user",
            "content": PROMPT.format(
                ticker=ticker,
                recent_pct=recent_pct,
                headline=(headline or "no recent headline")[:300],
                publisher=publisher or "—",
            ),
        }],
    )
    raw = r.content[0].text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"direction": "neutral", "magnitude_pct": 0.0, "confidence": -1, "rationale": "parse failed"}
    try:
        out = json.loads(m.group(0))
        out["direction"] = out.get("direction", "neutral").lower()
        if out["direction"] not in ("up", "down", "neutral"):
            out["direction"] = "neutral"
        return out
    except json.JSONDecodeError:
        return {"direction": "neutral", "magnitude_pct": 0.0, "confidence": -1, "rationale": "json error"}
