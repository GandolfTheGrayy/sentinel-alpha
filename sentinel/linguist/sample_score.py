"""Linguist: certainty/hesitation scoring of corporate text via Claude."""
from __future__ import annotations

import json
import re

import anthropic

PROMPT = """Score the linguistic certainty of the text below on a 0-100 integer scale.
0 = heavily hedged ("may", "could", "subject to"). 100 = fully confident, declarative.
Return ONLY a JSON object with keys "score" (int) and "reasoning" (one sentence).

Text:
\"\"\"{text}\"\"\""""


def score_text(text: str, client: anthropic.Anthropic | None = None, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Return {'score': int, 'reasoning': str} for the given text."""
    client = client or anthropic.Anthropic()
    r = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(text=text[:2000])}],
    )
    raw = r.content[0].text
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {"score": -1, "reasoning": f"parse failed: {raw[:120]}"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"score": -1, "reasoning": f"json error: {exc}"}
