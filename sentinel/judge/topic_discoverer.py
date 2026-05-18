"""Judge: identify the day's distinct investment-relevant stories from broad market news."""
from __future__ import annotations

import json
import re

import anthropic

MODEL = "claude-sonnet-4-6"

PROMPT = """You are an investment forecaster reviewing today's market news. Identify the {n} most consequential DISTINCT stories — each one a separate developing thesis that could move stocks. Group related headlines under one story.

Output ONLY a JSON array of {n} objects, no prose, no markdown fences:
[
  {{
    "topic_id": "<slug-with-dashes>",
    "title": "<8-12 word headline-style title>",
    "summary": "<1-2 sentence neutral summary of what's happening>",
    "why_it_matters": "<one sentence on market impact>"
  }}
]

Rules:
- Pick stories that affect tradeable US equities, not just commentary.
- Spread across sectors — don't return 4 stories that are all about the same mega-cap.
- Prefer specific developing events (earnings, regulatory rulings, M&A, FDA, geopolitical) over generic market commentary.
- Skip pure crypto, pure macro-without-tickers, or pure index-level chatter.

HEADLINES:
{headlines}"""


def _format_headlines(items: list[dict]) -> str:
    """Render headlines as a numbered list for the prompt."""
    lines = []
    for i, h in enumerate(items, 1):
        when = h.get("when", "")[:10]
        lines.append(f"{i}. [{when}] {h['headline']} ({h.get('source', '?')})")
        if h.get("summary"):
            lines.append(f"   → {h['summary'][:200]}")
    return "\n".join(lines)


def identify_topics(headlines: list[dict], client: anthropic.Anthropic | None = None, n: int = 4) -> list[dict]:
    """Cluster headlines into N distinct investment-relevant stories of the day."""
    if not headlines:
        return []
    client = client or anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT.format(n=n, headlines=_format_headlines(headlines))}],
    )
    raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
        return [t for t in out if t.get("topic_id") and t.get("title")][:n]
    except json.JSONDecodeError:
        return []
