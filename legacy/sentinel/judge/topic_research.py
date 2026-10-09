"""Judge: deep multi-source research on a single topic via Gemini grounded search.

Returns a synthesis plus the list of tickers the story most affects. Predictions
are then made per affected ticker — universe is dynamic, driven by today's news.
"""
from __future__ import annotations

import json
import os
import re

MODEL = "gemini-2.5-flash"


def _genai():
    """Lazy-import + configure Gemini SDK."""
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing")
    genai.configure(api_key=key)
    return genai


PROMPT = """Conduct deep research on this developing market story. Use Google Search aggressively — find 5+ articles from DIFFERENT outlets, read them, synthesize a complete picture.

STORY:
Title: {title}
Summary: {summary}

Output ONLY a JSON object, no prose outside it:
{{
  "synthesis": "<2-3 paragraph synthesis pulling from multiple sources>",
  "consensus_view": "<what most outlets/analysts agree on>",
  "contrarian_view": "<what dissenting views or risks exist>",
  "affected_tickers": [
    {{"ticker": "<symbol>", "name": "<company>", "exposure": "high|medium|low", "direction_bias": "up|down|neutral", "rationale": "<one sentence on why this story moves this stock>"}}
  ],
  "sources": [{{"title": "<article title>", "outlet": "<outlet>"}}]
}}

Rules:
- affected_tickers: 2-5 distinct US-listed equities, spanning at least 2 outlets' coverage. Don't only return mega-caps unless they are genuinely central.
- direction_bias must reflect the LIKELY 5-trading-day move, not the multi-year thesis.
- If you can't find solid sources, return affected_tickers: [] rather than guess."""


def deep_research(topic: dict) -> dict:
    """Run grounded research; return synthesis + affected_tickers + sources."""
    g = _genai()
    model = g.GenerativeModel(MODEL, tools=[{"google_search": {}}])
    try:
        r = model.generate_content(
            PROMPT.format(title=topic.get("title", ""), summary=topic.get("summary", "")),
            generation_config={"max_output_tokens": 2000, "temperature": 0.3},
        )
        text = r.text or ""
    except Exception as exc:
        return {"synthesis": f"(research failed: {exc})", "affected_tickers": [], "sources": []}

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"synthesis": text[:600], "affected_tickers": [], "sources": []}
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"synthesis": text[:600], "affected_tickers": [], "sources": []}

    tickers = []
    for t in (out.get("affected_tickers") or []):
        sym = (t.get("ticker") or "").upper().strip()
        if re.fullmatch(r"[A-Z][A-Z.\-]{0,7}", sym):
            tickers.append({**t, "ticker": sym})
    out["affected_tickers"] = tickers[:5]
    out.setdefault("synthesis", "")
    out.setdefault("consensus_view", "")
    out.setdefault("contrarian_view", "")
    out.setdefault("sources", [])
    return out
