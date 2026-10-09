"""Generate a short Claude postmortem for a resolved prediction that missed direction."""
from __future__ import annotations

import anthropic

MODEL = "claude-haiku-4-5-20251001"

PROMPT = """A directional stock prediction missed. In 2-3 sentences, explain what the model missed. Be specific — don't say "the market is unpredictable." Identify what evidence was overweighted or what factor was ignored.

Ticker: {ticker}
Predicted: {direction} ({magnitude_pct}%, confidence {confidence})
Actual: {actual_direction} ({actual_pct}%)
Original rationale: {rationale}
Headline at time: {headline}
Filing context: {filing_context}

Plain prose. No bullet points. No fluff."""


def generate(pred: dict, client: anthropic.Anthropic | None = None) -> str:
    """Return a short postmortem string for a resolved-and-missed prediction."""
    client = client or anthropic.Anthropic()
    filing = pred.get("filing") or {}
    filing_str = f"{filing.get('form', '')} {filing.get('filed', '')}" if filing else "(none)"
    msg = client.messages.create(
        model=MODEL,
        max_tokens=220,
        messages=[{
            "role": "user",
            "content": PROMPT.format(
                ticker=pred.get("ticker", "?"),
                direction=pred.get("direction", "?"),
                magnitude_pct=pred.get("magnitude_pct", 0),
                confidence=pred.get("confidence", 0),
                actual_direction=pred.get("actual_direction", "?"),
                actual_pct=pred.get("actual_pct", 0),
                rationale=pred.get("rationale", ""),
                headline=pred.get("headline", "")[:200],
                filing_context=filing_str,
            ),
        }],
    )
    return msg.content[0].text.strip()
