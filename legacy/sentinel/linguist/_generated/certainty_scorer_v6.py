"""
Certainty vs. Hesitation scoring for corporate sentiment analysis.

This module provides a prompt template system that leverages Claude to analyze
corporate text (earnings calls, SEC filings, news) and extract structured
confidence signals. It detects linguistic markers of conviction, uncertainty,
and forward-looking hedges, then returns a CertaintyScore dataclass that feeds
into the Judge's prediction pipeline.

Part of the Linguist pillar: semantic reasoning on raw text before RAG lookup.
"""

from dataclasses import dataclass
from typing import Optional
import os
from anthropic import Anthropic

@dataclass
class CertaintyScore:
    """Structured output of a certainty analysis pass."""
    ticker: str
    text_source: str  # e.g., "earnings_call", "sec_filing", "news_headline"
    overall_certainty: float  # 0.0 (uncertain) to 1.0 (highly certain)
    conviction_markers: list[str]  # e.g., ["strong guidance", "record revenue"]
    hesitation_markers: list[str]  # e.g., ["may", "could", "risk", "headwind"]
    forward_looking_hedges: list[str]  # e.g., "subject to", "assumes favorable conditions"
    tone_shift_detected: bool  # True if language differs from prior filings
    confidence_in_score: float  # meta-confidence: how confident are we in this score
    reasoning: str  # Explanation from Claude


def build_certainty_prompt(
    ticker: str,
    text: str,
    source: str,
    historical_context: Optional[str] = None,
) -> str:
    """
    Construct a Claude prompt for certainty scoring of corporate text.

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        text: The corporate text to analyze
        source: Source type ("earnings_call", "sec_filing", "news_headline", etc.)
        historical_context: Prior filing/statement for tone-shift detection

    Returns:
        A multi-shot prompt ready to send to Claude.
    """
    prompt = f"""You are a financial sentiment analyst. Analyze the following {source} excerpt for {ticker} and score its certainty level.

## Instructions
1. Identify conviction markers: words/phrases showing strong confidence (e.g., "record", "exceeded", "accelerating", "confident").
2. Identify hesitation markers: hedging language (e.g., "may", "could", "uncertain", "headwind", "risk", "challenge").
3. Detect forward-looking hedges: disclaimers that limit claims (e.g., "subject to", "assumes", "conditions permitting").
4. Estimate overall certainty as a float from 0.0 (very uncertain) to 1.0 (highly certain).
5. If historical context is provided, flag any significant tone shift.
6. Provide a confidence score (0.0–1.0) in your own scoring — how reliable is this analysis given text length/clarity?

## Text to Analyze
Source: {source}
Ticker: {ticker}

---
{text}
---

## Historical Context (if available)
{historical_context or "(None provided)"}

## Output Format
Respond in JSON format ONLY. No markdown, no extra text.

{{
  "overall_certainty": <float 0.0–1.0>,
  "conviction_markers": [<list of strings>],
  "hesitation_markers": [<list of strings>],
  "forward_looking_hedges": [<list of strings>],
  "tone_shift_detected": <boolean>,
  "confidence_in_score": <float 0.0–1.0>,
  "reasoning": "<brief explanation of the scoring>"
}}"""
    return prompt


def score_certainty(
    ticker: str,
    text: str,
    source: str,
    historical_context: Optional[str] = None,
) -> CertaintyScore:
    """
    Score certainty of corporate text using Claude, return structured CertaintyScore.

    Args:
        ticker: Stock ticker
        text: Corporate text to analyze
        source: Source type
        historical_context: Prior text for tone-shift comparison

    Returns:
        CertaintyScore dataclass with all extracted signals.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = Anthropic()
    prompt = build_certainty_prompt(ticker, text, source, historical_context)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = response.content[0].text
    import json

    parsed = json.loads(response_text)

    return CertaintyScore(
        ticker=ticker,
        text_source=source,
        overall_certainty=float(parsed["overall_certainty"]),
        conviction_markers=parsed["conviction_markers"],
        hesitation_markers=parsed["hesitation_markers"],
        forward_looking_hedges=parsed["forward_looking_hedges"],
        tone_shift_detected=bool(parsed["tone_shift_detected"]),
        confidence_in_score=float(parsed["confidence_in_score"]),
        reasoning=str(parsed["reasoning"]),
    )


def batch_score_certainty(
    ticker: str,
    texts: list[dict],
) -> list[CertaintyScore]:
    """
    Score multiple text samples for the same ticker in sequence.

    Args:
        ticker: Stock ticker
        texts: List of dicts with keys "text", "source", and optional "historical_context"

    Returns:
        List of CertaintyScore objects.
    """
    results = []
    for item in texts:
        score = score_certainty(
            ticker=ticker,
            text=item["text"],
            source=item["source"],
            historical_context=item.get("historical_context"),
        )
        results.append(score)
    return results


if __name__ == "__main__":
    sample_text = """
    We delivered record revenues of $120B in Q4, exceeding guidance by 8%.
    Our cloud business accelerated 42% YoY, driven by strong demand from enterprise customers.
    We remain confident in our ability to sustain double-digit growth, though we acknowledge
    macro headwinds may present challenges in certain geographies.
    """
    score = score_certainty(
        ticker="SAMPLE",
        text=sample_text,
        source="earnings_call",
    )
    print(f"Overall Certainty: {score.overall_certainty}")
    print(f"Conviction Markers: {score.conviction_markers}")
    print(f"Hesitation Markers: {score.hesitation_markers}")
    print(f"Reasoning: {score.reasoning}")
