"""
Linguistic Tells Extractor — identifies specific linguistic patterns in corporate
text that historically precede price movements.

Integrates with Sentinel's Linguist pillar to detect hedging language, certainty
shifts, regulatory euphemisms, and other tells that correlate with market moves.
Uses Claude for nuanced pattern recognition on a per-company basis.
"""

import os
from typing import TypedDict
import anthropic


class LinguisticTell(TypedDict):
    """Represents a single detected linguistic tell with metadata."""
    tell_type: str
    phrase: str
    severity: str
    confidence: float
    explanation: str


class TellsExtractionResult(TypedDict):
    """Result of tells extraction from a text block."""
    tickers: list[str]
    tells: list[LinguisticTell]
    overall_signal: str
    reasoning: str


def extract_tells(
    text: str,
    ticker: str = "",
    company_name: str = "",
) -> TellsExtractionResult:
    """
    Extract linguistic tells from corporate text using Claude reasoning.

    Args:
        text: Block of corporate text (earnings call, filing, press release, etc.)
        ticker: Stock ticker symbol (optional, for context)
        company_name: Company name (optional, for context)

    Returns:
        TellsExtractionResult containing detected tells, severity, and signal direction.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context_line = ""
    if ticker or company_name:
        context_line = f"[Context: {company_name or ticker}]"

    system_prompt = """You are a linguistic analyst specializing in corporate communication patterns.
Your task is to identify specific linguistic "tells" — phrases, hedging patterns, tone shifts, 
and regulatory euphemisms — that historically correlate with stock price movements.

Focus on these tell categories:
1. HEDGING: "may," "might," "could," "uncertain," "headwinds," "challenging environment"
2. CERTAINTY SHIFT: Reduced confidence vs. prior communications, walking back prior claims
3. REGULATORY WHISPERS: Euphemisms for legal/compliance risk ("we are cooperating," "under review")
4. GUIDANCE RETREAT: Lowered outlooks, removed forward guidance, "we are taking a prudent approach"
5. CASH BURN SIGNALS: "liquidity," "covenant," "working capital," "restructuring"
6. COMPETITIVE EROSION: "pricing pressure," "market share," "intensified competition"
7. POSITIVE TELLS: "accelerating," "record," "outperform," "strategic momentum" (rare but noted)

For each tell detected:
- Quote the exact phrase or paraphrase the pattern
- Assign severity: LOW, MEDIUM, HIGH, CRITICAL
- Confidence 0.0–1.0 that this pattern appeared in the text
- Brief explanation of why this tell matters

Return JSON with structure:
{
  "tickers": ["TICKER"],
  "tells": [
    {
      "tell_type": "HEDGING|CERTAINTY_SHIFT|REGULATORY_WHISPERS|GUIDANCE_RETREAT|CASH_BURN_SIGNALS|COMPETITIVE_EROSION|POSITIVE_TELLS",
      "phrase": "exact quote or paraphrase",
      "severity": "LOW|MEDIUM|HIGH|CRITICAL",
      "confidence": 0.85,
      "explanation": "Why this matters for price prediction"
    }
  ],
  "overall_signal": "BEARISH|NEUTRAL|BULLISH",
  "reasoning": "Summary of tell pattern direction"
}
"""

    user_prompt = f"""{context_line}

Analyze this corporate text for linguistic tells:

---
{text}
---

Respond with valid JSON only, no markdown or explanation."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    response_text = message.content[0].text

    import json
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        result = {
            "tickers": [ticker] if ticker else [],
            "tells": [],
            "overall_signal": "NEUTRAL",
            "reasoning": "Failed to parse Claude response; defaulting to neutral."
        }

    return result


def score_tells_direction(tells: list[LinguisticTell]) -> dict[str, float]:
    """
    Aggregate tell severity and confidence into directional bias scores.

    Args:
        tells: List of extracted linguistic tells.

    Returns:
        Dict with 'bearish_score' and 'bullish_score' (both 0.0–1.0).
    """
    bearish_types = {
        "HEDGING", "CERTAINTY_SHIFT", "REGULATORY_WHISPERS",
        "GUIDANCE_RETREAT", "CASH_BURN_SIGNALS", "COMPETITIVE_EROSION"
    }
    bullish_types = {"POSITIVE_TELLS"}

    severity_weight = {
        "LOW": 0.25,
        "MEDIUM": 0.5,
        "HIGH": 0.75,
        "CRITICAL": 1.0,
    }

    bearish_score = 0.0
    bullish_score = 0.0

    for tell in tells:
        weight = severity_weight.get(tell["severity"], 0.5)
        confidence = tell["confidence"]
        combined = weight * confidence

        if tell["tell_type"] in bearish_types:
            bearish_score += combined
        elif tell["tell_type"] in bullish_types:
            bullish_score += combined

    bearish_score = min(bearish_score, 1.0)
    bullish_score = min(bullish_score, 1.0)

    return {
        "bearish_score": bearish_score,
        "bullish_score": bullish_score,
    }


if __name__ == "__main__":
    sample_text = """
    Our Q3 results reflect a challenging operating environment. While we remain committed
    to our strategic priorities, we are taking a prudent approach to capital allocation.
    We are currently cooperating with regulatory bodies on ongoing investigations.
    Pricing pressure in key markets has impacted margins, though we believe this is temporary.
    Going forward, we may need to revise certain guidance assumptions, but we remain confident
    in the long-term value creation story.
    """

    result = extract_tells(
        text=sample_text,
        ticker="ACME",
        company_name="Acme Corp"
    )

    import json
    print(json.dumps(result, indent=2))

    scores = score_tells_direction(result.get("tells", []))
    print(f"\nDirectional Scores: {scores}")
