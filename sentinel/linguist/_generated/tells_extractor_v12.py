"""
Sentinel Linguist — Tells Extractor

Identifies specific linguistic tells in corporate text (earnings calls, SEC filings,
press releases) that historically precede stock price movements. Uses Claude to
perform nuanced pattern matching against a curated library of tell signatures.

This module feeds into the Judge's prediction pipeline by enriching sentiment
signals with tell-based confidence adjustments.
"""

import os
from typing import TypedDict
import anthropic


class TellSignature(TypedDict):
    """A linguistic tell pattern and its historical directional bias."""
    name: str
    description: str
    historical_bias: str  # "bullish", "bearish", or "neutral"
    confidence_weight: float


# Curated library of tells observed in historical market data
TELL_SIGNATURES: list[TellSignature] = [
    {
        "name": "guidance_withdrawal",
        "description": "Executive explicitly withdraws or suspends forward guidance",
        "historical_bias": "bearish",
        "confidence_weight": 0.85,
    },
    {
        "name": "conservative_language_shift",
        "description": "Sudden increase in hedge words ('may', 'could', 'uncertain') vs prior tone",
        "historical_bias": "bearish",
        "confidence_weight": 0.72,
    },
    {
        "name": "margin_defense",
        "description": "Repeated emphasis on 'maintaining margins' or 'cost discipline' under pressure",
        "historical_bias": "bearish",
        "confidence_weight": 0.68,
    },
    {
        "name": "cash_position_mention",
        "description": "Unusual emphasis on cash reserves, liquidity, or balance sheet strength",
        "historical_bias": "bearish",
        "confidence_weight": 0.70,
    },
    {
        "name": "channel_inventory_reset",
        "description": "Mention of customer inventory normalization or channel corrections",
        "historical_bias": "bearish",
        "confidence_weight": 0.75,
    },
    {
        "name": "aggressive_capex_expansion",
        "description": "Confident capex guidance increase tied to market opportunity",
        "historical_bias": "bullish",
        "confidence_weight": 0.73,
    },
    {
        "name": "market_share_gains",
        "description": "Specific quantified claims of market share gains in key segments",
        "historical_bias": "bullish",
        "confidence_weight": 0.78,
    },
    {
        "name": "pricing_power_assertion",
        "description": "Explicit statement of successful price increases or premium positioning",
        "historical_bias": "bullish",
        "confidence_weight": 0.76,
    },
    {
        "name": "product_momentum_language",
        "description": "Accelerating adoption language ('ramping', 'inflection', 'breakthrough')",
        "historical_bias": "bullish",
        "confidence_weight": 0.71,
    },
    {
        "name": "management_turnover_hint",
        "description": "Subtle language suggesting leadership transition, strategy pivot, or restructuring",
        "historical_bias": "bearish",
        "confidence_weight": 0.65,
    },
]


def extract_tells(
    corporate_text: str,
    company_name: str = "Unknown",
    context_label: str = "earnings call",
) -> dict:
    """
    Identify linguistic tells in corporate text using Claude reasoning.

    Args:
        corporate_text: Raw text from earnings call, 10-Q, press release, etc.
        company_name: Name of the company for context.
        context_label: Source type ('earnings_call', '10-q', '8-k', etc.)

    Returns:
        Dictionary with detected tells, their confidence scores, and composite bias signal.
    """
    if not corporate_text or len(corporate_text.strip()) < 50:
        return {
            "detected_tells": [],
            "bullish_tells": [],
            "bearish_tells": [],
            "composite_bias": "neutral",
            "composite_confidence": 0.0,
            "reasoning": "Text too short or empty for analysis.",
        }

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    tell_specs = "\n".join(
        [
            f"  - {s['name']}: {s['description']} (bias: {s['historical_bias']}, weight: {s['confidence_weight']})"
            for s in TELL_SIGNATURES
        ]
    )

    prompt = f"""You are a financial linguist analyzing corporate communications for tells that precede stock price moves.

Company: {company_name}
Context: {context_label}

Known Tell Signatures (from historical backtesting):
{tell_specs}

Text to analyze:
---
{corporate_text[:8000]}
---

Task:
1. Scan the text for exact or near-exact matches to the tell signatures above.
2. For each detected tell, provide: (a) the tell name, (b) the exact quote or paraphrase, (c) your confidence (0.0–1.0) that this is a genuine signal.
3. Aggregate the signals: compute a composite bias (bullish/bearish/neutral) and confidence.
4. Flag any contradictions or hedging within the same statement.

Respond in this exact JSON format:
{{
  "detected_tells": [
    {{"name": "tell_name", "quote": "...", "confidence": 0.85}},
    ...
  ],
  "composite_bias": "bullish|bearish|neutral",
  "composite_confidence": 0.72,
  "key_contradictions": ["if any..."],
  "reasoning": "Brief explanation of bias derivation."
}}

Be conservative: only include tells you are >60% confident about. If no tells detected, return empty "detected_tells" array."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    import json

    try:
        result_text = response.content[0].text
        # Extract JSON from response (may be wrapped in markdown)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result = json.loads(result_text)
    except (json.JSONDecodeError, IndexError, AttributeError):
        result = {
            "detected_tells": [],
            "composite_bias": "neutral",
            "composite_confidence": 0.0,
            "reasoning": "Failed to parse Claude response.",
        }

    # Enrich with signature metadata
    bullish_tells = []
    bearish_tells = []

    for tell in result.get("detected_tells", []):
        sig = next(
            (s for s in TELL_SIGNATURES if s["name"] == tell["name"]), None
        )
        if sig:
            tell["historical_bias"] = sig["historical_bias"]
            tell["signature_weight"] = sig["confidence_weight"]
            if sig["historical_bias"] == "bullish":
                bullish_tells.append(tell)
            elif sig["historical_bias"] == "bearish":
                bearish_tells.append(tell)

    result["bullish_tells"] = bullish_tells
    result["bearish_tells"] = bearish_tells

    return result


def score_tells(tells_result: dict) -> float:
    """
    Convert tells extraction result into a normalized sentiment score (-1.0 to +1.0).

    Args:
        tells_result: Output from extract_tells().

    Returns:
        Float score: positive = bullish bias, negative = bearish bias.
    """
    bullish_tells = tells_result.get("bullish_tells", [])
    bearish_tells = tells_result.get("bearish_tells", [])

    bullish_score = sum(
        t.get("confidence", 0.5) * t.get
