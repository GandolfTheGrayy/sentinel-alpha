"""
Linguistic Tells Extractor — Sentinel/Linguist pillar.

Identifies specific linguistic patterns and "tells" in corporate text (earnings calls,
SEC filings, press releases) that historically correlate with price movements.
Uses Claude for nuanced pattern recognition; outputs structured tells with confidence
scores for downstream Judge reasoning.

This module supports the Linguist pillar's mission to detect hesitation, regulatory
whispers, and tone shifts that precede market moves.
"""

import json
import re
from typing import TypedDict, Optional
from anthropic import Anthropic

# Type definitions for tell extraction results
class LinguisticTell(TypedDict):
    """A single identified linguistic tell with confidence."""
    category: str  # e.g., "hedging", "urgency", "regulatory_caution"
    pattern: str   # the matched text snippet
    confidence: float  # 0.0–1.0
    reasoning: str  # why this matters


class TellsExtractionResult(TypedDict):
    """Complete extraction result for a text block."""
    ticker: str
    source: str  # e.g., "10-K", "earnings_call", "press_release"
    tells: list[LinguisticTell]
    overall_tone: str  # "positive", "neutral", "cautious", "defensive"
    summary: str


def extract_tells(
    text: str,
    ticker: str,
    source: str = "unknown",
    client: Optional[Anthropic] = None
) -> TellsExtractionResult:
    """
    Extract linguistic tells from corporate text using Claude.

    Args:
        text: The corporate text block to analyze (earnings call, 10-K excerpt, etc.)
        ticker: Stock ticker symbol for context.
        source: Source type (e.g., "10-K", "earnings_call", "press_release").
        client: Optional Anthropic client; if None, creates a new one.

    Returns:
        TellsExtractionResult with identified tells, tone classification, and summary.
    """
    if client is None:
        client = Anthropic()

    # Truncate very long texts to avoid token limits
    truncated = text[:8000] if len(text) > 8000 else text

    prompt = f"""You are a financial linguistics expert analyzing corporate communications for Sentinel.
Your task: identify specific linguistic "tells" — patterns of language that historically precede stock price movements.

CONTEXT:
- Ticker: {ticker}
- Source: {source}
- Text excerpt: {truncated}

TELLS TO DETECT (high-confidence patterns):

1. HEDGING: excessive use of "may", "could", "potentially", "subject to", "if conditions permit"
   - Suggests management uncertainty or risk avoidance.

2. URGENCY REVERSAL: sudden shift from confident language to cautious qualifiers
   - "We were on track... but headwinds have emerged"
   - Signals unplanned deterioration.

3. REGULATORY CAUTION: mention of audits, investigations, compliance reviews, FDA/SEC scrutiny
   - "Under review", "compliance matter", "regulatory inquiry"
   - Often precedes negative guidance or restatements.

4. EARNINGS MANIPULATION SIGNALS: emphasis on non-GAAP, one-time items, or unusual accounting changes
   - Excessive focus on adjusted metrics while GAAP declines.

5. GUIDANCE VAGUENESS: refusal to provide ranges, frequent "we'll provide updates later"
   - Suggests management has visibility issues.

6. FATIGUE MARKERS: repeated mentions of "unprecedented challenges", "labor constraints", "supply issues"
   - When fatigue language persists past prior calls, suggests structural problems, not transient.

7. CUSTOMER CONCENTRATION WORRY: new emphasis on customer concentration risk or customer loss
   - May indicate customer defection or concentration risk.

8. BUYBACK/DIVIDEND PAUSE: announcements of suspended or reduced capital returns
   - Signals cash stress or confidence collapse.

9. DEBT COMMENTARY SHIFT: increased focus on covenant concerns, refinancing risks, or debt restructuring
   - Often precedes credit downgrades or liquidity events.

10. FORWARD-LOOKING STATEMENT RETREAT: reduction in forward guidance specificity or historical range

OUTPUT INSTRUCTIONS:
Return a JSON object with this exact structure:
{
  "tells": [
    {
      "category": "<one of the above>",
      "pattern": "<exact quoted text from the input>",
      "confidence": <float 0.0–1.0>,
      "reasoning": "<one sentence: why this matters for price prediction>"
    }
  ],
  "overall_tone": "<positive|neutral|cautious|defensive>",
  "summary": "<2-3 sentences summarizing the key linguistic signals and their aggregate implication>"
}

Be strict: only flag tells you find explicit evidence for. Confidence >= 0.6 is required for inclusion.
Return ONLY valid JSON, no markdown, no preamble."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    response_text = response.content[0].text.strip()

    # Parse JSON response
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback if Claude returns malformed JSON
        parsed = {
            "tells": [],
            "overall_tone": "neutral",
            "summary": "Could not parse linguistic analysis."
        }

    tells: list[LinguisticTell] = []
    for tell_obj in parsed.get("tells", []):
        tells.append(LinguisticTell(
            category=tell_obj.get("category", "unknown"),
            pattern=tell_obj.get("pattern", ""),
            confidence=float(tell_obj.get("confidence", 0.0)),
            reasoning=tell_obj.get("reasoning", "")
        ))

    result = TellsExtractionResult(
        ticker=ticker,
        source=source,
        tells=tells,
        overall_tone=parsed.get("overall_tone", "neutral"),
        summary=parsed.get("summary", "")
    )

    return result


def rank_tells_by_confidence(tells: list[LinguisticTell]) -> list[LinguisticTell]:
    """
    Sort linguistic tells by confidence score (descending).

    Args:
        tells: List of LinguisticTell objects.

    Returns:
        Sorted list, highest confidence first.
    """
    return sorted(tells, key=lambda t: t["confidence"], reverse=True)


def filter_tells_by_threshold(
    tells: list[LinguisticTell],
    min_confidence: float = 0.65
) -> list[LinguisticTell]:
    """
    Filter tells to only those meeting a minimum confidence threshold.

    Args:
        tells: List of LinguisticTell objects.
        min_confidence: Minimum confidence score (0.0–1.0).

    Returns:
        Filtered list of high-confidence tells.
    """
    return [t for t in tells if t["confidence"] >= min_confidence]


def categorize_tells_by_type(tells: list[LinguisticTell]) -> dict[str, list[LinguisticTell]]:
    """
    Group tells by their category for analysis.

    Args:
        tells: List of LinguisticTell objects.

    Returns:
        Dictionary mapping category name to list of tells in that category.
    """
    grouped: dict[str, list[LinguisticTell]] = {}
    for tell in tells:
        category = tell["category"]
        if category not in grouped:
            grouped[category] = []
        grouped[category].append(tell)
    return grouped


def compute_tells_score(
    tells: list[LinguisticTell],
    bearish_categories: Optional[list[str]] = None
) -> float:
    """
    Compute aggregate linguistic risk score from tells.

    Args:
        tells: List of LinguisticTell objects.
        bearish_categories: Categories to weight as bearish signals; defaults to all.

    Returns:
        Aggregated score 0.0–1.0 (higher = more bearish linguistic signals).
    """
    if not tells:
        return 0.0

    if bearish
