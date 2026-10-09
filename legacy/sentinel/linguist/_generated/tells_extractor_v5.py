"""
Linguistic Tells Extractor — identifies specific linguistic patterns in corporate text
that historically precede price movements.

Part of Sentinel's Linguist pillar. Uses Claude (claude-sonnet-4-6) to analyze
corporate communications (earnings calls, SEC filings, press releases) for
cautionary language, certainty shifts, and regulatory whispers that correlate
with subsequent market volatility or directional moves.

Integrates with Historian RAG to cross-reference identified tells against
historical precedents, and feeds results into Judge for final prediction scoring.
"""

import os
import json
from typing import Optional
import anthropic


def extract_tells(
    text: str,
    ticker: str = "UNKNOWN",
    source_type: str = "earnings_call",
    confidence_threshold: float = 0.6,
) -> dict:
    """Extract linguistic tells from corporate text using Claude analysis.
    
    Args:
        text: Corporate document text (earnings call transcript, 10-K, press release, etc.)
        ticker: Stock ticker symbol for context.
        source_type: Type of source ('earnings_call', '8-k', '10-q', '10-k', 'press_release').
        confidence_threshold: Minimum confidence (0-1) to include a tell in results.
    
    Returns:
        Dict with keys:
        - 'tells': list of identified tells with type, quote, confidence, explanation
        - 'overall_tone': 'bullish', 'bearish', or 'neutral'
        - 'certainty_delta': float indicating shift from historical baseline (-1 to +1)
        - 'regulatory_whispers': list of potential regulatory red flags
        - 'raw_response': full Claude response for audit trail
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are a financial linguist specializing in identifying tells—
specific linguistic patterns in corporate communications that historically precede
stock price movements.

You analyze corporate text (earnings calls, SEC filings, press releases) for:

1. CAUTIONARY LANGUAGE: Forward-looking disclaimers, hedge words ("may", "could",
   "attempting"), and downside caveats that often signal management concern.

2. CERTAINTY SHIFTS: Comparison of confidence language vs. prior communications.
   Sudden reduction in commitments ("aiming" vs. "will") is a red flag.

3. VAGUE GUIDANCE: Unusual softening of targets, removal of specific numbers,
   or deferral to future periods.

4. REGULATORY WHISPERS: Mentions of audits, investigations, compliance reviews,
   or unusual legal language that may precede formal disclosures.

5. EXECUTIVE TONE CHANGES: Departures from typical CEO communication style,
   unusual terseness, or defensive language.

6. REVENUE/MARGIN LANGUAGE: Shifts in how management describes business drivers,
   pricing power, or cost controls.

Output your analysis as a JSON object with this exact structure:
{
  "tells": [
    {
      "type": "cautionary_language|certainty_shift|vague_guidance|regulatory_whisper|tone_change|revenue_margin",
      "quote": "exact excerpt from text",
      "confidence": 0.0-1.0,
      "explanation": "why this is a tell and what it may signal"
    }
  ],
  "overall_tone": "bullish|bearish|neutral",
  "certainty_delta": -1.0 to 1.0,
  "regulatory_whispers": ["potential issue 1", "potential issue 2"],
  "summary": "brief overall assessment"
}"""

    user_prompt = f"""Analyze the following {source_type} text from {ticker} for linguistic tells:

---
{text[:8000]}
---

Identify tells with high confidence (>0.6). Focus on language that historically
correlates with subsequent price moves. Return valid JSON."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )
    
    response_text = message.content[0].text
    
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            parsed = json.loads(response_text[json_start:json_end])
        else:
            parsed = {
                "tells": [],
                "overall_tone": "neutral",
                "certainty_delta": 0.0,
                "regulatory_whispers": [],
                "summary": "Parse error; raw response returned."
            }
    
    filtered_tells = [
        t for t in parsed.get("tells", [])
        if t.get("confidence", 0) >= confidence_threshold
    ]
    
    return {
        "ticker": ticker,
        "source_type": source_type,
        "tells": filtered_tells,
        "overall_tone": parsed.get("overall_tone", "neutral"),
        "certainty_delta": parsed.get("certainty_delta", 0.0),
        "regulatory_whispers": parsed.get("regulatory_whispers", []),
        "summary": parsed.get("summary", ""),
        "raw_response": response_text,
    }


def batch_extract_tells(
    documents: list[dict],
    confidence_threshold: float = 0.6,
) -> list[dict]:
    """Extract tells from multiple documents in sequence.
    
    Args:
        documents: List of dicts with keys 'text', 'ticker', 'source_type'.
        confidence_threshold: Minimum confidence for inclusion.
    
    Returns:
        List of extraction results (one per document).
    """
    results = []
    for doc in documents:
        result = extract_tells(
            text=doc.get("text", ""),
            ticker=doc.get("ticker", "UNKNOWN"),
            source_type=doc.get("source_type", "earnings_call"),
            confidence_threshold=confidence_threshold,
        )
        results.append(result)
    return results


def score_tells_severity(tells_result: dict) -> float:
    """Aggregate tells into a single severity score (0-1, higher = more bearish signal).
    
    Args:
        tells_result: Output from extract_tells().
    
    Returns:
        Float from 0 (bullish) to 1 (bearish).
    """
    tells = tells_result.get("tells", [])
    if not tells:
        base = 0.5
    else:
        avg_confidence = sum(t.get("confidence", 0) for t in tells) / len(tells)
        base = avg_confidence
    
    tone_adjustment = {
        "bearish": +0.15,
        "neutral": 0.0,
        "bullish": -0.15,
    }.get(tells_result.get("overall_tone", "neutral"), 0.0)
    
    certainty_delta = tells_result.get("certainty_delta", 0.0)
    regulatory_weight = min(0.1, 0.05 * len(tells_result.get("regulatory_whispers", [])))
    
    severity = base + tone_adjustment - (certainty_delta * 0.1) + regulatory_weight
    return max(0.0, min(1.0, severity))


if __name__ == "__main__":
    sample_text = """
    Q3 2024 Earnings Call Transcript
    
    CEO: We're pleased with our performance this quarter. Revenue grew 12% YoY,
    though we're facing some headwinds in our core markets. We're attempting to
    maintain margin discipline while investing in new initiatives. Guidance for Q4
    may be conservative given macro uncertainty.
    
    CFO: Free cash flow declined due to working capital movements and one-time
    integration costs. We may need to revisit our capital allocation strategy.
    
    Analyst: Any concerns on the regulatory front?
    
    CEO: We're in discussions with regulators on a routine compliance matter. Nothing
