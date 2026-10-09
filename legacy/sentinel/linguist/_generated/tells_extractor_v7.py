"""
Linguistic Tells Extractor — identifies specific linguistic markers in corporate text
that historically precede price movements.

This module uses Claude (via Anthropic SDK) to analyze corporate communications
(SEC filings, earnings calls, press releases) for semantic and syntactic "tells":
  - Hedging language (might, could, subject to)
  - Cautionary tone shifts
  - Leadership changes in narrative emphasis
  - Regulatory/legal repositioning
  - Earnings guidance walkbacks or revisions

Output: scored tells with historical precedent lookup, fed into Judge for weighting.
Part of the Sentinel Linguist pillar.
"""

import os
import json
from typing import TypedDict, Optional
import anthropic


class Tell(TypedDict):
    """A linguistic tell: marker + confidence + historical precedent."""
    marker: str
    category: str
    confidence: float
    quote: str
    historical_precedent: Optional[str]


class TellsExtraction(TypedDict):
    """Complete extraction result for a text block."""
    ticker: str
    source: str
    tells: list[Tell]
    overall_sentiment_shift: str
    reasoning: str


def extract_tells(
    text: str,
    ticker: str,
    source: str = "unknown",
) -> TellsExtraction:
    """
    Extract linguistic tells from corporate text using Claude reasoning.
    
    Args:
        text: Block of corporate communication (SEC filing, earnings call, etc.)
        ticker: Stock ticker symbol for context.
        source: Source identifier (e.g., "10-K", "earnings_call", "press_release").
    
    Returns:
        TellsExtraction dict with tells, sentiment shift, and reasoning.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    extraction_prompt = f"""
You are a financial linguistics expert analyzing corporate text for "tells" — 
linguistic markers that historically precede stock price movements.

Analyze the following text from {ticker} ({source}) and extract tells in these categories:
  1. Hedging Language: "might", "could", "subject to", "if conditions permit"
  2. Tone Shift: changes in confidence, urgency, or formality vs. prior communications
  3. Leadership Narrative: shifts in what executives emphasize or downplay
  4. Regulatory/Legal Repositioning: new caveats, disclaimers, or compliance language
  5. Guidance Revision: subtle walkbacks or upgrades in forward statements
  6. Revenue/Margin Caution: softening language around financial metrics
  7. Competitive Positioning: new or strengthened claims about market position

For each tell:
  - Quote the specific phrase (or paraphrase if implicit).
  - Assign confidence 0.0–1.0 based on strength of signal.
  - Classify the likely market implication: "bearish", "bullish", or "neutral".
  - Note any historical precedent (e.g., "similar hedge preceded Q2 earnings miss").

Then rate overall sentiment shift: "strengthening", "weakening", or "neutral".

Return a JSON object with keys:
  {{"tells": [...], "overall_sentiment_shift": "...", "reasoning": "..."}}

TEXT TO ANALYZE:
---
{text[:2000]}
---
"""
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": extraction_prompt}],
    )
    
    response_text = message.content[0].text
    
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
        else:
            parsed = {}
    except json.JSONDecodeError:
        parsed = {}
    
    tells: list[Tell] = []
    for tell_raw in parsed.get("tells", []):
        tell: Tell = {
            "marker": tell_raw.get("marker", "unknown"),
            "category": tell_raw.get("category", "unclassified"),
            "confidence": float(tell_raw.get("confidence", 0.5)),
            "quote": tell_raw.get("quote", ""),
            "historical_precedent": tell_raw.get("historical_precedent"),
        }
        tells.append(tell)
    
    result: TellsExtraction = {
        "ticker": ticker,
        "source": source,
        "tells": tells,
        "overall_sentiment_shift": parsed.get("overall_sentiment_shift", "neutral"),
        "reasoning": parsed.get("reasoning", ""),
    }
    
    return result


def batch_extract_tells(
    texts: list[dict],
) -> list[TellsExtraction]:
    """
    Extract tells from multiple text blocks (e.g., multiple filings).
    
    Args:
        texts: List of dicts with keys "text", "ticker", "source".
    
    Returns:
        List of TellsExtraction results.
    """
    results: list[TellsExtraction] = []
    for item in texts:
        result = extract_tells(
            text=item["text"],
            ticker=item.get("ticker", "UNKNOWN"),
            source=item.get("source", "unknown"),
        )
        results.append(result)
    return results


def score_tells_consensus(extractions: list[TellsExtraction]) -> dict:
    """
    Aggregate tells across multiple extractions into a consensus score.
    
    Args:
        extractions: List of TellsExtraction results (e.g., multiple filings for same ticker).
    
    Returns:
        Dict with aggregated confidence, tells frequency, overall direction.
    """
    all_tells: list[Tell] = []
    for extraction in extractions:
        all_tells.extend(extraction["tells"])
    
    if not all_tells:
        return {
            "consensus_direction": "neutral",
            "avg_confidence": 0.0,
            "tell_count": 0,
            "category_distribution": {},
        }
    
    bearish_conf = sum(t["confidence"] for t in all_tells if t.get("category") == "bearish")
    bullish_conf = sum(t["confidence"] for t in all_tells if t.get("category") == "bullish")
    
    if bullish_conf > bearish_conf:
        direction = "bullish"
    elif bearish_conf > bullish_conf:
        direction = "bearish"
    else:
        direction = "neutral"
    
    avg_conf = sum(t["confidence"] for t in all_tells) / len(all_tells) if all_tells else 0.0
    
    category_dist: dict = {}
    for tell in all_tells:
        cat = tell.get("category", "unclassified")
        category_dist[cat] = category_dist.get(cat, 0) + 1
    
    return {
        "consensus_direction": direction,
        "avg_confidence": avg_conf,
        "tell_count": len(all_tells),
        "category_distribution": category_dist,
    }


if __name__ == "__main__":
    sample_text = """
    Forward-looking statements: This company's future performance could be affected
    by various factors, subject to market conditions and regulatory approval. We might
    see headwinds in Q3 if supply chain disruptions persist. Management has noted a shift
    in customer demand patterns, though we remain cautiously optimistic about long-term
    positioning. Legal has added new compliance language regarding international operations.
    """
    
    result = extract_tells(
        text=sample_text,
        ticker="TEST",
        source="earnings_call",
    )
    
    print(json.dumps(result, indent=2, default=str))
    
    consensus = score_tells_consensus([result])
    print("\nConsensus:")
    print(json.dumps(consensus, indent=2))
