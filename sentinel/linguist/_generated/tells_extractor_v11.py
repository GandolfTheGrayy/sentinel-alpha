"""
Sentinel Linguist: Tells Extractor

This module uses Claude to identify linguistic "tells" — specific word patterns,
hedging language, tone shifts, and regulatory markers — within corporate text
(earnings calls, SEC filings, press releases) that historically precede price moves.

Tells are extracted via few-shot prompting and returned with confidence scores
and historical precedent lookups. Output is suitable for feeding into the Judge
for final prediction weighting.
"""

import os
from typing import TypedDict, Optional
import anthropic


class Tell(TypedDict):
    """A single linguistic tell extracted from corporate text."""
    category: str  # e.g. "hedging", "guidance_cut", "tone_shift", "risk_acknowledgment"
    phrase: str  # The actual quoted phrase or pattern
    confidence: float  # 0.0–1.0, Claude's confidence in the tell's presence
    direction: str  # "bullish", "bearish", or "neutral"
    explanation: str  # Why this tell matters historically


class TellsExtractionResult(TypedDict):
    """Result of extracting tells from a single document."""
    ticker: str
    document_type: str  # "earnings_call", "10-Q", "8-K", "press_release"
    tells: list[Tell]
    overall_sentiment_shift: str  # "improving", "deteriorating", "stable"
    anomalies: list[str]  # Unusual patterns flagged by Claude


def extract_tells(
    ticker: str,
    document_type: str,
    text: str,
    context_prior_tone: Optional[str] = None,
) -> TellsExtractionResult:
    """
    Extract linguistic tells from corporate text using Claude.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
        document_type: Type of document ("earnings_call", "10-Q", "8-K", "press_release")
        text: Full text of the document to analyze
        context_prior_tone: Optional prior sentiment for drift detection (e.g., "cautious", "optimistic")
    
    Returns:
        TellsExtractionResult with extracted tells, confidence scores, and anomalies
    """
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    system_prompt = """You are a financial linguist specializing in identifying subtle 
tells in corporate communications that precede stock price moves. Analyze the provided 
text and extract specific linguistic patterns.

Focus on:
1. HEDGING: Words like "may", "could", "might", "if", "subject to" that weaken claims
2. GUIDANCE CUTS: Reduction in forward guidance, lowered targets, narrowed ranges
3. TONE SHIFT: Changes from prior communications (e.g., from bullish to cautious)
4. RISK ACKNOWLEDGMENT: Explicit enumeration of risks, use of "material adverse"
5. INSIDER CAUTION: Reduced insider buying mentions, increased selling language
6. SEQUENTIAL WEAKNESS: Phrases like "demand normalization", "macro headwinds"
7. REGULATORY WHISPERS: New compliance costs, investigation mentions, policy uncertainty

For each tell identified, provide:
- The exact phrase or pattern found
- Confidence (0.0–1.0) in the tell's genuine presence
- Direction (bullish/bearish/neutral)
- Brief explanation of historical precedent

Return JSON with structure:
{
  "tells": [
    {
      "category": "...",
      "phrase": "...",
      "confidence": 0.85,
      "direction": "bearish",
      "explanation": "..."
    }
  ],
  "overall_sentiment_shift": "improving|deteriorating|stable",
  "anomalies": ["pattern1", "pattern2"]
}
"""
    
    user_prompt = f"""Analyze this {document_type} for {ticker}:

{"Context: Prior tone was: " + context_prior_tone if context_prior_tone else ""}

---TEXT---
{text[:8000]}  # Truncate to avoid token limits
---END---

Extract all linguistic tells you detect. Be precise with quotes and confidence."""
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    response_text = message.content[0].text
    
    import json
    try:
        # Extract JSON from response (Claude may wrap it in markdown)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        # Fallback if Claude's response doesn't parse
        parsed = {
            "tells": [],
            "overall_sentiment_shift": "stable",
            "anomalies": ["Failed to parse Claude response"]
        }
    
    return TellsExtractionResult(
        ticker=ticker,
        document_type=document_type,
        tells=parsed.get("tells", []),
        overall_sentiment_shift=parsed.get("overall_sentiment_shift", "stable"),
        anomalies=parsed.get("anomalies", [])
    )


def batch_extract_tells(
    ticker: str,
    documents: list[dict],
) -> list[TellsExtractionResult]:
    """
    Extract tells from multiple documents for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        documents: List of dicts with keys "type", "text", optional "prior_tone"
    
    Returns:
        List of TellsExtractionResult, one per document
    """
    results = []
    for doc in documents:
        result = extract_tells(
            ticker=ticker,
            document_type=doc.get("type", "unknown"),
            text=doc.get("text", ""),
            context_prior_tone=doc.get("prior_tone")
        )
        results.append(result)
    return results


def synthesize_tells(
    results: list[TellsExtractionResult],
) -> dict:
    """
    Synthesize multiple tell-extraction results into a single summary.
    
    Args:
        results: List of TellsExtractionResult from multiple documents
    
    Returns:
        Aggregated tells summary with most critical patterns and overall direction
    """
    all_tells = []
    sentiment_shifts = []
    all_anomalies = []
    
    for result in results:
        all_tells.extend(result["tells"])
        sentiment_shifts.append(result["overall_sentiment_shift"])
        all_anomalies.extend(result["anomalies"])
    
    # Sort by confidence descending
    all_tells.sort(key=lambda t: t["confidence"], reverse=True)
    
    # Determine net direction
    bearish_count = sum(1 for t in all_tells if t["direction"] == "bearish")
    bullish_count = sum(1 for t in all_tells if t["direction"] == "bullish")
    net_direction = "bearish" if bearish_count > bullish_count else ("bullish" if bullish_count > bearish_count else "neutral")
    
    return {
        "top_tells": all_tells[:10],  # Top 10 by confidence
        "total_tells_count": len(all_tells),
        "net_direction": net_direction,
        "sentiment_progression": sentiment_shifts,
        "critical_anomalies": list(set(all_anomalies)),
        "average_confidence": sum(t["confidence"] for t in all_tells) / len(all_tells) if all_tells else 0.0
    }
