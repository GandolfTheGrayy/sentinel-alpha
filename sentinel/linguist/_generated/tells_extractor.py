"""
Linguistic Tells Extractor — identifies specific linguistic patterns in corporate text
that historically precede stock price movements.

This module uses Claude to analyze blocks of corporate communication (earnings calls,
SEC filings, press releases, guidance) and extract "tells" — linguistic markers,
hedging patterns, tone shifts, and semantic anomalies that correlate with future
price volatility or directional moves.

Tells are weighted by historical co-occurrence with known market events and serve
as input features for the Judge's predictor pipeline.
"""

import os
import json
from typing import TypedDict, Optional
from anthropic import Anthropic

# Initialize Anthropic client from environment
_client = Anthropic()
_model = "claude-sonnet-4-6"


class Tell(TypedDict):
    """A single linguistic tell extracted from corporate text."""
    category: str  # "hedging" | "urgency" | "tone_shift" | "ambiguity" | "emphasis" | "evasion"
    phrase: str  # The exact phrase or pattern from the source text
    confidence: float  # 0.0 to 1.0, how confident this is a real tell
    historical_signal: str  # "bullish" | "bearish" | "neutral" | "mixed"
    explanation: str  # Why this tell matters


class TellsResult(TypedDict):
    """Output of tells extraction for a single text block."""
    ticker: str
    source_type: str  # "earnings_call" | "sec_filing" | "press_release" | "guidance"
    tells: list[Tell]
    overall_tone: str  # "optimistic" | "neutral" | "cautious" | "defensive"
    summary: str  # High-level interpretation


def extract_tells(
    text: str,
    ticker: str,
    source_type: str = "sec_filing",
    conversation_history: Optional[list[dict]] = None,
) -> TellsResult:
    """
    Extract linguistic tells from a block of corporate text using Claude reasoning.
    
    Analyzes the input for hedging language, urgency markers, tone shifts, and other
    patterns that historically correlate with stock price moves. Uses multi-turn
    conversation to refine and cross-check findings.
    
    Args:
        text: Corporate text block to analyze (earnings call transcript, 10-K, etc.)
        ticker: Stock ticker for context
        source_type: Category of text ("earnings_call", "sec_filing", "press_release", "guidance")
        conversation_history: Optional list of prior messages for multi-turn reasoning
    
    Returns:
        TellsResult dict containing extracted tells, overall tone, and summary
    """
    if conversation_history is None:
        conversation_history = []
    
    # Initial extraction prompt
    system_prompt = f"""You are a senior financial linguist analyzing corporate communications for the Sentinel Sentiment Engine.

Your task: extract specific linguistic "tells" — patterns of speech that historically precede stock price moves.

Focus on:
1. HEDGING: "may", "could", "expected to", "guidance", conditional statements
2. URGENCY: rare exclamation marks, superlatives, repeated emphasis
3. TONE_SHIFT: sudden changes in formality, confidence, or directness
4. AMBIGUITY: vague quantifiers, undefined timelines, unclear metrics
5. EMPHASIS: repetition, capitalization, unusual punctuation or structure
6. EVASION: redirects, topic changes, non-answers to direct questions

For each tell, assign:
- category (one of the 6 above)
- exact_phrase (verbatim from text, max 20 words)
- confidence (0.0–1.0)
- historical_signal ("bullish" | "bearish" | "neutral" | "mixed")
- explanation (why it matters, with historical context if known)

Ticker: {ticker}
Source Type: {source_type}

Extract only the most significant tells (3–8). Ignore boilerplate.
Return JSON only, no preamble."""

    # First turn: extraction
    user_message = f"Analyze this corporate text for linguistic tells:\n\n{text[:3000]}"
    
    conversation_history.append({"role": "user", "content": user_message})
    
    response = _client.messages.create(
        model=_model,
        max_tokens=1500,
        system=system_prompt,
        messages=conversation_history,
    )
    
    extraction_text = response.content[0].text
    conversation_history.append({"role": "assistant", "content": extraction_text})
    
    # Second turn: tone assessment and cross-check
    followup = (
        "Now assess the overall tone of this text (optimistic/neutral/cautious/defensive) "
        "and summarize the key insight for price prediction in 1–2 sentences. "
        "Return JSON with keys: 'overall_tone', 'summary'."
    )
    conversation_history.append({"role": "user", "content": followup})
    
    tone_response = _client.messages.create(
        model=_model,
        max_tokens=500,
        system=system_prompt,
        messages=conversation_history,
    )
    
    tone_text = tone_response.content[0].text
    conversation_history.append({"role": "assistant", "content": tone_text})
    
    # Parse both responses
    tells = []
    overall_tone = "neutral"
    summary = ""
    
    try:
        # Extract JSON from first response
        extraction_json = json.loads(_extract_json(extraction_text))
        if isinstance(extraction_json, dict) and "tells" in extraction_json:
            tells = extraction_json["tells"]
        elif isinstance(extraction_json, list):
            tells = extraction_json
        
        # Extract tone and summary from second response
        tone_json = json.loads(_extract_json(tone_text))
        overall_tone = tone_json.get("overall_tone", "neutral")
        summary = tone_json.get("summary", "")
    except (json.JSONDecodeError, ValueError):
        # Fallback if parsing fails
        tells = []
        overall_tone = "neutral"
        summary = "Extraction failed; check raw response."
    
    return TellsResult(
        ticker=ticker,
        source_type=source_type,
        tells=tells,
        overall_tone=overall_tone,
        summary=summary,
    )


def batch_extract_tells(
    texts: list[dict],
    ticker: str,
) -> list[TellsResult]:
    """
    Extract tells from multiple text blocks for a single ticker.
    
    Processes a list of (text, source_type) tuples and returns aggregated results.
    
    Args:
        texts: List of dicts with keys 'text' and 'source_type'
        ticker: Stock ticker
    
    Returns:
        List of TellsResult dicts, one per input text
    """
    results = []
    shared_history = []  # Maintain conversation context across batch
    
    for item in texts:
        result = extract_tells(
            text=item["text"],
            ticker=ticker,
            source_type=item.get("source_type", "sec_filing"),
            conversation_history=shared_history,
        )
        results.append(result)
    
    return results


def _extract_json(text: str) -> str:
    """
    Extract JSON object or array from text that may contain markdown or preamble.
    
    Args:
        text: Raw response text from Claude
    
    Returns:
        JSON string (first valid JSON object or array found)
    """
    # Try to find JSON block between ``` markers
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    
    # Fallback: find first { or [ and last } or ]
    text = text.strip()
    if text.startswith("{") or text.startswith("["):
        for end_char in ("}",
