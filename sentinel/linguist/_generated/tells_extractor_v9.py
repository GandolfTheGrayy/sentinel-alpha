"""
Linguistic Tells Extractor for Sentinel Sentiment Engine.

This module uses Claude to identify specific linguistic tells in corporate text
that historically precede price moves. It analyzes blocks of text (earnings calls,
SEC filings, press releases) for hesitation markers, uncertainty language,
forward guidance shifts, and other sentiment signals that correlate with
market volatility.

Integrates with the Linguist pillar to enrich sentiment scoring beyond
simple polarity analysis. Outputs structured tells with confidence scores
for downstream RAG weighting and Judge calibration.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import anthropic


@dataclass
class LinguisticTell:
    """Represents a single identified linguistic tell with metadata."""

    category: str
    tell_text: str
    quote: str
    confidence: float
    historical_correlation: Optional[str] = None


@dataclass
class TellsExtractionResult:
    """Result of extracting tells from a block of corporate text."""

    ticker: str
    source_type: str
    text_snippet: str
    tells: list[LinguisticTell]
    overall_sentiment_shift: str
    raw_response: str


def extract_tells(
    text: str,
    ticker: str,
    source_type: str = "earnings_call",
    historical_context: Optional[str] = None,
) -> TellsExtractionResult:
    """
    Extract linguistic tells from corporate text using Claude reasoning.

    Args:
        text: Block of corporate text (earnings transcript, SEC filing, etc.)
        ticker: Stock ticker symbol for context.
        source_type: Type of source (earnings_call, sec_8k, press_release, etc.)
        historical_context: Optional background on this company's historical tells.

    Returns:
        TellsExtractionResult containing structured tells, sentiment shift, and raw response.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """You are a linguistic analyst specializing in detecting tells in corporate communications that historically precede stock price movements.

Your task is to identify specific linguistic markers that correlate with market moves:
- Hesitation markers (hedging language, filler words, increased qualifiers)
- Uncertainty escalation (shifts from confident to cautious tone)
- Forward guidance retreat (softening or withdrawing previous statements)
- Evasion patterns (topic avoidance, non-answers to direct questions)
- Tone shifts (sudden changes in confidence or formality)
- Regulatory whispers (cautious language around compliance or legal matters)
- Cash flow language shifts (changes in discussion of liquidity or capital allocation)

Return your analysis as a JSON object with the following structure:
{
  "tells": [
    {
      "category": "hesitation_markers|uncertainty_escalation|guidance_retreat|evasion|tone_shift|regulatory_whisper|cash_flow_shift",
      "tell_text": "brief description of the tell",
      "quote": "exact quote from text demonstrating the tell",
      "confidence": 0.0-1.0,
      "historical_correlation": "brief note on historical correlation if known"
    }
  ],
  "overall_sentiment_shift": "positive|neutral|negative|mixed",
  "summary": "brief synthesis of what the tells collectively suggest"
}

Be rigorous: only flag tells you are confident are present. Avoid over-interpretation."""

    user_message = f"""Analyze the following {source_type} from {ticker} for linguistic tells:

{text}

{f"Historical context: {historical_context}" if historical_context else ""}

Extract tells using the JSON structure specified."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = message.content[0].text

    try:
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            parsed = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")
    except (json.JSONDecodeError, ValueError) as e:
        return TellsExtractionResult(
            ticker=ticker,
            source_type=source_type,
            text_snippet=text[:500],
            tells=[],
            overall_sentiment_shift="unknown",
            raw_response=response_text,
        )

    tells = []
    for tell_dict in parsed.get("tells", []):
        tells.append(
            LinguisticTell(
                category=tell_dict.get("category", "unknown"),
                tell_text=tell_dict.get("tell_text", ""),
                quote=tell_dict.get("quote", ""),
                confidence=float(tell_dict.get("confidence", 0.5)),
                historical_correlation=tell_dict.get("historical_correlation"),
            )
        )

    return TellsExtractionResult(
        ticker=ticker,
        source_type=source_type,
        text_snippet=text[:500],
        tells=tells,
        overall_sentiment_shift=parsed.get("overall_sentiment_shift", "unknown"),
        raw_response=response_text,
    )


def batch_extract_tells(
    texts: list[dict],
    historical_context: Optional[str] = None,
) -> list[TellsExtractionResult]:
    """
    Extract tells from multiple text blocks (e.g., earnings transcripts from multiple quarters).

    Args:
        texts: List of dicts with keys: text, ticker, source_type.
        historical_context: Optional shared historical context for all extractions.

    Returns:
        List of TellsExtractionResult objects.
    """
    results = []
    for item in texts:
        result = extract_tells(
            text=item["text"],
            ticker=item.get("ticker", "UNKNOWN"),
            source_type=item.get("source_type", "unknown"),
            historical_context=historical_context,
        )
        results.append(result)
    return results


def tells_to_sentiment_weight(result: TellsExtractionResult) -> float:
    """
    Convert TellsExtractionResult to a sentiment weight for downstream scoring.

    Args:
        result: Extracted tells and sentiment from corporate text.

    Returns:
        Float weight [-1.0, 1.0] where -1.0 = very negative tells, 1.0 = very positive.
    """
    if not result.tells:
        base_weight = 0.0
    else:
        avg_confidence = sum(t.confidence for t in result.tells) / len(result.tells)

        negative_categories = {
            "hesitation_markers",
            "uncertainty_escalation",
            "guidance_retreat",
            "evasion",
            "regulatory_whisper",
        }
        positive_categories = set()

        negative_tells = [
            t for t in result.tells if t.category in negative_categories
        ]
        positive_tells = [t for t in result.tells if t.category in positive_categories]

        if len(negative_tells) > len(positive_tells):
            base_weight = -avg_confidence
        elif len(positive_tells) > len(negative_tells):
            base_weight = avg_confidence
        else:
            base_weight = 0.0

    sentiment_shift_multiplier = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
        "mixed": 0.0,
        "unknown": 0.0,
    }
    shift_mult = sentiment_shift_multiplier.get(result.overall_sentiment_shift, 0.0)

    final_weight = base_weight * shift_mult
    return max(-1.0, min(1.0, final_weight))


if __name__ == "__main__":
