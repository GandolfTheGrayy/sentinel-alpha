"""
Sentinel Linguist: Tells Extractor

Identifies linguistic 'tells'—specific word choices, tone shifts, hedging patterns,
and regulatory language anomalies—in corporate text (earnings calls, SEC filings,
press releases) that historically correlate with stock price movements.

Uses Claude for nuanced reasoning to flag patterns like:
  - Excessive hedging ("may", "could", "potentially")
  - Guidance withdrawal or cautious forward statements
  - Sudden shifts in executive tone or disclosure depth
  - Regulatory language red flags (restatement language, going concern)
  - Unusual emphasis patterns (repetition, capitalization)

Output: structured tells with confidence scores and historical precedent notes.
"""

import os
from dataclasses import dataclass
from typing import Optional
import anthropic


@dataclass
class Tell:
    """A single linguistic tell with confidence and rationale."""
    category: str
    signal: str
    confidence: float
    rationale: str
    historical_precedent: Optional[str] = None


@dataclass
class TellsReport:
    """Aggregated tells extraction from a text block."""
    ticker: str
    source_type: str
    tells: list[Tell]
    overall_sentiment_vector: str
    extraction_confidence: float


def extract_tells(
    text: str,
    ticker: str,
    source_type: str = "filing",
    context_date: Optional[str] = None,
) -> TellsReport:
    """
    Extract linguistic tells from corporate text using Claude reasoning.

    Args:
        text: Raw text block (earnings transcript, 8-K, press release, etc.)
        ticker: Stock ticker symbol for context.
        source_type: "filing", "earnings_call", "press_release", "investor_update".
        context_date: ISO date string for temporal anchoring in RAG queries.

    Returns:
        TellsReport with structured tells, confidence scores, and historical notes.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = f"""You are a financial linguist trained to identify tells—subtle linguistic patterns that historically precede stock price moves.

Analyze the following {source_type} from {ticker} (dated {context_date or "unknown"}):

---
{text[:4000]}
---

Identify AT MOST 5 high-confidence tells. For each, provide:
1. Category: "hedging", "guidance_shift", "tone_change", "regulatory_red_flag", "emphasis_anomaly", or "disclosure_depth"
2. Signal: The specific phrase, pattern, or absence noted
3. Confidence: 0.0–1.0 (how certain you are this is a genuine tell)
4. Rationale: Why this matters (1–2 sentences)
5. Historical precedent: If this pattern has preceded moves in similar companies (optional)

Also provide an overall_sentiment_vector: "bullish_tell", "bearish_tell", "mixed", or "neutral_noise".

Format as JSON:
{{
  "tells": [
    {{
      "category": "...",
      "signal": "...",
      "confidence": 0.85,
      "rationale": "...",
      "historical_precedent": "..."
    }}
  ],
  "overall_sentiment_vector": "bearish_tell",
  "extraction_confidence": 0.9
}}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response_text = message.content[0].text
    import json

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        data = {
            "tells": [],
            "overall_sentiment_vector": "neutral_noise",
            "extraction_confidence": 0.0,
        }

    tells = [
        Tell(
            category=t.get("category", "unknown"),
            signal=t.get("signal", ""),
            confidence=float(t.get("confidence", 0.0)),
            rationale=t.get("rationale", ""),
            historical_precedent=t.get("historical_precedent"),
        )
        for t in data.get("tells", [])
    ]

    return TellsReport(
        ticker=ticker,
        source_type=source_type,
        tells=tells,
        overall_sentiment_vector=data.get("overall_sentiment_vector", "neutral_noise"),
        extraction_confidence=float(data.get("extraction_confidence", 0.0)),
    )


def rank_tells_by_confidence(report: TellsReport) -> list[Tell]:
    """
    Sort tells in a report by confidence (highest first).

    Args:
        report: TellsReport object.

    Returns:
        Sorted list of Tell objects.
    """
    return sorted(report.tells, key=lambda t: t.confidence, reverse=True)


def filter_tells_by_category(
    report: TellsReport, category: str
) -> list[Tell]:
    """
    Filter tells by category (e.g., "hedging", "regulatory_red_flag").

    Args:
        report: TellsReport object.
        category: Category string to match.

    Returns:
        List of matching Tell objects.
    """
    return [t for t in report.tells if t.category == category]


if __name__ == "__main__":
    sample_text = """
    We are cautiously optimistic about the near-term environment, though
    headwinds remain. Revenue growth may face pressure in Q3 due to supply
    chain uncertainties we discussed last quarter. We are monitoring the
    situation closely and could adjust guidance if necessary.
    
    Our management team has dedicated significant resources to risk mitigation
    strategies. Going forward, we believe operational efficiency improvements
    will offset some macro headwinds, though we must emphasize the uncertain
    nature of these projections.
    """

    report = extract_tells(sample_text, ticker="ACME", source_type="earnings_call")
    print(f"Ticker: {report.ticker}")
    print(f"Source: {report.source_type}")
    print(f"Sentiment Vector: {report.overall_sentiment_vector}")
    print(f"Extraction Confidence: {report.extraction_confidence}")
    print(f"Top Tells ({len(report.tells)}):")
    for tell in rank_tells_by_confidence(report):
        print(f"  [{tell.category}] {tell.signal} (conf: {tell.confidence})")
        print(f"    {tell.rationale}")
