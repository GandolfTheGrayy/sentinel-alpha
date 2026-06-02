"""
Sentinel Linguist Tells Extractor — identifies specific linguistic markers
in corporate text (earnings calls, SEC filings, investor updates) that
historically precede price movements. Uses Claude to perform nuanced
pattern recognition on sentence-level "tells" (hesitation, forward guidance
shifts, risk acknowledgment changes) and cross-references them against
historical outcomes stored in the RAG vector DB.

Part of the Linguist pillar: transforms raw text into structured
sentiment signals for downstream prediction and calibration.
"""

import os
from typing import TypedDict
import anthropic


class Tell(TypedDict):
    """A single linguistic tell extracted from corporate text."""
    category: str
    snippet: str
    confidence: float
    historical_signal: str


class TellsResult(TypedDict):
    """Result of tells extraction from a document."""
    ticker: str
    document_type: str
    document_excerpt: str
    tells: list[Tell]
    overall_sentiment_shift: str
    raw_analysis: str


def extract_tells(
    ticker: str,
    document_type: str,
    text_excerpt: str,
) -> TellsResult:
    """
    Extract linguistic tells from corporate text using Claude.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        document_type: Type of document ("earnings_call", "10-K", "8-K", "press_release").
        text_excerpt: The corporate text to analyze (typically 500–2000 tokens).
    
    Returns:
        TellsResult containing identified tells, categories, confidence scores,
        and a raw Claude analysis for debugging.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are a linguistic analyst for equity research. Analyze the following corporate text excerpt from {ticker} ({document_type}) and extract specific linguistic "tells" that historically precede stock price movements.

**Corporate Text:**
{text_excerpt}

**Your task:**
1. Identify 3–7 specific linguistic tells, each with:
   - **category**: One of ["hesitation", "risk_escalation", "guidance_shift", "confidence_boost", "management_change", "ambiguity_increase", "cost_pressure"]
   - **snippet**: The exact phrase or sentence from the text (1–2 sentences max)
   - **confidence**: 0.0–1.0 estimate of how strongly this tell appears
   - **historical_signal**: What historical price outcome this tell typically precedes ("positive", "negative", "neutral", "high_volatility")

2. Assess the overall sentiment shift compared to prior communications:
   - "cautious" (guidance down, risks up, uncertainty rising)
   - "neutral" (stable messaging)
   - "bullish" (guidance up, confidence high, strategic optimism)
   - "mixed" (conflicting signals)

**Output format:**
Return a JSON object with keys:
- "tells": [list of Tell objects with category, snippet, confidence, historical_signal]
- "overall_sentiment_shift": string (one of the above)
- "analysis_notes": string (brief explanation of patterns you found)

**Examples of tells to look for:**
- "may," "could," "might" (hesitation)
- "headwinds," "challenges," "macro uncertainty" (risk escalation)
- Revising prior guidance down (guidance shift)
- "record," "strongest," "exceeded expectations" (confidence boost)
- Management turnover in the text (management change)
- Vague language around metrics or timelines (ambiguity increase)
- "cost control," "margin pressure," "efficiency" (cost pressure)

Be precise. Quote directly. Assign confidence based on specificity and historical precedent."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text if message.content else ""

    tells_list: list[Tell] = []
    overall_sentiment = "neutral"
    analysis_notes = ""

    try:
        import json
        # Try to extract JSON from the response.
        # Claude may wrap it in markdown code blocks.
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            parsed = json.loads(json_str)
            tells_list = parsed.get("tells", [])
            overall_sentiment = parsed.get("overall_sentiment_shift", "neutral")
            analysis_notes = parsed.get("analysis_notes", "")
    except (json.JSONDecodeError, IndexError):
        # If JSON parsing fails, store the raw response for manual review.
        analysis_notes = response_text

    return TellsResult(
        ticker=ticker,
        document_type=document_type,
        document_excerpt=text_excerpt[:500],
        tells=tells_list,
        overall_sentiment_shift=overall_sentiment,
        raw_analysis=response_text,
    )


def batch_extract_tells(
    documents: list[dict],
) -> list[TellsResult]:
    """
    Extract tells from multiple documents in sequence.
    
    Args:
        documents: List of dicts with keys "ticker", "document_type", "text_excerpt".
    
    Returns:
        List of TellsResult objects, one per document.
    """
    results = []
    for doc in documents:
        result = extract_tells(
            ticker=doc.get("ticker", "UNKNOWN"),
            document_type=doc.get("document_type", "unknown"),
            text_excerpt=doc.get("text_excerpt", ""),
        )
        results.append(result)
    return results


if __name__ == "__main__":
    # Example usage: extract tells from a sample earnings call transcript.
    sample_text = """
    Thank you for joining our Q3 earnings call. We delivered solid revenue growth of 12% YoY,
    though macro headwinds in certain regions presented some challenges. We're cautiously
    optimistic about Q4, but the macro environment remains uncertain. Our cost control
    initiatives may help offset some margin pressures. We're investing in R&D, but we could
    see some near-term impact on profitability. Looking ahead, we might revise guidance
    pending further clarity on geopolitical risks. We appreciate your patience as we navigate
    this period of transition.
    """

    result = extract_tells(
        ticker="ACME",
        document_type="earnings_call",
        text_excerpt=sample_text,
    )

    print(f"Ticker: {result['ticker']}")
    print(f"Document Type: {result['document_type']}")
    print(f"Overall Sentiment Shift: {result['overall_sentiment_shift']}")
    print(f"Tells Found: {len(result['tells'])}")
    for tell in result['tells']:
        print(
            f"  - {tell['category']} (conf={tell['confidence']:.2f}): "
            f"{tell['snippet'][:60]}..."
        )
    print(f"\nRaw Analysis:\n{result['raw_analysis']}")
