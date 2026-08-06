"""
Linguistic tells extractor for the Sentinel Sentiment Engine.

This module uses Claude to identify specific linguistic patterns and 'tells'
in corporate text (earnings calls, SEC filings, press releases) that historically
correlate with price movements. It performs nuanced semantic analysis to detect
hedging language, conviction shifts, forward guidance changes, and risk
acknowledgments that may signal management confidence or concern.

Fits into the Linguist pillar: transforms raw corporate prose into structured
signals for Judge scoring and RAG confidence weighting.
"""

import os
from typing import TypedDict
import anthropic


class TellsResult(TypedDict):
    """Structured output from tells extractor."""
    hedging_phrases: list[str]
    conviction_signals: list[str]
    risk_acknowledgments: list[str]
    guidance_shifts: list[str]
    tone_markers: dict[str, float]
    anomalies: list[str]
    confidence_score: float
    raw_analysis: str


def extract_tells(corporate_text: str, company_name: str = "") -> TellsResult:
    """
    Extract linguistic tells from corporate text using Claude reasoning.
    
    Args:
        corporate_text: Block of text from earnings call, filing, or press release.
        company_name: Optional company identifier for context.
    
    Returns:
        TellsResult dict with hedging, conviction, risk, and anomaly signals.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = f"""You are a linguistic analyst for a financial prediction system. Analyze the following corporate text for specific "tells" — linguistic patterns that historically correlate with stock price movements.

COMPANY: {company_name if company_name else "Unknown"}

TEXT:
{corporate_text}

Extract and categorize the following:

1. **Hedging Phrases**: Words/phrases that indicate uncertainty or reduced conviction:
   - Examples: "may", "could", "expect to", "subject to", "challenges remain", "headwinds"
   - List exact phrases found, up to 10.

2. **Conviction Signals**: Language suggesting management confidence or intent:
   - Examples: "confident", "committed to", "we will", "driving", "accelerating"
   - List exact phrases found, up to 10.

3. **Risk Acknowledgments**: New or escalated mentions of risks, competition, or headwinds:
   - Examples: "increased competition", "margin pressure", "supply chain", "regulatory risk"
   - List exact risks mentioned, up to 10.

4. **Guidance Shifts**: Changes in forward outlook, targets, or tone vs. historical patterns:
   - Examples: "raising guidance", "maintaining expectations", "cautious near-term"
   - Describe the shift, up to 5.

5. **Tone Markers** (0-1 scale):
   - optimism_index: Overall positive vs. negative tone (0=very negative, 1=very positive)
   - management_conviction: How firm/decisive vs. equivocal (0=highly hedged, 1=very committed)
   - risk_disclosure_intensity: How much new risk language introduced (0=minimal, 1=extensive)

6. **Anomalies**: Unusual phrases, tone breaks, or inconsistencies that stand out:
   - Examples: sudden formality shift, contradictory statements, atypical vocabulary
   - List up to 5.

7. **Confidence Score** (0-1): How confident in this analysis given text length and clarity.

Return ONLY valid JSON (no markdown, no extra text) in this exact format:
{
  "hedging_phrases": ["phrase1", "phrase2"],
  "conviction_signals": ["signal1", "signal2"],
  "risk_acknowledgments": ["risk1", "risk2"],
  "guidance_shifts": ["shift1"],
  "tone_markers": {
    "optimism_index": 0.6,
    "management_conviction": 0.4,
    "risk_disclosure_intensity": 0.7
  },
  "anomalies": ["anomaly1"],
  "confidence_score": 0.85,
  "raw_analysis": "Brief summary of key findings."
}
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    response_text = message.content[0].text
    
    try:
        import json
        result = json.loads(response_text)
        return TellsResult(
            hedging_phrases=result.get("hedging_phrases", []),
            conviction_signals=result.get("conviction_signals", []),
            risk_acknowledgments=result.get("risk_acknowledgments", []),
            guidance_shifts=result.get("guidance_shifts", []),
            tone_markers=result.get("tone_markers", {}),
            anomalies=result.get("anomalies", []),
            confidence_score=result.get("confidence_score", 0.0),
            raw_analysis=result.get("raw_analysis", "")
        )
    except (json.JSONDecodeError, KeyError) as e:
        return TellsResult(
            hedging_phrases=[],
            conviction_signals=[],
            risk_acknowledgments=[],
            guidance_shifts=[],
            tone_markers={},
            anomalies=[f"Parse error: {str(e)}"],
            confidence_score=0.0,
            raw_analysis=response_text
        )


def batch_extract_tells(texts: list[dict]) -> list[TellsResult]:
    """
    Extract tells from a batch of corporate texts.
    
    Args:
        texts: List of dicts with 'text' and optional 'company' keys.
    
    Returns:
        List of TellsResult dicts, one per input text.
    """
    results = []
    for item in texts:
        text = item.get("text", "")
        company = item.get("company", "")
        result = extract_tells(text, company)
        results.append(result)
    return results


if __name__ == "__main__":
    sample_text = """
    We are pleased to report Q3 results with revenue growth of 15% year-over-year.
    However, we continue to face headwinds in the Asia-Pacific region due to increased
    competition and supply chain challenges. While we remain confident in our strategic
    direction, we may need to adjust margins in the near term. We are maintaining our
    full-year guidance, though we expect the fourth quarter could see some softness.
    Our team is committed to driving efficiency and we will continue to invest in R&D.
    """
    
    tells = extract_tells(sample_text, "Example Corp")
    print("Tells Extraction Result:")
    print(f"Hedging Phrases: {tells['hedging_phrases']}")
    print(f"Conviction Signals: {tells['conviction_signals']}")
    print(f"Risk Acknowledgments: {tells['risk_acknowledgments']}")
    print(f"Tone Markers: {tells['tone_markers']}")
    print(f"Confidence: {tells['confidence_score']}")
    print(f"Analysis: {tells['raw_analysis']}")
