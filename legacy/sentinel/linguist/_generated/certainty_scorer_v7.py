"""
Certainty vs. Hesitation scoring system for Sentinel Linguist pillar.

This module provides prompt templates and scoring logic to analyze corporate
text (earnings calls, SEC filings, news) for linguistic markers of confidence
or uncertainty. Uses Claude for nuanced reasoning to produce structured
CertaintyScore objects that feed into the Judge's prediction pipeline.

Integration point: Called by sentinel/linguist/sample_score.py during
multi-source sentiment aggregation.
"""

import os
from dataclasses import dataclass
from typing import Optional
import anthropic


@dataclass
class CertaintyScore:
    """Structured output from LLM certainty analysis."""
    
    overall_certainty: float
    """Overall confidence score (0.0–1.0): 1.0 = high conviction, 0.0 = high uncertainty."""
    
    conviction_markers: list[str]
    """List of identified confidence keywords/phrases (e.g., 'will', 'confident', 'strong')."""
    
    hesitation_markers: list[str]
    """List of identified uncertainty keywords/phrases (e.g., 'may', 'could', 'uncertain')."""
    
    tone_shift: float
    """Directional sentiment shift (-1.0 to +1.0): +1.0 = increasingly bullish, -1.0 = increasingly bearish."""
    
    regulatory_whispers: Optional[str]
    """Any detected hints of regulatory concern, restatement risk, or compliance issues."""
    
    explanation: str
    """Human-readable summary of scoring rationale."""


def build_certainty_prompt(text: str, context: Optional[str] = None) -> str:
    """
    Construct Claude prompt for analyzing certainty markers in corporate text.
    
    Args:
        text: The corporate text (earnings transcript, 10-K excerpt, news, etc.)
        context: Optional background (ticker, document type, date) for framing.
    
    Returns:
        Formatted prompt string ready for Claude API.
    """
    context_block = f"\n[Context: {context}]" if context else ""
    
    prompt = f"""You are a linguistic analyst for a financial sentiment engine. Analyze the following corporate text for confidence vs. hesitation markers.

{context_block}

TEXT TO ANALYZE:
---
{text}
---

Perform these checks:

1. **Conviction Markers**: Identify words/phrases indicating high confidence (e.g., 'will', 'expect', 'confident', 'strong', 'robust', 'solid').

2. **Hesitation Markers**: Identify words/phrases indicating uncertainty or caution (e.g., 'may', 'could', 'might', 'uncertain', 'challenging', 'headwinds', 'downside risk').

3. **Overall Certainty Score**: Rate the speaker's overall conviction on a scale of 0.0 (maximum uncertainty) to 1.0 (maximum certainty). Consider the ratio and intensity of conviction vs. hesitation markers.

4. **Tone Shift**: Detect any directional shift in sentiment:
   - +1.0: increasingly bullish/optimistic
   - 0.0: neutral or balanced
   - -1.0: increasingly bearish/pessimistic

5. **Regulatory Whispers**: Flag any subtle hints of:
   - Regulatory scrutiny or investigation
   - Restatement risk or accounting concerns
   - Compliance warnings
   - Litigation threats
   Leave blank if none detected.

6. **Explanation**: Provide a brief (1-2 sentence) rationale for your overall_certainty score.

Output ONLY valid JSON in this exact format (no markdown, no extra text):
{{
  "overall_certainty": <float 0.0-1.0>,
  "conviction_markers": [<list of strings>],
  "hesitation_markers": [<list of strings>],
  "tone_shift": <float -1.0 to +1.0>,
  "regulatory_whispers": <null or string>,
  "explanation": "<string>"
}}
"""
    return prompt


def score_certainty(
    text: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None
) -> CertaintyScore:
    """
    Call Claude to analyze certainty markers and return structured CertaintyScore.
    
    Args:
        text: Corporate text to analyze.
        context: Optional context string (e.g., "AAPL 10-K 2024-01-15").
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
    
    Returns:
        CertaintyScore dataclass populated with Claude's analysis.
    
    Raises:
        ValueError: If Claude response cannot be parsed as valid JSON.
        anthropic.APIError: If API call fails.
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_certainty_prompt(text, context)
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    response_text = message.content[0].text.strip()
    
    # Parse JSON response
    import json
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude response not valid JSON: {response_text}") from e
    
    # Validate and construct CertaintyScore
    try:
        return CertaintyScore(
            overall_certainty=float(parsed["overall_certainty"]),
            conviction_markers=list(parsed["conviction_markers"]),
            hesitation_markers=list(parsed["hesitation_markers"]),
            tone_shift=float(parsed["tone_shift"]),
            regulatory_whispers=parsed.get("regulatory_whispers"),
            explanation=str(parsed["explanation"])
        )
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"Invalid response structure: {parsed}") from e


def batch_score_certainty(
    texts: list[tuple[str, Optional[str]]],
    api_key: Optional[str] = None
) -> list[CertaintyScore]:
    """
    Score multiple text snippets (e.g., from different news sources or SEC docs).
    
    Args:
        texts: List of (text, context) tuples.
        api_key: Anthropic API key.
    
    Returns:
        List of CertaintyScore objects in same order as input.
    """
    results = []
    for text, context in texts:
        score = score_certainty(text, context, api_key)
        results.append(score)
    return results
