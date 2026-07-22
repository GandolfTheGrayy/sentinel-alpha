"""
Certainty vs. Hesitation Scoring Engine for Sentinel Linguist.

This module provides prompt templates and scoring logic to analyze corporate text
(earnings calls, SEC filings, press releases) for linguistic markers of confidence
or doubt. Uses Claude via the Anthropic SDK to assess certainty levels and return
structured CertaintyScore objects for downstream use in the Judge pillar.

Integrates with sentinel/linguist/sample_score.py as the reasoning backbone.
"""

from dataclasses import dataclass
from typing import Optional
import os
import json
import anthropic


@dataclass
class CertaintyScore:
    """Structured output from certainty analysis of corporate text."""
    
    text_snippet: str
    overall_certainty: float
    hesitation_markers: list[str]
    confidence_markers: list[str]
    reasoning: str
    raw_response: Optional[str] = None


CERTAINTY_SCORING_PROMPT = """You are a linguistic analyst for financial sentiment. Analyze the following corporate text for markers of certainty vs. hesitation.

Look for:
- CONFIDENCE MARKERS: "will", "expect", "confident", "committed", "strong", "expect to", "on track"
- HESITATION MARKERS: "may", "could", "might", "uncertain", "challenging", "headwinds", "assume", "subject to", "if conditions", "volatile"
- MODAL VERBS: strength of commitment language
- NEGATIONS & CAVEATS: hedge words, risk disclosures, contingencies

Respond in JSON format with:
{{
  "overall_certainty": <0.0 to 1.0 float>,
  "confidence_markers": [<list of detected confident phrases>],
  "hesitation_markers": [<list of detected hesitant phrases>],
  "reasoning": "<brief explanation of scoring>"
}}

CORPORATE TEXT:
{text}"""


def score_certainty(
    text: str,
    company_name: Optional[str] = None,
    document_type: Optional[str] = None
) -> CertaintyScore:
    """
    Analyze corporate text for certainty vs. hesitation using Claude reasoning.
    
    Args:
        text: Corporate text (filing, earnings call, press release) to analyze
        company_name: Optional context for the company (for reasoning clarity)
        document_type: Optional type (e.g., "earnings_call", "10-Q", "press_release")
    
    Returns:
        CertaintyScore dataclass with structured certainty metrics
    
    Raises:
        anthropic.APIError: If API call fails
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    context_prefix = ""
    if company_name and document_type:
        context_prefix = f"[{company_name} — {document_type}] "
    elif company_name:
        context_prefix = f"[{company_name}] "
    
    prompt = CERTAINTY_SCORING_PROMPT.format(text=text)
    if context_prefix:
        prompt = context_prefix + prompt
    
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
    
    response_text = message.content[0].text
    parsed = json.loads(response_text)
    
    return CertaintyScore(
        text_snippet=text[:200] + ("..." if len(text) > 200 else ""),
        overall_certainty=float(parsed.get("overall_certainty", 0.5)),
        hesitation_markers=parsed.get("hesitation_markers", []),
        confidence_markers=parsed.get("confidence_markers", []),
        reasoning=parsed.get("reasoning", ""),
        raw_response=response_text
    )


BATCH_CERTAINTY_PROMPT = """Analyze each of the following corporate text snippets for certainty vs. hesitation.
For each snippet, return a separate JSON object in a JSON array.

Each object should contain:
{{
  "snippet_id": <integer>,
  "overall_certainty": <0.0 to 1.0>,
  "confidence_markers": [<list>],
  "hesitation_markers": [<list>],
  "reasoning": "<brief>"
}}

SNIPPETS (JSON array of objects with 'id' and 'text'):
{snippets_json}

Return ONLY a JSON array of analysis objects, no other text."""


def batch_score_certainty(
    text_snippets: list[tuple[int, str]],
    company_name: Optional[str] = None
) -> list[CertaintyScore]:
    """
    Analyze multiple text snippets in a single API call for efficiency.
    
    Args:
        text_snippets: List of (id, text) tuples to score
        company_name: Optional company context for all snippets
    
    Returns:
        List of CertaintyScore objects corresponding to input snippets
    
    Raises:
        anthropic.APIError: If API call fails
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    snippets_for_prompt = [
        {"id": snippet_id, "text": text}
        for snippet_id, text in text_snippets
    ]
    snippets_json = json.dumps(snippets_for_prompt)
    
    prompt = BATCH_CERTAINTY_PROMPT.format(snippets_json=snippets_json)
    if company_name:
        prompt = f"[Company: {company_name}] " + prompt
    
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    response_text = message.content[0].text
    parsed_array = json.loads(response_text)
    
    results = []
    for parsed, (snippet_id, original_text) in zip(parsed_array, text_snippets):
        score = CertaintyScore(
            text_snippet=original_text[:200] + ("..." if len(original_text) > 200 else ""),
            overall_certainty=float(parsed.get("overall_certainty", 0.5)),
            hesitation_markers=parsed.get("hesitation_markers", []),
            confidence_markers=parsed.get("confidence_markers", []),
            reasoning=parsed.get("reasoning", ""),
            raw_response=response_text
        )
        results.append(score)
    
    return results


if __name__ == "__main__":
    sample_text = """
    We are confident in our ability to deliver strong results this year. While market
    conditions remain uncertain and we may face headwinds in Q3, we expect to grow
    revenue by at least 15%. Our teams are committed to execution, though we acknowledge
    risks from currency volatility and potential supply chain disruptions.
    """
    
    result = score_certainty(sample_text, "TechCorp", "earnings_call")
    print(f"Overall Certainty: {result.overall_certainty}")
    print(f"Confidence Markers: {result.confidence_markers}")
    print(f"Hesitation Markers: {result.hesitation_markers}")
    print(f"Reasoning: {result.reasoning}")
