"""
Sentinel Linguist — Certainty vs. Hesitation Scoring Engine

This module provides prompt templates and structured scoring for analyzing corporate text
(earnings calls, SEC filings, press releases) to extract certainty and hesitation signals.
Uses Claude (anthropic SDK) for nuanced linguistic reasoning, returning a CertaintyScore
dataclass that feeds into the historian RAG and judge post-mortem pipeline.

Role in Sentinel:
- Accepts raw corporate text blobs from scout ingestion
- Constructs few-shot prompt templates optimized for certainty detection
- Calls Claude Sonnet for reasoning-grade analysis
- Returns structured scores: overall_certainty (0–1), hesitation_flags (list),
  key_phrases (list), confidence_percentile (0–100)
- Outputs feed into historian for event correlation and judge for heuristic refinement
"""

import os
from dataclasses import dataclass
from typing import Optional
import anthropic


@dataclass
class CertaintyScore:
    """Structured output from certainty analysis on corporate text."""

    text_source: str
    """Source label (e.g., 'earnings_call', 'sec_8k', 'press_release')."""

    overall_certainty: float
    """Aggregate certainty score, 0.0 (maximum hesitation) to 1.0 (maximum confidence)."""

    hesitation_flags: list[str]
    """List of detected hesitation markers (e.g., 'may', 'could', 'uncertain', 'TBD')."""

    conviction_phrases: list[str]
    """List of high-conviction statements extracted from text."""

    confidence_percentile: int
    """Analyst's confidence in the score itself, 0–100."""

    raw_reasoning: str
    """Claude's full reasoning trace for audit/refinement."""


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================


CERTAINTY_SYSTEM_PROMPT = """You are a financial linguistic analyst for Sentinel, an autonomous
sentiment prediction engine. Your job is to analyze corporate text and extract two signals:

1. CERTAINTY: How confident/assertive is the speaker/writer?
   - 1.0 = "We will achieve X revenue", "Completed deployment"
   - 0.5 = "We expect to achieve X", "On track for deployment"
   - 0.0 = "We may achieve X", "Deployment uncertain", "TBD"

2. HESITATION: Linguistic markers of doubt, caveats, or ambiguity.

Your response MUST be valid JSON with this exact structure:
{
  "overall_certainty": <float 0.0–1.0>,
  "hesitation_flags": [<list of hesitation words/phrases>],
  "conviction_phrases": [<list of high-confidence statements>],
  "confidence_percentile": <int 0–100>,
  "reasoning": "<your step-by-step analysis>"
}

Return ONLY the JSON object, no markdown, no preamble."""


CERTAINTY_USER_PROMPT_TEMPLATE = """Analyze the following corporate text for certainty vs. hesitation:

TEXT SOURCE: {text_source}
---
{text_body}
---

Focus on:
1. Quantitative vs. qualitative claims ("will" vs. "may", "achieved" vs. "expect")
2. Regulatory/legal hedging language (disclaimer density, "forward-looking statements")
3. Specific timelines vs. vague commitments
4. Tone shifts that suggest management confidence changes

Return structured JSON with overall_certainty, hesitation_flags, conviction_phrases, and your confidence in the score."""


# ============================================================================
# SCORING FUNCTION
# ============================================================================


def score_certainty(
    text: str,
    text_source: str = "unknown",
    api_key: Optional[str] = None,
) -> CertaintyScore:
    """
    Analyze corporate text for certainty vs. hesitation using Claude.

    Args:
        text: Raw corporate text (earnings transcript, SEC filing excerpt, etc.)
        text_source: Label for provenance (e.g., 'earnings_call', 'sec_8k', 'press_release')
        api_key: Anthropic API key; defaults to ANTHROPIC_API_KEY env var.

    Returns:
        CertaintyScore dataclass with structured certainty metrics and reasoning.

    Raises:
        ValueError: If text is empty or API key missing.
        anthropic.APIError: On Claude API failure.
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY env var or api_key param required")

    client = anthropic.Anthropic(api_key=key)

    user_message = CERTAINTY_USER_PROMPT_TEMPLATE.format(
        text_source=text_source,
        text_body=text,
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=CERTAINTY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = response.content[0].text.strip()

    import json

    parsed = json.loads(response_text)

    return CertaintyScore(
        text_source=text_source,
        overall_certainty=float(parsed.get("overall_certainty", 0.5)),
        hesitation_flags=parsed.get("hesitation_flags", []),
        conviction_phrases=parsed.get("conviction_phrases", []),
        confidence_percentile=int(parsed.get("confidence_percentile", 50)),
        raw_reasoning=parsed.get("reasoning", ""),
    )


# ============================================================================
# BATCH SCORING & CONFIDENCE WEIGHTING
# ============================================================================


def score_certainty_batch(
    texts: list[dict],
    api_key: Optional[str] = None,
) -> list[CertaintyScore]:
    """
    Score multiple texts; aggregate confidence into weighted ensemble.

    Args:
        texts: List of dicts with 'text' and 'source' keys.
        api_key: Anthropic API key.

    Returns:
        List of CertaintyScore objects in input order.
    """
    results = []
    for item in texts:
        score = score_certainty(
            text=item["text"],
            text_source=item.get("source", "unknown"),
            api_key=api_key,
        )
        results.append(score)
    return results


def aggregate_certainty_scores(
    scores: list[CertaintyScore],
) -> dict:
    """
    Aggregate multiple CertaintyScore results into ensemble metrics.

    Args:
        scores: List of CertaintyScore objects.

    Returns:
        Dict with ensemble_certainty, avg_hesitation_count, confidence_weighted_certainty.
    """
    if not scores:
        return {
            "ensemble_certainty": 0.5,
            "avg_hesitation_count": 0,
            "confidence_weighted_certainty": 0.5,
        }

    certainties = [s.overall_certainty for s in scores]
    confidences = [s.confidence_percentile for s in scores]

    ensemble = sum(certainties) / len(certainties)
    avg_hesitations = sum(len(s.hesitation_flags) for s in scores) / len(scores)

    confidence_weights = [c / 100.0 for c in confidences]
    weighted_sum = sum(
        c * w for c, w in zip(certainties, confidence_weights)
    )
    weight_total = sum(confidence_weights)
    weighted_certainty = weighted_sum / weight_total if weight_total > 0 else ensemble

    return {
        "ensemble_certainty": round(ensemble, 3),
        "avg_hesitation_count": round(avg_hesitations, 2),
        "confidence_weighted_certainty": round(weighted_certainty, 3),
        "num_scores": len(scores),
    }


# ============================================================================
# REGULATORY WHISPERS DETECTOR (bonus: hedging language patterns)
# ============================================================================


REGULATORY_WHISPERS_PROMPT
