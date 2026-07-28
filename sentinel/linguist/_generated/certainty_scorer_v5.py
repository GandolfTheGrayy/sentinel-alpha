"""
Sentinel Linguist: Certainty vs. Hesitation Scoring via LLM Prompt Templates.

This module provides prompt engineering and structured scoring for analyzing corporate
text (earnings calls, SEC filings, news) to detect management confidence levels,
regulatory caution, and linguistic drift. Uses Claude Sonnet for nuanced reasoning
and returns a CertaintyScore dataclass for downstream prediction weighting.

Role in Sentinel: The Linguist pillar uses these templates and scoring logic to
convert raw corporate language into quantitative confidence metrics that feed the
Judge's per-ticker prediction engine.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional
import json
import os

import anthropic


class ConfidenceLevel(Enum):
    """Discrete confidence bucket for quick filtering."""
    VERY_LOW = 1
    LOW = 2
    NEUTRAL = 3
    HIGH = 4
    VERY_HIGH = 5


@dataclass
class CertaintyScore:
    """Structured output from LLM certainty analysis."""
    ticker: str
    source_type: str  # "earnings_call", "10-K", "news", "sec_filing", etc.
    text_excerpt: str  # First 200 chars of analyzed text
    overall_certainty: float  # 0.0–1.0
    confidence_level: ConfidenceLevel
    hesitation_indicators: list[str]  # ["lacks conviction", "pending review", ...]
    regulatory_caution: float  # 0.0–1.0, how much "legalese" hedging detected
    tone_shifts: list[str]  # ["optimism→caution", "guidance raise→maintenance", ...]
    key_phrases: list[str]  # Extracted confident or hedged phrases
    reasoning: str  # Concise explanation of the score
    model_used: str
    tokens_used: int
    raw_response: Optional[str] = None  # For debugging


class CertaintyPromptTemplate:
    """Constructs and manages Claude prompts for certainty scoring."""

    @staticmethod
    def build_scoring_prompt(
        text: str,
        ticker: str,
        source_type: str,
        context: Optional[str] = None
    ) -> str:
        """Build a structured prompt for LLM certainty analysis."""
        context_block = f"\n[CONTEXT: {context}]" if context else ""
        return f"""You are a financial linguistic analyst. Analyze the following corporate text for management certainty, hesitation, and regulatory caution.

[TICKER: {ticker}]
[SOURCE: {source_type}]
[TEXT]
{text[:2000]}
{context_block}

Score this text on these dimensions:

1. OVERALL_CERTAINTY (0.0–1.0): How confident is the tone? High = strong assertions, low = hedging, "pending review", "subject to".
2. HESITATION_INDICATORS (list): Explicit phrases like "may", "could", "subject to", "pending", "challenges", "headwinds".
3. REGULATORY_CAUTION (0.0–1.0): Density of legal hedging language, disclaimers, safe-harbor markers.
4. TONE_SHIFTS (list): Any detectable shifts from prior statements (e.g., "last quarter bullish → now cautious").
5. KEY_PHRASES (list): 3–5 most confident or most hesitant phrases that drove your score.
6. REASONING (1–2 sentences): Why you assigned this certainty level.

Return ONLY valid JSON (no markdown, no preamble):
{{
  "overall_certainty": <float 0–1>,
  "confidence_level": "<VERY_LOW|LOW|NEUTRAL|HIGH|VERY_HIGH>",
  "hesitation_indicators": ["phrase1", "phrase2"],
  "regulatory_caution": <float 0–1>,
  "tone_shifts": ["shift1"],
  "key_phrases": ["phrase1", "phrase2"],
  "reasoning": "explanation"
}}
"""

    @staticmethod
    def build_drift_detection_prompt(
        current_text: str,
        historical_texts: list[str],
        ticker: str
    ) -> str:
        """Build a prompt for detecting linguistic drift over time."""
        historical_block = "\n---\n".join(historical_texts[-3:])  # Last 3 samples
        return f"""You are a financial linguistic drift detector. Compare recent corporate language with historical patterns.

[TICKER: {ticker}]

[CURRENT TEXT]
{current_text[:1500]}

[HISTORICAL SAMPLES (oldest → newest)]
{historical_block[:2000]}

Detect linguistic drift: tone changes, confidence shifts, new risk language, etc.

Return ONLY valid JSON:
{{
  "has_drift": <true|false>,
  "drift_magnitude": <float 0–1>,
  "drift_direction": "<more_confident|more_hesitant|stable>",
  "detected_shifts": ["shift1", "shift2"],
  "anomaly_flag": <true|false>,
  "reasoning": "explanation"
}}
"""

    @staticmethod
    def build_regulatory_whispers_prompt(
        sec_text: str,
        ticker: str
    ) -> str:
        """Build a prompt for detecting subtle regulatory warnings in SEC filings."""
        return f"""You are a regulatory analyst. Detect subtle warnings, risk escalations, and buried disclosures in SEC filings.

[TICKER: {ticker}]
[10-K/8-K EXCERPT]
{sec_text[:2500]}

Look for:
- New or expanded risk factors.
- Buried negative guidance in footnotes.
- Changes in auditor comments or accounting practices.
- Litigation/investigation mentions.
- Regulatory or compliance warnings.

Return ONLY valid JSON:
{{
  "regulatory_risk_score": <float 0–1>,
  "detected_warnings": ["warning1", "warning2"],
  "buried_disclosures": ["disclosure1"],
  "severity": "<low|medium|high>",
  "reasoning": "explanation"
}}
"""


def score_corporate_text(
    text: str,
    ticker: str,
    source_type: str,
    context: Optional[str] = None
) -> CertaintyScore:
    """
    Score a corporate text for certainty, hesitation, and regulatory caution using Claude.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = CertaintyPromptTemplate.build_scoring_prompt(
        text=text,
        ticker=ticker,
        source_type=source_type,
        context=context
    )

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response_text = response.content[0].text.strip()
    parsed = json.loads(response_text)

    confidence_level = ConfidenceLevel[parsed["confidence_level"]]

    return CertaintyScore(
        ticker=ticker,
        source_type=source_type,
        text_excerpt=text[:200],
        overall_certainty=parsed["overall_certainty"],
        confidence_level=confidence_level,
        hesitation_indicators=parsed.get("hesitation_indicators", []),
        regulatory_caution=parsed.get("regulatory_caution", 0.0),
        tone_shifts=parsed.get("tone_shifts", []),
        key_phrases=parsed.get("key_phrases", []),
        reasoning=parsed["reasoning"],
        model_used="claude-3-5-sonnet-20241022",
        tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        raw_response=response_text
    )


def detect_linguistic_drift(
    current_text: str,
    historical_texts: list[str],
    ticker: str
) -> dict:
    """
    Detect tone and confidence shifts relative to historical corporate statements.
    """
    api_key = os.get
