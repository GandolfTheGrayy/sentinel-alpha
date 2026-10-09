"""
Sentinel Linguist: Certainty vs. Hesitation Prompt Template System

This module provides LLM-driven analysis of corporate text (SEC filings, earnings calls,
news) to extract a structured CertaintyScore reflecting management confidence, tone drift,
and regulatory language patterns. Used by sentinel/linguist/sample_score.py as the
reasoning engine via Claude Sonnet 4.6.

The prompter builds context-aware prompts that guide Claude to score certainty on:
  - Quantified commitments vs. conditional language
  - Historical tone baseline vs. current drift
  - Regulatory whisper intensity (use of disclaimers, caveats, boilerplate)
  - Sentiment polarity within the text

Returns structured CertaintyScore objects for aggregation into final predictions.
"""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class CertaintyScore:
    """
    Structured output from LLM certainty analysis of corporate text.
    
    Attributes:
        ticker: Stock ticker symbol (e.g., 'AAPL').
        text_source: Origin of analyzed text (e.g., '10-Q', 'earnings_call', 'news').
        overall_certainty: Float [0, 1] — overall confidence/decisiveness in tone.
        commitment_strength: Float [0, 1] — degree of quantified vs. conditional language.
        regulatory_caution: Float [0, 1] — intensity of disclaimers/boilerplate hedging.
        sentiment_polarity: Float [-1, 1] — positive (>0) to negative (<0) sentiment.
        linguistic_drift: Float [-1, 1] — shift from historical baseline (-1=more hesitant, 1=more confident).
        key_phrases: List of extracted high-conviction or high-caution phrases.
        reasoning: Brief explanation of score derivation.
        confidence_in_score: Float [0, 1] — LLM's self-reported confidence in this analysis.
    """
    ticker: str
    text_source: str
    overall_certainty: float
    commitment_strength: float
    regulatory_caution: float
    sentiment_polarity: float
    linguistic_drift: float
    key_phrases: list[str]
    reasoning: str
    confidence_in_score: float

    def to_dict(self) -> dict:
        """Convert CertaintyScore to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert CertaintyScore to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def build_certainty_prompt(
    ticker: str,
    text: str,
    text_source: str,
    historical_baseline: Optional[str] = None,
    regulatory_context: Optional[str] = None,
) -> str:
    """
    Build a structured prompt for Claude to analyze certainty and hesitation in corporate text.
    
    Args:
        ticker: Stock ticker symbol.
        text: The corporate text to analyze (SEC filing excerpt, news, earnings call transcript).
        text_source: Description of text origin (e.g., '10-Q filing', 'earnings call Q&A').
        historical_baseline: Optional prior tone/language pattern for drift detection.
        regulatory_context: Optional note about current regulatory environment or recent events.
    
    Returns:
        Formatted prompt string ready for Claude API submission.
    """
    baseline_section = ""
    if historical_baseline:
        baseline_section = f"""

**Historical Baseline (for drift detection):**
{historical_baseline}

Compare the current text's tone, conviction, and hedging patterns to this baseline.
Flag shifts toward greater caution or greater confidence."""

    regulatory_section = ""
    if regulatory_context:
        regulatory_section = f"""

**Regulatory / Market Context:**
{regulatory_context}

Consider how external pressures may influence cautious or confident language."""

    prompt = f"""You are a linguistic analyst specializing in corporate communications and financial disclosure language. Your task is to score the certainty vs. hesitation present in the following {text_source} from {ticker}.

**Text to analyze:**
{text}
{baseline_section}
{regulatory_section}

**Your analysis should produce a JSON response with these fields:**

{{
  "overall_certainty": <float 0-1: 1=highly confident/decisive tone, 0=highly hedged/uncertain>,
  "commitment_strength": <float 0-1: 1=specific quantified commitments, 0=vague conditional language>,
  "regulatory_caution": <float 0-1: 1=heavy disclaimers/boilerplate, 0=minimal hedging>,
  "sentiment_polarity": <float -1 to 1: -1=very negative, 0=neutral, 1=very positive>,
  "linguistic_drift": <float -1 to 1: -1=more hesitant than baseline, 0=no change, 1=more confident than baseline>,
  "key_phrases": [<list of 3-5 high-conviction or high-caution phrases extracted from text>],
  "reasoning": "<2-3 sentence explanation of how you arrived at these scores>",
  "confidence_in_score": <float 0-1: your confidence in this analysis given text quality/clarity>
}}

**Scoring guidelines:**

1. **Overall Certainty**: Look for:
   - Strong declarative statements ("We will achieve...", "Our plan is...") → higher certainty
   - Modal verbs ("may", "could", "might", "appears to") → lower certainty
   - Frequency of conditionals ("if", "assuming", "subject to") → lower certainty

2. **Commitment Strength**: Distinguish:
   - Quantified targets ("30% revenue growth by Q4 2025") → higher
   - Vague goals ("significant improvements", "enhanced efficiency") → lower
   - Forward-looking statements with disclaimers → lower

3. **Regulatory Caution**: Measure:
   - Density of "safe harbor" disclaimers, risk factor boilerplate
   - References to "forward-looking statements"
   - Frequency of "we cannot guarantee", "unforeseen circumstances", etc.

4. **Sentiment Polarity**: Assess:
   - Positive terms (growth, strength, opportunity) → positive
   - Negative terms (risk, decline, challenge) → negative
   - Neutral/balanced discussion → near zero

5. **Linguistic Drift** (if baseline provided):
   - Compare conviction level, specificity, hedging patterns
   - Note shifts in tone or frequency of caution language

6. **Key Phrases**: Extract:
   - Strongest conviction statements
   - Highest-caution hedges
   - Novel or surprising commitments

Return ONLY valid JSON. No markdown, no prose, no preamble."""

    return prompt


def parse_certainty_response(
    response_text: str,
    ticker: str,
    text_source: str,
) -> CertaintyScore:
    """
    Parse Claude's JSON response into a CertaintyScore dataclass.
    
    Args:
        response_text: Raw text response from Claude (should be valid JSON).
        ticker: Stock ticker for context.
        text_source: Description of analyzed text origin.
    
    Returns:
        Populated CertaintyScore object.
    
    Raises:
        ValueError: If JSON parsing fails or required fields are missing.
    """
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Claude response as JSON: {e}") from e

    required_fields = {
        "overall_certainty",
        "commitment_strength",
        "regulatory_caution",
        "sentiment_polarity",
        "linguistic_drift",
        "key_phrases",
        "reasoning",
        "confidence_in_score",
    }

    missing = required_fields - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields in response: {missing}")

    return CertaintyScore(
        ticker=ticker,
        text_source=text_source,
        overall_certainty=float(data["overall_certainty"]),
        commitment_strength=float(data["commitment_strength"]),
        regulatory_caution=float(data["regulatory_caution"]),
        sentiment_polarity=float(data["sentiment_polarity"]),
