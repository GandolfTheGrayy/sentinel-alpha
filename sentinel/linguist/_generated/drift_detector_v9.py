"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Analyzes a company's current 10-Q filing against a rolling 30-day baseline
of SEC filings and news sentiment to detect significant tone shifts.
Flags language changes that may signal emerging risk or opportunity.

Integrates with:
  - scout/sec_filings.py: retrieves historical 10-Q texts
  - scout/news.py: gathers recent sentiment corpus
  - historian/rag_query.py: vector similarity lookups
  - judge/predictor.py: feeds drift signals into final scoring

Uses Claude (Sonnet 4.6) for nuanced linguistic reasoning.
"""

import os
import json
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import anthropic


@dataclass
class DriftSignal:
    """Represents a detected linguistic drift in company filings."""

    ticker: str
    timestamp: str
    baseline_tone: str
    current_tone: str
    drift_magnitude: float
    key_shifts: list[str]
    confidence: float
    interpretation: str
    risk_flag: bool


def build_rolling_baseline(
    ticker: str,
    baseline_texts: list[str],
) -> str:
    """
    Summarize 30-day rolling baseline of SEC filings and news for a ticker.

    Args:
        ticker: stock symbol (e.g., 'AAPL')
        baseline_texts: list of text snippets from past 30 days

    Returns:
        JSON string describing baseline tone, key themes, language patterns
    """
    if not baseline_texts:
        return json.dumps(
            {
                "ticker": ticker,
                "baseline_period_days": 30,
                "text_count": 0,
                "tone": "neutral",
                "themes": [],
                "warning_keywords": [],
            }
        )

    combined = "\n\n".join(baseline_texts[:10])

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""
Analyze the following corpus of SEC filings and news for {ticker} over the past 30 days.
Identify the dominant tone, recurring themes, and warning/positive language patterns.

Corpus (first 10 documents):
{combined}

Respond with a JSON object containing:
- tone: "cautious", "neutral", "optimistic", or "mixed"
- themes: list of 3-5 key business themes
- warning_keywords: list of phrases suggesting risk (e.g., "headwinds", "margin pressure")
- positive_keywords: list of phrases suggesting strength
- tone_confidence: float 0.0-1.0
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = message.content[0].text
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            baseline_json = response_text[start_idx:end_idx]
            return baseline_json
    except (IndexError, ValueError):
        pass

    return json.dumps(
        {
            "ticker": ticker,
            "tone": "neutral",
            "themes": [],
            "warning_keywords": [],
        }
    )


def analyze_current_filing(
    ticker: str,
    current_10q_text: str,
) -> str:
    """
    Extract tone, themes, and language patterns from current 10-Q.

    Args:
        ticker: stock symbol
        current_10q_text: full text of most recent 10-Q filing

    Returns:
        JSON string describing current filing's linguistic profile
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    snippet = current_10q_text[:5000]

    prompt = f"""
Analyze the linguistic profile of this 10-Q filing excerpt for {ticker}.
Identify tone, language intensity, risk language, and confidence markers.

Filing excerpt:
{snippet}

Respond with a JSON object containing:
- tone: "cautious", "neutral", "optimistic", or "mixed"
- language_intensity: "subdued", "moderate", or "emphatic"
- risk_language_density: float 0.0-1.0 (fraction of text expressing caution)
- confidence_markers: list of phrases showing management confidence or doubt
- key_themes: list of 3-5 topics emphasized
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        response_text = message.content[0].text
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            current_json = response_text[start_idx:end_idx]
            return current_json
    except (IndexError, ValueError):
        pass

    return json.dumps(
        {
            "ticker": ticker,
            "tone": "neutral",
            "language_intensity": "moderate",
        }
    )


def detect_drift(
    ticker: str,
    baseline_profile: str,
    current_profile: str,
    current_filing_text: str,
) -> DriftSignal:
    """
    Compare baseline and current linguistic profiles; quantify drift magnitude.

    Args:
        ticker: stock symbol
        baseline_profile: JSON from build_rolling_baseline()
        current_profile: JSON from analyze_current_filing()
        current_filing_text: raw 10-Q text for quote extraction

    Returns:
        DriftSignal dataclass with magnitude, interpretation, risk flag
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""
Compare these two linguistic profiles for {ticker}.
Baseline (30-day rolling):
{baseline_profile}

Current 10-Q:
{current_profile}

Quantify the drift magnitude (0.0-1.0) and interpret what it signals.
Is there a notable tone shift? Risk emergence? Confidence swing?

Respond with a JSON object containing:
- drift_magnitude: float 0.0-1.0
- tone_shift: string describing the change
- key_shifts: list of 2-3 specific changes
- interpretation: brief explanation of what the drift suggests
- risk_flag: boolean (true if drift suggests emerging risk)
- confidence: float 0.0-1.0 in the assessment
"""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    response_text = message.content[0].text
    start_idx = response_text.find("{")
    end_idx = response_text.rfind("}") + 1

    try:
        result_json = json.loads(response_text[start_idx:end_idx])
    except (ValueError, IndexError):
        result_json = {
            "drift_magnitude": 0.0,
            "tone_shift": "unknown",
            "key_shifts": [],
            "interpretation": "Unable to parse drift analysis",
            "risk_flag": False,
            "confidence": 0.3,
        }

    baseline_tone = json.loads(baseline_profile).get("tone", "neutral")
    current_tone = json.loads(current_profile).get("tone", "neutral")

    return DriftSignal(
        ticker=ticker,
        timestamp=datetime.utcnow().isoformat(),
        baseline_tone=baseline_tone,
        current_tone=current_tone,
        drift_magnitude=float(result_json.get("drift_magnitude", 0.0)),
        key_shifts=result_json.get("key_shifts", []),
        confidence=float(result_json.get("confidence", 0.
