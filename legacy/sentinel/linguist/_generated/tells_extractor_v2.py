"""
Linguistic Tells Extractor for Sentinel Sentiment Engine.

This module identifies specific linguistic patterns and "tells" in corporate text
(earnings calls, SEC filings, press releases) that historically precede stock price
movements. Uses Claude for nuanced pattern recognition and confidence scoring.

Integrates into the Linguist pillar to augment sentiment analysis with behavioral
linguistic signals: hedging language, management tone shifts, euphemistic phrasing,
and anomalous disclosure patterns.
"""

import os
from typing import Optional
from anthropic import Anthropic

# Initialize Claude client from environment
_client = Anthropic()


def extract_tells(
    text: str,
    company_name: str = "Unknown",
    context_type: str = "filing",
    prior_tells: Optional[dict] = None,
) -> dict:
    """
    Extract linguistic tells from corporate text using Claude reasoning.

    Args:
        text: The corporate document text to analyze (earnings call, filing, etc.)
        company_name: Name of the company for context.
        context_type: Type of document ("filing", "earnings_call", "press_release").
        prior_tells: Optional dict of previously extracted tells for comparison/drift.

    Returns:
        Dict with keys:
            - "tells": list of identified tells with reasoning
            - "confidence_score": float 0-1 indicating pattern strength
            - "risk_direction": "bullish", "bearish", or "neutral"
            - "anomalies": list of unusual linguistic patterns
            - "drift_signals": list of tone/language shifts vs. prior_tells (if provided)
    """

    system_prompt = """You are a senior financial linguist analyzing corporate communications
for behavioral tells that historically precede stock price moves. Extract patterns in:

1. **Hedging Language**: "may", "could", "potentially" density and clustering
2. **Management Tone**: Confidence vs. defensiveness, active vs. passive voice
3. **Euphemistic Phrasing**: Softening of negative facts ("headwinds" vs. "decline")
4. **Disclosure Patterns**: Buried bad news, unusual section reordering, omissions
5. **Guidance Ambiguity**: Vague forward-looking statements, narrowed/widened ranges
6. **Regulatory Language**: Sudden increase in legal disclaimers or risk factors

Respond in JSON format with high precision. Flag both bullish tells (confidence in
growth, specific guidance, reduced caveats) and bearish tells (increased hedging,
tone defensiveness, vague guidance)."""

    user_message = f"""Analyze this {context_type} from {company_name}:

---
{text[:3000]}
---

Extract linguistic tells. Return JSON with:
{{
  "tells": [
    {{"pattern": str, "quote": str, "bearish_signal": bool, "confidence": 0-1}}
  ],
  "hedging_density": float (0-1),
  "tone_confidence": str ("high", "medium", "low"),
  "anomalies": [str],
  "risk_direction": "bullish" | "bearish" | "neutral",
  "summary": str (one sentence)
}}"""

    if prior_tells:
        user_message += f"\n\nPrior tells for drift detection:\n{prior_tells}\nNote any significant tone/language shifts."

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        raw_response = response.content[0].text

        import json

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            # Fallback: Claude returned non-JSON, wrap it
            parsed = {
                "tells": [],
                "hedging_density": 0.5,
                "tone_confidence": "low",
                "anomalies": [raw_response[:100]],
                "risk_direction": "neutral",
                "summary": raw_response[:200],
                "raw_response": raw_response,
            }

        # Compute aggregate confidence score
        if parsed.get("tells"):
            confidence_score = sum(
                t.get("confidence", 0.5) for t in parsed["tells"]
            ) / len(parsed["tells"])
        else:
            confidence_score = 0.5

        drift_signals = []
        if prior_tells:
            if prior_tells.get("tone_confidence") != parsed.get("tone_confidence"):
                drift_signals.append(
                    f"Tone shifted from {prior_tells.get('tone_confidence')} to {parsed.get('tone_confidence')}"
                )
            prior_hedging = prior_tells.get("hedging_density", 0.5)
            curr_hedging = parsed.get("hedging_density", 0.5)
            if abs(prior_hedging - curr_hedging) > 0.2:
                drift_signals.append(
                    f"Hedging language increased by {(curr_hedging - prior_hedging):.1%}"
                )

        return {
            "tells": parsed.get("tells", []),
            "confidence_score": confidence_score,
            "risk_direction": parsed.get("risk_direction", "neutral"),
            "anomalies": parsed.get("anomalies", []),
            "tone_confidence": parsed.get("tone_confidence", "medium"),
            "hedging_density": parsed.get("hedging_density", 0.5),
            "drift_signals": drift_signals,
            "summary": parsed.get("summary", ""),
        }

    except Exception as e:
        return {
            "tells": [],
            "confidence_score": 0.0,
            "risk_direction": "neutral",
            "anomalies": [f"Extraction error: {str(e)}"],
            "tone_confidence": "unknown",
            "hedging_density": 0.5,
            "drift_signals": [],
            "summary": f"Failed to extract tells: {str(e)}",
            "error": str(e),
        }


def compare_tells_over_time(
    tells_history: list[dict], window_size: int = 3
) -> dict:
    """
    Detect linguistic drift by comparing tells across recent documents.

    Args:
        tells_history: Ordered list of tells dicts from extract_tells() over time.
        window_size: Number of recent documents to include in trend.

    Returns:
        Dict with keys:
            - "drift_trend": "increasing_hedging", "increasing_confidence", "stable"
            - "confidence_trend": list of confidence scores over window
            - "risk_direction_changes": count of direction reversals
            - "anomaly_clusters": grouped anomalies by pattern
    """

    if not tells_history or len(tells_history) < 2:
        return {
            "drift_trend": "insufficient_data",
            "confidence_trend": [],
            "risk_direction_changes": 0,
            "anomaly_clusters": [],
        }

    recent = tells_history[-window_size:]
    confidence_scores = [t.get("confidence_score", 0.5) for t in recent]
    risk_directions = [t.get("risk_direction", "neutral") for t in recent]

    # Count direction reversals
    direction_changes = sum(
        1
        for i in range(1, len(risk_directions))
        if risk_directions[i] != risk_directions[i - 1]
    )

    # Detect hedging trend
    hedging_scores = [t.get("hedging_density", 0.5) for t in recent]
    avg_hedging = sum(hedging_scores) / len(hedging_scores)
    if avg_hedging > 0.65:
        drift_trend = "increasing_hedging"
    elif avg_hedging < 0.35:
        drift_trend = "increasing_confidence"
    else:
        drift_trend = "stable"

    # Cluster anomalies
    all_anomalies = []
    for t in recent:
        all_anomalies.extend(t.get("anomalies", []))

    anomaly_clusters = {}
    for anomaly in all_
