"""
Sentinel Linguist Sentiment Aggregator — composite SentimentResidual scoring.

This module synthesizes Scout signals (price momentum, volume, news sentiment)
and Linguist scores (certainty, linguistic drift, regulatory whispers) into a
unified SentimentResidual metric. The aggregator applies weighted formulas
calibrated against historical prediction accuracy, enabling Judge to rank
confidence in directional calls.

Role in Sentinel:
  - Combines heterogeneous sentiment sources into a single, interpretable score.
  - Supplies SentimentResidual to Judge's per-ticker predictor for final ranking.
  - Supports post-mortem recalibration of weights based on prediction outcomes.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import anthropic


@dataclass
class ScoutSignals:
    """Container for Scout-derived sentiment inputs."""
    price_momentum: float  # -1.0 to +1.0; -1 = strong downtrend, +1 = strong uptrend
    volume_strength: float  # 0.0 to +1.0; abnormal volume indicator
    news_sentiment_score: float  # -1.0 to +1.0; aggregate headline polarity
    reddit_sentiment: float  # -1.0 to +1.0; r/investing, r/stocks discourse
    github_health_delta: float  # -1.0 to +1.0; developer activity trend (for tech)


@dataclass
class LinguistScores:
    """Container for Linguist-derived sentiment inputs."""
    certainty_score: float  # 0.0 to +1.0; confidence in directional bias
    hesitation_markers: float  # 0.0 to +1.0; prevalence of hedging language
    linguistic_drift: float  # -1.0 to +1.0; tone shift vs. 30-day baseline
    regulatory_whispers: float  # -1.0 to +1.0; regulatory risk sentiment


@dataclass
class SentimentResidual:
    """Unified sentiment score for a ticker at a given moment."""
    ticker: str
    timestamp: str
    composite_score: float  # -1.0 to +1.0; final directional prediction
    confidence: float  # 0.0 to +1.0; meta-confidence in the score
    component_breakdown: Dict[str, float]  # scores of each pillar before weighting
    reasoning: str  # qualitative explanation from Claude


def normalize_signal(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp and normalize a signal to [-1.0, +1.0] range."""
    return max(min_val, min(max_val, value))


def aggregate_scout_signals(scout: ScoutSignals) -> Tuple[float, Dict[str, float]]:
    """
    Combine Scout signals into a single momentum score via weighted average.
    Returns (momentum_score, component_dict) where each is [-1, +1].
    """
    weights = {
        "price_momentum": 0.40,
        "volume_strength": 0.20,
        "news_sentiment": 0.25,
        "reddit_sentiment": 0.10,
        "github_health": 0.05,
    }
    
    components = {
        "price_momentum": normalize_signal(scout.price_momentum),
        "volume_strength": normalize_signal(scout.volume_strength - 0.5) * 2,  # center at 0
        "news_sentiment": normalize_signal(scout.news_sentiment_score),
        "reddit_sentiment": normalize_signal(scout.reddit_sentiment),
        "github_health": normalize_signal(scout.github_health_delta),
    }
    
    momentum_score = sum(
        components[k] * weights[k]
        for k in components.keys()
    )
    
    return normalize_signal(momentum_score), components


def aggregate_linguist_scores(linguist: LinguistScores) -> Tuple[float, Dict[str, float]]:
    """
    Combine Linguist scores into a single bias score via weighted average.
    Certainty acts as a gate; hesitation and drift modulate direction.
    Returns (bias_score, component_dict) where bias is [-1, +1].
    """
    # Decay hesitation influence; high hesitation → lower confidence
    hesitation_penalty = normalize_signal(linguist.hesitation_markers * 0.5)
    
    weights = {
        "certainty": 0.50,
        "linguistic_drift": 0.35,
        "regulatory_whispers": 0.15,
    }
    
    components = {
        "certainty": normalize_signal(linguist.certainty_score * 2 - 1),  # map [0,1] → [-1,1]
        "linguistic_drift": normalize_signal(linguist.linguistic_drift),
        "regulatory_whispers": normalize_signal(linguist.regulatory_whispers),
        "hesitation_penalty": -hesitation_penalty,
    }
    
    bias_score = (
        components["certainty"] * weights["certainty"]
        + components["linguistic_drift"] * weights["linguistic_drift"]
        + components["regulatory_whispers"] * weights["regulatory_whispers"]
        + components["hesitation_penalty"] * 0.1
    )
    
    return normalize_signal(bias_score), components


def compute_sentiment_residual_with_claude(
    ticker: str,
    timestamp: str,
    scout: ScoutSignals,
    linguist: LinguistScores,
    momentum: float,
    bias: float,
    momentum_components: Dict[str, float],
    bias_components: Dict[str, float],
) -> SentimentResidual:
    """
    Use Claude Sonnet 4.6 to synthesize momentum + bias into a final score
    and generate qualitative reasoning. Fallback to heuristic if API fails.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    prompt = f"""You are Sentinel's Linguist Judge. Synthesize Scout momentum and Linguist bias into a final SentimentResidual score.

Ticker: {ticker}
Timestamp: {timestamp}

Scout Signals → Momentum Score: {momentum:.3f}
  - Price momentum: {scout.price_momentum:.3f}
  - Volume strength: {scout.volume_strength:.3f}
  - News sentiment: {scout.news_sentiment_score:.3f}
  - Reddit sentiment: {scout.reddit_sentiment:.3f}
  - GitHub health delta: {scout.github_health_delta:.3f}

Linguist Scores → Bias Score: {bias:.3f}
  - Certainty: {linguist.certainty_score:.3f}
  - Hesitation markers: {linguist.hesitation_markers:.3f}
  - Linguistic drift: {linguist.linguistic_drift:.3f}
  - Regulatory whispers: {linguist.regulatory_whispers:.3f}

Task:
1. Determine if momentum and bias are aligned (both positive/negative) or divergent.
2. If aligned, confidence is high. If divergent, require strong evidence to override.
3. Output a JSON object with:
   {{"composite_score": <float -1.0 to +1.0>, "confidence": <float 0.0 to +1.0>, "reasoning": "<brief explanation>"}}

Be concise. Output ONLY the JSON object, no markdown, no preamble."""
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = message.content[0].text.strip()
        result = json.loads(response_text)
        composite = normalize_signal(result.get("composite_score", momentum))
        confidence = max(0.0, min(1.0, result.get("confidence", 0.5)))
        reasoning = result.get("reasoning", "Claude synthesis complete.")
