"""
Sentiment Aggregator for Sentinel Sentiment Engine.

This module synthesizes Scout signals (price momentum, news tone, Reddit/HN sentiment,
SEC filing sentiment) and Linguist scores (certainty, hesitation, linguistic drift,
regulatory whispers) into a unified SentimentResidual composite score.

The aggregator uses a weighted formula where weights are calibrated against historical
market moves via the Judge post-mortem feedback loop. Output includes both a scalar
residual score (-1.0 to +1.0) and a confidence interval for downstream prediction.

Role in Sentinel:
  - Consumed by sentinel/judge/predictor.py to generate per-ticker price predictions.
  - Weights are refined daily by sentinel/judge/resolver.py post-mortem analysis.
  - Used in sentinel/judge/baselines.py for threshold-based trading signals.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScoutSignals:
    """Raw signals from sentinel/scout modules."""
    
    price_momentum: float  # -1.0 (down) to +1.0 (up), 1d/5d/20d composite
    volume_spike: float    # 0.0 (none) to +1.0 (extreme spike)
    news_tone: float       # -1.0 (very negative) to +1.0 (very positive)
    reddit_sentiment: float  # -1.0 to +1.0
    hackernews_sentiment: float  # -1.0 to +1.0
    sec_filing_sentiment: Optional[float]  # None if no recent filing, else -1.0 to +1.0
    timestamp: datetime


@dataclass
class LinguistScores:
    """Linguistic and semantic signals from sentinel/linguist modules."""
    
    certainty_score: float  # 0.0 (very uncertain) to +1.0 (highly certain)
    hesitation_index: float  # 0.0 (confident) to +1.0 (many hedges/qualifiers)
    linguistic_drift: float  # -1.0 (tone shift negative) to +1.0 (shift positive)
    regulatory_whispers: float  # 0.0 (none detected) to +1.0 (strong regulatory risk signal)
    timestamp: datetime


@dataclass
class SentimentResidual:
    """Composite sentiment output combining Scout and Linguist signals."""
    
    ticker: str
    residual_score: float  # Net sentiment: -1.0 (strong bearish) to +1.0 (strong bullish)
    confidence: float  # 0.0 to 1.0; how much to trust this residual
    component_breakdown: Dict[str, float]  # Per-component contribution to residual_score
    scout_signals: ScoutSignals
    linguist_scores: LinguistScores
    aggregation_weights: Dict[str, float]  # Weights used in this aggregation
    timestamp: datetime


class SentimentAggregator:
    """Combines Scout signals and Linguist scores into a composite SentimentResidual."""

    # Default weights; calibrated via daily post-mortem feedback (sentinel/judge/resolver.py)
    DEFAULT_WEIGHTS = {
        "price_momentum": 0.15,
        "volume_spike": 0.08,
        "news_tone": 0.12,
        "reddit_sentiment": 0.10,
        "hackernews_sentiment": 0.08,
        "sec_filing_sentiment": 0.12,
        "certainty_amplifier": 0.15,  # Multiplier on overall score based on certainty
        "hesitation_damper": -0.08,  # Subtractive: high hesitation lowers confidence
        "linguistic_drift": 0.10,
        "regulatory_whispers_penalty": -0.06,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        weights_db_path: Optional[Path] = None,
    ):
        """
        Initialize aggregator with optional custom weights or DB-loaded calibration.

        Args:
            weights: Optional dict overriding DEFAULT_WEIGHTS.
            weights_db_path: Optional path to SQLite DB storing calibrated weights per ticker.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.weights_db_path = weights_db_path
        self._load_calibrated_weights()

    def _load_calibrated_weights(self) -> None:
        """Load per-ticker calibrated weights from SQLite if available."""
        if not self.weights_db_path or not self.weights_db_path.exists():
            return

        try:
            conn = sqlite3.connect(self.weights_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT ticker, weights_json, calibrated_at FROM weight_calibration "
                "ORDER BY calibrated_at DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                _, weights_json, _ = row
                self.calibrated_weights = json.loads(weights_json)
            conn.close()
        except Exception as e:
            print(f"Warning: failed to load calibrated weights from {self.weights_db_path}: {e}")
            self.calibrated_weights = None

    def aggregate(
        self,
        ticker: str,
        scout_signals: ScoutSignals,
        linguist_scores: LinguistScores,
    ) -> SentimentResidual:
        """
        Aggregate Scout and Linguist signals into a unified SentimentResidual score.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL").
            scout_signals: Raw signals from Scout ingestion pipeline.
            linguist_scores: Linguistic analysis scores from Linguist pipeline.

        Returns:
            SentimentResidual with composite score, confidence, and component breakdown.
        """
        # Retrieve ticker-specific weights if available, else use global defaults
        weights = self._get_weights_for_ticker(ticker)

        # Build component scores
        component_scores = {}

        # Scout signal contributions
        component_scores["price_momentum"] = (
            scout_signals.price_momentum * weights["price_momentum"]
        )
        component_scores["volume_spike"] = (
            scout_signals.volume_spike * weights["volume_spike"]
        )
        component_scores["news_tone"] = scout_signals.news_tone * weights["news_tone"]
        component_scores["reddit_sentiment"] = (
            scout_signals.reddit_sentiment * weights["reddit_sentiment"]
        )
        component_scores["hackernews_sentiment"] = (
            scout_signals.hackernews_sentiment * weights["hackernews_sentiment"]
        )

        # SEC filing sentiment (optional; weight only if present)
        if scout_signals.sec_filing_sentiment is not None:
            component_scores["sec_filing_sentiment"] = (
                scout_signals.sec_filing_sentiment * weights["sec_filing_sentiment"]
            )
        else:
            component_scores["sec_filing_sentiment"] = 0.0

        # Linguist signal contributions
        component_scores["linguistic_drift"] = (
            linguist_scores.linguistic_drift * weights["linguistic_drift"]
        )

        # Regulatory whispers as a penalty
        component_scores["regulatory_whispers_penalty"] = (
            linguist_scores.regulatory_whispers * weights["regulatory_whispers_penalty"]
        )

        # Sum raw components
        raw_residual = sum(component_scores.values())

        # Apply certainty amplifier: high certainty boosts signal, low certainty dampens it
        certainty_multiplier = 1.0 + (
            (linguist_scores.certainty_score - 0.5) * weights["certainty_amplifier"]
        )
        adjusted_residual = raw_residual * certainty_multiplier

        # Apply hesitation damper: subtracts confidence if hesitation is high
        hesitation_penalty = linguist_scores.hesitation_index
