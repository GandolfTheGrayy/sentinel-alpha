"""
Sentiment Aggregator for Sentinel Sentiment Engine.

This module combines Scout-sourced signals (Reddit sentiment, HN developer mood,
GitHub health, price momentum) with Linguist-computed scores (certainty analysis,
linguistic drift, regulatory whispers) into a unified SentimentResidual score.

The aggregator applies a weighted formula calibrated against historical market moves,
producing a composite sentiment signal (range: -1.0 to +1.0) that feeds into
the Judge agent's daily post-mortem and prediction pipeline.

Role in Sentinel:
  - Consumes raw Scout signals and Linguist linguistic scores
  - Applies domain-weighted aggregation formula
  - Outputs SentimentResidual with confidence bounds
  - Feeds prediction input to Judge calibration loop
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import sqlite3
import statistics


@dataclass
class ScoutSignal:
    """Raw signal from Scout ingestion layer."""
    source: str  # "reddit", "hackernews", "github", "price_momentum"
    ticker: str
    timestamp: datetime
    raw_score: float  # Typically -1.0 to +1.0
    volume: int  # Number of mentions, commits, etc.
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class LinguistScore:
    """Linguistic analysis score from Linguist reasoning layer."""
    metric: str  # "certainty", "linguistic_drift", "regulatory_whispers", "earnings_tone"
    ticker: str
    timestamp: datetime
    score: float  # -1.0 (bearish/uncertain) to +1.0 (bullish/certain)
    confidence: float  # 0.0 to 1.0: how sure we are of this metric
    reasoning: str


@dataclass
class SentimentResidual:
    """Composite sentiment signal with provenance."""
    ticker: str
    timestamp: datetime
    composite_score: float  # -1.0 to +1.0
    confidence_lower: float  # 95% CI lower bound
    confidence_upper: float  # 95% CI upper bound
    component_breakdown: Dict[str, float]  # {source: weighted_contribution}
    signal_count: int  # Total signals aggregated
    reasoning_summary: str


class SentimentAggregator:
    """
    Combines Scout signals and Linguist scores into SentimentResidual.
    
    Weighting schema (calibrated via Judge post-mortems):
      - Reddit sentiment: 0.20 (high volume but noise)
      - HN developer mood: 0.15 (lower volume, tech-savvy)
      - GitHub health: 0.10 (structural signal, slow-moving)
      - Price momentum (24h): 0.10 (contrarian indicator)
      - Certainty score: 0.20 (high confidence LLM reasoning)
      - Linguistic drift: 0.15 (tone shift detection)
      - Regulatory whispers: 0.10 (hedging language in filings)
    
    Total: 1.00 (weights sum to 1.0)
    """

    DEFAULT_WEIGHTS = {
        "reddit": 0.20,
        "hackernews": 0.15,
        "github": 0.10,
        "price_momentum": 0.10,
        "certainty": 0.20,
        "linguistic_drift": 0.15,
        "regulatory_whispers": 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        db_path: Optional[str] = None,
    ):
        """
        Initialize aggregator with optional custom weights and persistence.
        
        Args:
            weights: Custom weight overrides. Must sum to ~1.0. Defaults to DEFAULT_WEIGHTS.
            db_path: Optional SQLite path for logging aggregation history.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        self.db_path = db_path
        if db_path:
            self._init_db()

    def _validate_weights(self) -> None:
        """Ensure weights sum to approximately 1.0."""
        total = sum(self.weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {total}. "
                f"Weights: {self.weights}"
            )

    def _init_db(self) -> None:
        """Create SQLite schema for sentiment history if db_path provided."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_residuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                composite_score REAL NOT NULL,
                confidence_lower REAL NOT NULL,
                confidence_upper REAL NOT NULL,
                component_breakdown TEXT NOT NULL,
                signal_count INTEGER NOT NULL,
                reasoning_summary TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def aggregate(
        self,
        scout_signals: List[ScoutSignal],
        linguist_scores: List[LinguistScore],
        ticker: str,
    ) -> SentimentResidual:
        """
        Compute composite SentimentResidual from Scout + Linguist inputs.
        
        Args:
            scout_signals: List of raw Scout-sourced signals (Reddit, HN, GitHub, price).
            linguist_scores: List of Linguist-computed linguistic metrics.
            ticker: Stock ticker symbol.
        
        Returns:
            SentimentResidual with composite score, confidence bounds, and breakdown.
        
        Raises:
            ValueError: If no signals or scores provided, or timestamp mismatch.
        """
        if not scout_signals and not linguist_scores:
            raise ValueError("Must provide at least one Scout signal or Linguist score")

        timestamp = self._infer_timestamp(scout_signals, linguist_scores)
        
        # Aggregate Scout signals by source
        scout_agg = self._aggregate_scout_signals(scout_signals)
        
        # Aggregate Linguist scores by metric
        linguist_agg = self._aggregate_linguist_scores(linguist_scores)
        
        # Weighted sum
        component_breakdown = {}
        weighted_sum = 0.0
        total_weight_applied = 0.0

        # Apply Scout signal weights
        for source, (score, volume) in scout_agg.items():
            weight = self.weights.get(source, 0.0)
            if weight > 0:
                component_breakdown[f"scout_{source}"] = score * weight
                weighted_sum += score * weight
                total_weight_applied += weight

        # Apply Linguist score weights
        for metric, (score, confidence) in linguist_agg.items():
            weight = self.weights.get(metric, 0.0)
            if weight > 0:
                # Dampen linguistic scores by their confidence
                damped_score = score * confidence
                component_breakdown[f"linguist_{metric}"] = damped_score * weight
                weighted_sum += damped_score * weight
                total_weight_applied += weight

        # Normalize if we didn't use all weights (missing signals)
        if total_weight_applied > 0:
            composite_score = weighted_sum / total_weight_applied
        else:
            composite_score = 0.0

        # Clamp to [-1, 1]
        composite_score = max(-1.0, min(1.0, composite_score))

        # Compute confidence interval
        all_scores = [sig.raw_score for sig in scout_signals] + \
                     [ling.score for ling in linguist_scores]
        ci_lower, ci_upper = self._compute_confidence_interval(all_scores, composite_score)

        # Generate reasoning summary
        reasoning = self._generate_reasoning(
            scout_agg, linguist_agg, composite_score, len
