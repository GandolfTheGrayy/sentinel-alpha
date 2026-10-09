"""
Sentiment Aggregator for Sentinel Sentiment Engine.

This module combines Scout-sourced signals (price momentum, news sentiment,
Reddit/HN community health, GitHub metrics) and Linguist-computed scores
(certainty, linguistic drift, regulatory whispers) into a unified
SentimentResidual composite score with weighted formula.

The SentimentResidual is a [-1.0, 1.0] float encoding the strength and
direction of bullish vs. bearish signals, suitable for downstream prediction
and backtesting. Weights are calibrated empirically and stored in YAML config.
"""

import json
import sqlite3
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class ScoutSignals:
    """Container for raw Scout measurements."""
    
    ticker: str
    timestamp: datetime
    price_momentum: float  # [-1, 1]: negative = down, positive = up
    news_sentiment: float  # [-1, 1]: negative = bearish, positive = bullish
    reddit_engagement: float  # [0, 1]: sentiment from subreddit community
    hn_mentions: int  # raw count of Hacker News mentions (positive = relevance)
    github_health: float  # [0, 1]: repo stars/commits trend (if applicable)
    
    def to_dict(self) -> Dict:
        """Serialize to dict, converting datetime to ISO string."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class LinguistScores:
    """Container for Linguist-computed sentiment metrics."""
    
    ticker: str
    timestamp: datetime
    certainty_score: float  # [0, 1]: confidence in bullish/bearish signal
    linguistic_drift: float  # [-1, 1]: tone shift from baseline
    regulatory_whispers: float  # [-1, 1]: regulatory risk signal
    
    def to_dict(self) -> Dict:
        """Serialize to dict, converting datetime to ISO string."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class SentimentResidual:
    """Composite sentiment score from aggregation."""
    
    ticker: str
    timestamp: datetime
    residual_score: float  # [-1, 1]: final bullish/bearish signal
    component_scores: Dict[str, float]  # breakdown: {signal_name: weight*value}
    aggregation_method: str  # "weighted_mean" | "confidence_weighted"
    confidence: float  # [0, 1]: overall confidence in the residual
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class SentimentAggregator:
    """
    Combines Scout signals and Linguist scores into SentimentResidual.
    
    Weights are loaded from YAML config; aggregation formula is:
      residual_score = sum(weight[i] * normalized_signal[i]) / sum(weight[i])
    
    Normalization ensures all signals are on [-1, 1] scale before weighting.
    """
    
    DEFAULT_WEIGHTS = {
        'price_momentum': 0.25,
        'news_sentiment': 0.20,
        'certainty_score': 0.20,
        'linguistic_drift': 0.15,
        'regulatory_whispers': -0.10,  # negative weight = risk factor
        'reddit_engagement': 0.10,
        'github_health': 0.05,
        'hn_mentions': 0.05,
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize aggregator with optional YAML config override.
        
        Args:
            config_path: Path to YAML file with custom weights.
        """
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
                self.weights.update(cfg.get('sentiment_weights', {}))
    
    def _normalize_signal(self, value: float, signal_type: str) -> float:
        """
        Normalize signal to [-1, 1] range based on type.
        
        Args:
            value: Raw signal value.
            signal_type: Name of signal (e.g., 'price_momentum', 'hn_mentions').
        
        Returns:
            Normalized value in [-1, 1].
        """
        if signal_type in ('price_momentum', 'news_sentiment', 'linguistic_drift',
                          'regulatory_whispers'):
            # Already expected to be in [-1, 1]
            return np.clip(value, -1.0, 1.0)
        elif signal_type in ('certainty_score', 'reddit_engagement', 'github_health'):
            # [0, 1] → [-1, 1] by scaling: 2*x - 1
            clipped = np.clip(value, 0.0, 1.0)
            return 2.0 * clipped - 1.0
        elif signal_type == 'hn_mentions':
            # Raw count → sigmoid-like saturation at [0, 50] mentions
            # Clamp and scale to [-1, 1]
            normalized = min(value / 50.0, 1.0)
            return 2.0 * normalized - 1.0
        else:
            return np.clip(value, -1.0, 1.0)
    
    def aggregate(self, scout_signals: ScoutSignals,
                 linguist_scores: LinguistScores) -> SentimentResidual:
        """
        Combine Scout and Linguist measurements into SentimentResidual.
        
        Args:
            scout_signals: Raw Scout measurements.
            linguist_scores: Linguist-computed scores.
        
        Returns:
            SentimentResidual with composite score and component breakdown.
        """
        if scout_signals.ticker != linguist_scores.ticker:
            raise ValueError("Ticker mismatch between Scout and Linguist inputs.")
        
        # Collect all signals and normalize.
        signals_dict = {
            'price_momentum': scout_signals.price_momentum,
            'news_sentiment': scout_signals.news_sentiment,
            'reddit_engagement': scout_signals.reddit_engagement,
            'hn_mentions': scout_signals.hn_mentions,
            'github_health': scout_signals.github_health,
            'certainty_score': linguist_scores.certainty_score,
            'linguistic_drift': linguist_scores.linguistic_drift,
            'regulatory_whispers': linguist_scores.regulatory_whispers,
        }
        
        normalized = {}
        for sig_name, sig_value in signals_dict.items():
            normalized[sig_name] = self._normalize_signal(sig_value, sig_name)
        
        # Weighted sum.
        weighted_sum = 0.0
        weight_sum = 0.0
        component_scores = {}
        
        for sig_name, norm_value in normalized.items():
            weight = self.weights.get(sig_name, 0.0)
            weighted_contribution = weight * norm_value
            weighted_sum += weighted_contribution
            weight_sum += abs(weight)  # Use absolute for denominator
            component_scores[sig_name] = weighted_contribution
        
        # Normalize by total weight magnitude.
        residual_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        residual_score = np.clip(residual_score, -1.0, 1.0)
        
        # Confidence: average certainty and regulatory confidence.
        confidence = (
            linguist_scores.certainty_score * 0.7 +
            (1.0 - abs(linguist_scores.regulatory
