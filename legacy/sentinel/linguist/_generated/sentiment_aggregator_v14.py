"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout-sourced signals (price momentum, news volume, social sentiment)
with Linguist-computed scores (certainty, linguistic drift, regulatory whispers)
into a unified SentimentResidual composite score.

The SentimentResidual represents net bullish/bearish bias: positive values
indicate upside conviction, negative values indicate downside risk, and
magnitude reflects confidence. This score feeds directly into Judge's
per-ticker prediction pipeline.

Formula:
  SentimentResidual = (
      w_news * normalized(news_sentiment) +
      w_certainty * normalized(linguist_certainty) +
      w_momentum * normalized(price_momentum) +
      w_social * normalized(social_sentiment) +
      w_drift * normalized(linguistic_drift)
  )
where weights sum to 1.0 and normalization uses z-score via historical quartiles.

Weights are configurable via YAML and auto-tuned by Judge's post-mortem
calibration loop.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3
import numpy as np


@dataclass
class SentimentSignals:
    """Container for raw Scout and Linguist signals for a single ticker."""
    ticker: str
    timestamp: datetime
    news_sentiment: float  # [-1.0, 1.0], mean polarity of recent headlines
    news_volume: int  # count of relevant articles in last 24h
    linguist_certainty: float  # [0.0, 1.0], Claude confidence in direction
    social_sentiment: float  # [-1.0, 1.0], mean Reddit/HN upvote ratio
    price_momentum: float  # [-1.0, 1.0], (current - 20d_ma) / 20d_ma, clamped
    linguistic_drift: float  # [-1.0, 1.0], tone shift vs. 90d baseline
    regulatory_whispers: float  # [0.0, 1.0], anomaly flag for SEC language


@dataclass
class SentimentResidual:
    """Composite sentiment score with component breakdown and confidence band."""
    ticker: str
    timestamp: datetime
    residual: float  # [-1.0, 1.0], net sentiment direction
    confidence: float  # [0.0, 1.0], magnitude of conviction
    component_scores: Dict[str, float]  # breakdown by signal type
    signal_counts: Dict[str, int]  # count of data points per signal
    notes: Optional[str] = None


class SentimentAggregator:
    """
    Combines Scout and Linguist signals into a SentimentResidual composite.
    
    Implements z-score normalization, configurable weighting, and historical
    calibration via SQLite event log. Designed for daily batch or live streaming.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        db_path: str = ":memory:"
    ) -> None:
        """
        Initialize aggregator with weights and optional SQLite backend.
        
        Args:
            weights: Signal weight dict with keys {news_sentiment, certainty,
                    momentum, social_sentiment, drift}. Defaults to uniform.
            db_path: Path to SQLite DB for historical calibration. Use ":memory:"
                    for ephemeral sessions.
        """
        self.weights = weights or {
            "news_sentiment": 0.25,
            "certainty": 0.30,
            "momentum": 0.20,
            "social_sentiment": 0.15,
            "drift": 0.10,
        }
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {weight_sum}")
        
        self.db_path = db_path
        self._init_db()
        self._load_calibration()
    
    def _init_db(self) -> None:
        """Create SQLite schema for signal history and calibration."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                residual REAL,
                confidence REAL,
                news_sentiment REAL,
                certainty REAL,
                momentum REAL,
                social_sentiment REAL,
                drift REAL,
                UNIQUE(ticker, timestamp)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS calibration (
                signal TEXT PRIMARY KEY,
                mean REAL NOT NULL,
                stddev REAL NOT NULL,
                q1 REAL NOT NULL,
                q3 REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_calibration(self) -> None:
        """Load historical mean/stddev for z-score normalization."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT signal, mean, stddev, q1, q3 FROM calibration")
        rows = c.fetchall()
        conn.close()
        
        self.calibration = {
            row[0]: {
                "mean": row[1],
                "stddev": row[2],
                "q1": row[3],
                "q3": row[4],
            }
            for row in rows
        }
    
    def aggregate(self, signals: SentimentSignals) -> SentimentResidual:
        """
        Compute composite SentimentResidual from raw Scout/Linguist signals.
        
        Returns SentimentResidual with residual in [-1, 1] and confidence in [0, 1].
        """
        # Normalize each signal via z-score (or clamp if no calibration yet).
        norm_news = self._normalize("news_sentiment", signals.news_sentiment)
        norm_cert = self._normalize("certainty", signals.linguist_certainty)
        norm_mom = self._normalize("momentum", signals.price_momentum)
        norm_social = self._normalize("social_sentiment", signals.social_sentiment)
        norm_drift = self._normalize("drift", signals.linguistic_drift)
        
        # Compute weighted residual.
        residual = (
            self.weights["news_sentiment"] * norm_news +
            self.weights["certainty"] * norm_cert +
            self.weights["momentum"] * norm_mom +
            self.weights["social_sentiment"] * norm_social +
            self.weights["drift"] * norm_drift
        )
        
        # Clamp to [-1, 1].
        residual = np.clip(residual, -1.0, 1.0)
        
        # Confidence = absolute value (magnitude of conviction).
        confidence = abs(residual)
        
        # Component breakdown for audit trail.
        component_scores = {
            "news_sentiment": norm_news,
            "certainty": norm_cert,
            "momentum": norm_mom,
            "social_sentiment": norm_social,
            "drift": norm_drift,
        }
        
        # Signal counts (for weighting later in Judge).
        signal_counts = {
            "news_volume": signals.news_volume,
            "regulatory_whispers": int(signals.regulatory_whispers > 0.5),
        }
        
        result = SentimentResidual(
            ticker=signals.ticker,
            timestamp=signals.timestamp,
            residual=residual,
            confidence=confidence,
            component_scores=component_scores,
            signal_counts=signal_counts,
            notes=self._generate_notes(signals, component_scores),
        )
        
        # Log to DB for post-mortem calibration.
        self._log_signal(signals, result)
        
        return result
    
    def _normalize(self, signal_name
