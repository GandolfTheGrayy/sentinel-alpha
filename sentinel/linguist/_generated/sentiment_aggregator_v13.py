"""
Sentiment Aggregator for Sentinel Sentiment Engine.

Combines Scout-sourced signals (price momentum, news volume/sentiment,
SEC filing urgency) with Linguist-computed scores (certainty, linguistic drift,
regulatory whispers) into a unified SentimentResidual composite metric.

SentimentResidual = weighted sum of normalized Scout signals + Linguist scores,
calibrated by confidence and historical volatility. Output range: [-1.0, 1.0],
where negative = bearish, positive = bullish.

Used by Judge.predictor to bias final price-movement predictions.
"""

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScoutSignals:
    """Raw signals from Scout ingestion layer."""

    ticker: str
    timestamp: datetime
    price_momentum: float  # normalized [-1, 1]: (price - MA50) / MA50
    news_volume: float  # normalized [0, 1]: count / max_observed
    news_sentiment: float  # [-1, 1]: aggregate polarity from headlines
    sec_filing_urgency: float  # [0, 1]: 1 = 8-K, 0.5 = 10-Q, 0 = other
    reddit_mentions: float  # normalized [0, 1]
    reddit_sentiment: float  # [-1, 1]: aggregate sentiment from threads


@dataclass
class LinguistScores:
    """Computed scores from Linguist reasoning layer."""

    ticker: str
    timestamp: datetime
    certainty: float  # [0, 1]: LLM confidence in text interpretation
    linguistic_drift: float  # [-1, 1]: negative = tone shift toward negative
    regulatory_whispers: float  # [0, 1]: urgency/risk signals in SEC text
    management_tone: float  # [-1, 1]: sentiment of earnings call transcript


@dataclass
class SentimentResidual:
    """Composite sentiment metric combining Scout + Linguist."""

    ticker: str
    timestamp: datetime
    residual_score: float  # [-1, 1]: final aggregated sentiment
    scout_contribution: float  # [0, 1]: weight of Scout signals
    linguist_contribution: float  # [0, 1]: weight of Linguist scores
    confidence: float  # [0, 1]: blend of input certainties
    component_breakdown: dict  # keys: "momentum", "news", "sec", "reddit", "drift", "whispers"


class SentimentAggregator:
    """
    Aggregates Scout and Linguist signals into SentimentResidual scores.
    
    Maintains historical residuals in SQLite for RAG corpus and post-mortem calibration.
    """

    def __init__(self, db_path: str = "sentinel_residuals.db"):
        """Initialize aggregator with SQLite backend for historical storage."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create residuals table if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_residuals (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    residual_score REAL,
                    scout_contribution REAL,
                    linguist_contribution REAL,
                    confidence REAL,
                    component_breakdown TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON sentiment_residuals (ticker, timestamp)
                """
            )
            conn.commit()

    def aggregate(
        self,
        scout: ScoutSignals,
        linguist: LinguistScores,
        scout_weight: float = 0.6,
        linguist_weight: float = 0.4,
    ) -> SentimentResidual:
        """
        Compute SentimentResidual from Scout and Linguist inputs.

        Args:
            scout: Raw signals from data ingestion layer.
            linguist: Computed scores from LLM reasoning layer.
            scout_weight: Relative weight [0, 1] for Scout contribution.
            linguist_weight: Relative weight [0, 1] for Linguist contribution.

        Returns:
            SentimentResidual with composite score and component breakdown.
        """
        assert scout.ticker == linguist.ticker
        assert scout.timestamp == linguist.timestamp

        # Normalize weights to sum to 1.
        total_weight = scout_weight + linguist_weight
        scout_w = scout_weight / total_weight
        linguist_w = linguist_weight / total_weight

        # Scout sub-scores (already normalized to [-1, 1] or [0, 1]).
        scout_aggregate = (
            0.3 * scout.price_momentum
            + 0.2 * scout.news_sentiment
            + 0.15 * (2 * scout.news_volume - 1)  # convert [0,1] to [-1,1]
            + 0.15 * scout.sec_filing_urgency  # [0,1] → bias positive for urgency
            + 0.2 * scout.reddit_sentiment
        )

        # Linguist sub-scores (already normalized to appropriate ranges).
        linguist_aggregate = (
            0.3 * (2 * linguist.certainty - 1)  # [0,1] → [-1,1]
            + 0.25 * linguist.linguistic_drift
            + 0.25 * (2 * linguist.regulatory_whispers - 1)  # [0,1] → [-1,1]
            + 0.2 * linguist.management_tone
        )

        # Composite residual, clipped to [-1, 1].
        residual_score = np.clip(
            scout_w * scout_aggregate + linguist_w * linguist_aggregate,
            -1.0,
            1.0,
        )

        # Aggregate confidence: mean certainty from Linguist, scaled by Scout signal strength.
        confidence = np.clip(
            0.7 * linguist.certainty + 0.3 * (1 - abs(scout_aggregate)),
            0.0,
            1.0,
        )

        component_breakdown = {
            "momentum": scout.price_momentum,
            "news_sentiment": scout.news_sentiment,
            "news_volume": scout.news_volume,
            "sec_urgency": scout.sec_filing_urgency,
            "reddit_sentiment": scout.reddit_sentiment,
            "certainty": linguist.certainty,
            "linguistic_drift": linguist.linguistic_drift,
            "regulatory_whispers": linguist.regulatory_whispers,
            "management_tone": linguist.management_tone,
        }

        residual = SentimentResidual(
            ticker=scout.ticker,
            timestamp=scout.timestamp,
            residual_score=residual_score,
            scout_contribution=scout_w,
            linguist_contribution=linguist_w,
            confidence=confidence,
            component_breakdown=component_breakdown,
        )

        self.store(residual)
        return residual

    def store(self, residual: SentimentResidual) -> None:
        """Persist SentimentResidual to SQLite for historical analysis."""
        import json

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sentiment_residuals
                (ticker, timestamp, residual_score, scout_contribution,
                 linguist_contribution, confidence, component_breakdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    residual.ticker,
                    residual.timestamp.isoformat(),
                    residual.residual_score,
                    residual.scout_contribution,
                    residual.linguist_contribution,
                    residual.confidence,
                    json.dumps(residual.component_breakdown),
                ),
            )
