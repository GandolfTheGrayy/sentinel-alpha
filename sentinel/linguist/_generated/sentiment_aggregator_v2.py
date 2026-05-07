"""
Sentiment Aggregator for Sentinel Sentiment Engine.

This module combines Scout signals (price momentum, news sentiment, SEC filing tone)
and Linguist scores (certainty, hesitation, regulatory whispers) into a composite
SentimentResidual score. The residual represents the net sentiment signal after
accounting for baseline market factors, weighted by confidence in each signal source.

Used by judge/predictor.py to synthesize multi-modal sentiment into directional bias.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional


@dataclass
class SentimentSignal:
    """A single sentiment signal from Scout or Linguist with metadata."""
    source: str
    value: float
    confidence: float
    timestamp: int


@dataclass
class SentimentResidual:
    """Composite sentiment score combining weighted signals."""
    ticker: str
    residual_score: float
    component_breakdown: dict
    aggregate_confidence: float
    signal_count: int


def compute_weighted_sentiment(
    signals: list[SentimentSignal],
) -> float:
    """
    Compute weighted average sentiment from multiple signals.
    
    Weights each signal by its confidence; returns [-1.0, 1.0] where
    negative = bearish, positive = bullish, 0 = neutral.
    """
    if not signals:
        return 0.0
    
    total_weight = sum(s.confidence for s in signals)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(s.value * s.confidence for s in signals)
    return weighted_sum / total_weight


def aggregate_sentiment_residual(
    ticker: str,
    price_momentum: Optional[float] = None,
    news_sentiment: Optional[float] = None,
    sec_tone: Optional[float] = None,
    certainty_score: Optional[float] = None,
    hesitation_penalty: Optional[float] = None,
    regulatory_whisper: Optional[float] = None,
    confidence_overrides: Optional[dict] = None,
) -> SentimentResidual:
    """
    Aggregate Scout and Linguist signals into a composite SentimentResidual.
    
    Each input (price_momentum, news_sentiment, etc.) is optional and normalized
    to [-1, 1]. Confidence overrides allow caller to weight certain sources higher.
    Returns a SentimentResidual with breakdown of component contributions.
    """
    if confidence_overrides is None:
        confidence_overrides = {}
    
    signals = []
    
    # Default confidence weights (caller can override)
    default_weights = {
        "price_momentum": 0.25,
        "news_sentiment": 0.20,
        "sec_tone": 0.15,
        "certainty_score": 0.20,
        "hesitation_penalty": 0.10,
        "regulatory_whisper": 0.10,
    }
    
    # Add price momentum signal
    if price_momentum is not None:
        conf = confidence_overrides.get("price_momentum", default_weights["price_momentum"])
        signals.append(SentimentSignal(
            source="price_momentum",
            value=max(-1.0, min(1.0, price_momentum)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Add news sentiment signal
    if news_sentiment is not None:
        conf = confidence_overrides.get("news_sentiment", default_weights["news_sentiment"])
        signals.append(SentimentSignal(
            source="news_sentiment",
            value=max(-1.0, min(1.0, news_sentiment)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Add SEC tone signal
    if sec_tone is not None:
        conf = confidence_overrides.get("sec_tone", default_weights["sec_tone"])
        signals.append(SentimentSignal(
            source="sec_tone",
            value=max(-1.0, min(1.0, sec_tone)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Add Linguist certainty score (higher certainty = higher confidence in prediction)
    if certainty_score is not None:
        conf = confidence_overrides.get("certainty_score", default_weights["certainty_score"])
        signals.append(SentimentSignal(
            source="certainty_score",
            value=max(-1.0, min(1.0, certainty_score)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Add hesitation penalty (negative confidence dampener)
    if hesitation_penalty is not None:
        conf = confidence_overrides.get("hesitation_penalty", default_weights["hesitation_penalty"])
        signals.append(SentimentSignal(
            source="hesitation_penalty",
            value=max(-1.0, min(1.0, -hesitation_penalty)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Add regulatory whisper signal
    if regulatory_whisper is not None:
        conf = confidence_overrides.get("regulatory_whisper", default_weights["regulatory_whisper"])
        signals.append(SentimentSignal(
            source="regulatory_whisper",
            value=max(-1.0, min(1.0, regulatory_whisper)),
            confidence=max(0.0, min(1.0, conf)),
            timestamp=0,
        ))
    
    # Compute weighted sentiment
    residual_score = compute_weighted_sentiment(signals)
    
    # Build component breakdown
    component_breakdown = {s.source: s.value for s in signals}
    
    # Compute aggregate confidence as weighted mean
    total_confidence = sum(s.confidence for s in signals)
    aggregate_confidence = (
        total_confidence / len(signals)
        if len(signals) > 0
        else 0.0
    )
    
    return SentimentResidual(
        ticker=ticker,
        residual_score=residual_score,
        component_breakdown=component_breakdown,
        aggregate_confidence=aggregate_confidence,
        signal_count=len(signals),
    )


def store_sentiment_residual(
    db_path: str,
    residual: SentimentResidual,
    timestamp: int,
) -> None:
    """
    Persist a SentimentResidual to SQLite for historical tracking.
    
    Creates table if missing. Stores residual score, components, and confidence.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_residuals (
            ticker TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            residual_score REAL NOT NULL,
            aggregate_confidence REAL NOT NULL,
            signal_count INTEGER NOT NULL,
            component_breakdown TEXT NOT NULL,
            PRIMARY KEY (ticker, timestamp)
        )
    """)
    
    import json
    cursor.execute("""
        INSERT OR REPLACE INTO sentiment_residuals
        (ticker, timestamp, residual_score, aggregate_confidence, signal_count, component_breakdown)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        residual.ticker,
        timestamp,
        residual.residual_score,
        residual.aggregate_confidence,
        residual.signal_count,
        json.dumps(residual.component_breakdown),
    ))
    
    conn.commit()
    conn.close()


def retrieve_sentiment_residual(
    db_path: str,
    ticker: str,
    timestamp: int,
) -> Optional[SentimentResidual]:
    """
    Retrieve a stored SentimentResidual by ticker and timestamp.
    
    Returns None if not found.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT residual
