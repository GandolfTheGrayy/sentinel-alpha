"""
Sentiment Aggregator for Sentinel — combines Scout signals (price, news, filings,
social) and Linguist scores (certainty, drift, regulatory whispers) into a
composite SentimentResidual score via weighted formula.

This module synthesizes multi-source sentiment into a single directional signal
([-1.0, +1.0]) with confidence bounds, enabling Judge to calibrate predictions.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SentimentSignal:
    """Single sentiment input from Scout or Linguist pipeline."""
    source: str  # e.g. "news", "reddit", "sec_8k", "certainty_score"
    ticker: str
    value: float  # normalized to [-1.0, +1.0]
    weight: float  # importance multiplier
    timestamp: int  # unix seconds
    confidence: float  # [0.0, 1.0] how reliable this signal is


@dataclass
class SentimentResidual:
    """Composite sentiment output with bounds and attribution."""
    ticker: str
    residual: float  # weighted mean, [-1.0, +1.0]
    lower_bound: float  # pessimistic scenario
    upper_bound: float  # optimistic scenario
    confidence: float  # overall signal reliability
    component_count: int  # how many signals were aggregated
    leading_drivers: list[tuple[str, float]]  # (source, contribution) sorted by abs impact
    timestamp: int  # when aggregation occurred


DEFAULT_WEIGHTS = {
    "certainty_score": 0.25,  # Linguist's linguistic confidence
    "sec_8k": 0.20,  # Material events
    "sec_10q": 0.15,  # Quarterly fundamentals
    "news_headline": 0.15,  # Major press
    "reddit_wsb": 0.10,  # Retail chatter
    "github_stars": 0.08,  # Developer signals (tech only)
    "regulatory_whisper": 0.07,  # Drift detector flagged language
}


def aggregate_signals(
    signals: list[SentimentSignal],
    weights: Optional[dict[str, float]] = None,
) -> SentimentResidual:
    """
    Combine multiple SentimentSignals into a single SentimentResidual score.
    
    Uses weighted mean with confidence-adjusted contributions. Signals with
    higher confidence get amplified; lower-confidence signals dampen. Computes
    pessimistic (lower) and optimistic (upper) bounds via percentile sampling.
    """
    if not signals:
        return SentimentResidual(
            ticker="UNKNOWN",
            residual=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            confidence=0.0,
            component_count=0,
            leading_drivers=[],
            timestamp=0,
        )

    w = weights or DEFAULT_WEIGHTS
    ticker = signals[0].ticker

    # Normalize weights to sum to 1.0
    active_sources = {s.source for s in signals}
    active_weight_sum = sum(w.get(src, 0.01) for src in active_sources)
    normalized_weights = {
        src: w.get(src, 0.01) / active_weight_sum for src in active_sources
    }

    # Confidence-weighted contributions
    contributions = []
    total_weight = 0.0
    for signal in signals:
        # Scale signal by its confidence and normalized source weight
        adj_weight = (
            signal.weight
            * normalized_weights.get(signal.source, 0.01)
            * signal.confidence
        )
        contribution = signal.value * adj_weight
        contributions.append(contribution)
        total_weight += adj_weight

    if total_weight < 1e-9:
        total_weight = 1.0

    # Weighted residual
    residual = np.mean(contributions) if contributions else 0.0
    residual = np.clip(residual, -1.0, 1.0)

    # Confidence: average signal confidence, attenuated by component diversity
    avg_confidence = np.mean([s.confidence for s in signals])
    source_diversity = len(active_sources) / len(DEFAULT_WEIGHTS)
    combined_confidence = avg_confidence * (0.5 + 0.5 * source_diversity)
    combined_confidence = np.clip(combined_confidence, 0.0, 1.0)

    # Bounds: pessimistic (5th percentile) and optimistic (95th percentile) of signals
    signal_values = [s.value for s in signals]
    lower_bound = np.percentile(signal_values, 5)
    upper_bound = np.percentile(signal_values, 95)

    # Leading drivers: rank signals by absolute contribution
    driver_impact = [
        (signals[i].source, contributions[i]) for i in range(len(signals))
    ]
    driver_impact.sort(key=lambda x: abs(x[1]), reverse=True)
    top_drivers = driver_impact[:3]  # Top 3 contributors

    return SentimentResidual(
        ticker=ticker,
        residual=float(residual),
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
        confidence=float(combined_confidence),
        component_count=len(signals),
        leading_drivers=[(src, float(imp)) for src, imp in top_drivers],
        timestamp=int(signals[0].timestamp),
    )


def store_residual(
    db_path: str,
    residual: SentimentResidual,
) -> None:
    """
    Persist SentimentResidual to SQLite for historical tracking and backtest.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_residuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            residual REAL NOT NULL,
            lower_bound REAL NOT NULL,
            upper_bound REAL NOT NULL,
            confidence REAL NOT NULL,
            component_count INTEGER NOT NULL,
            leading_drivers TEXT,
            timestamp INTEGER NOT NULL
        )
        """
    )

    drivers_str = ";".join(
        f"{src}:{imp:.4f}" for src, imp in residual.leading_drivers
    )

    cursor.execute(
        """
        INSERT INTO sentiment_residuals
        (ticker, residual, lower_bound, upper_bound, confidence,
         component_count, leading_drivers, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            residual.ticker,
            residual.residual,
            residual.lower_bound,
            residual.upper_bound,
            residual.confidence,
            residual.component_count,
            drivers_str,
            residual.timestamp,
        ),
    )

    conn.commit()
    conn.close()


def retrieve_residuals(
    db_path: str,
    ticker: str,
    limit: int = 30,
) -> list[SentimentResidual]:
    """
    Retrieve recent SentimentResiduals for a ticker to track sentiment drift.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT ticker, residual, lower_bound, upper_bound, confidence,
               component_count, leading_drivers, timestamp
        FROM sentiment_residuals
        WHERE ticker = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (ticker, limit),
    )

    rows = cursor.fetchall()
    conn.close()

    residuals = []
    for row in rows:
        ticker_val, residual, lower, upper, conf, count, drivers_str, ts = row
        drivers = (
            [
                (src, float(imp))
                for src, imp in (d.split(":") for d in drivers_str.split(";"))
            ]
            if drivers_str
