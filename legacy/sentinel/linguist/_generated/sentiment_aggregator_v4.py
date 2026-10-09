"""
Sentiment Aggregator — Linguist Pillar

Combines Scout signals (news sentiment, Reddit/HN sentiment, SEC filing tone)
and Linguist scores (certainty, hesitation, linguistic drift) into a composite
SentimentResidual score via weighted formula. This residual quantifies the
confidence-adjusted net sentiment signal for each ticker, feeding into Judge
predictions and post-mortem calibration.

The aggregator weights each signal by:
  - Temporal recency (more recent = higher weight)
  - Source credibility (SEC filings > news headlines > social media)
  - Linguist certainty score (high certainty amplifies signal magnitude)

Output: SentimentResidual(ticker, composite_score, component_scores, weights, timestamp)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics


@dataclass
class SignalComponent:
    """Single input signal to aggregator (e.g., news headline sentiment)."""
    name: str
    value: float  # Typically in [-1.0, 1.0] range
    certainty: float  # [0.0, 1.0] — confidence in this signal
    source_weight: float  # [0.0, 1.0] — inherent credibility of source
    age_hours: float  # How old is this signal (recency penalty)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SentimentResidual:
    """Composite sentiment output for a single ticker."""
    ticker: str
    composite_score: float  # Weighted average sentiment, [-1.0, 1.0]
    component_scores: Dict[str, float]  # Per-signal final weighted score
    raw_weights: Dict[str, float]  # Normalized weights applied
    signal_count: int
    dominant_signal: Optional[str]  # Highest-impact component
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        dom = self.dominant_signal or "none"
        return (
            f"SentimentResidual(ticker={self.ticker}, "
            f"composite={self.composite_score:.3f}, "
            f"signals={self.signal_count}, dominant={dom})"
        )


class SentimentAggregator:
    """
    Aggregates multi-source sentiment signals into a single composite score
    per ticker, weighted by recency, source credibility, and certainty.
    """

    # Tuning parameters (refined via Judge post-mortem calibration)
    RECENCY_HALF_LIFE_HOURS = 24.0  # Signals decay by 50% every 24h
    SOURCE_WEIGHTS = {
        "sec_filing": 0.40,  # SEC filings most credible
        "news": 0.25,
        "reddit": 0.15,
        "hackernews": 0.15,
        "financial_analyst": 0.30,
    }
    CERTAINTY_AMPLIFICATION = 1.5  # Boost magnitude if Linguist is confident

    def __init__(self, half_life_hours: float = 24.0):
        """Initialize aggregator with optional custom recency decay rate."""
        self.RECENCY_HALF_LIFE_HOURS = half_life_hours
        self._signal_history: Dict[str, List[SignalComponent]] = {}

    def add_signal(
        self,
        ticker: str,
        name: str,
        value: float,
        certainty: float,
        source: str,
        age_hours: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Register a single sentiment signal for a ticker.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL")
            name: Human-readable signal label (e.g., "headline_tesla_delays")
            value: Sentiment value in [-1.0, 1.0] (negative=bearish, positive=bullish)
            certainty: Linguist confidence [0.0, 1.0]
            source: Signal source category (must be in SOURCE_WEIGHTS keys)
            age_hours: How old this signal is (for recency decay)
            timestamp: When signal was generated (defaults to now)
        """
        if ticker not in self._signal_history:
            self._signal_history[ticker] = []

        source_weight = self.SOURCE_WEIGHTS.get(source, 0.10)
        sig = SignalComponent(
            name=name,
            value=value,
            certainty=certainty,
            source_weight=source_weight,
            age_hours=age_hours,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._signal_history[ticker].append(sig)

    def _recency_decay(self, age_hours: float) -> float:
        """
        Exponential decay: signals older than half_life lose 50% weight.
        Returns multiplier in (0.0, 1.0].
        """
        decay_factor = 2.0 ** (-age_hours / self.RECENCY_HALF_LIFE_HOURS)
        return max(decay_factor, 0.01)  # Floor at 1% to avoid complete erasure

    def _compute_signal_weight(self, signal: SignalComponent) -> float:
        """
        Compute final weight for a single signal: recency × source × certainty.
        """
        recency = self._recency_decay(signal.age_hours)
        certainty_boost = 1.0 + (signal.certainty - 0.5) * (
            self.CERTAINTY_AMPLIFICATION - 1.0
        )
        weight = recency * signal.source_weight * certainty_boost
        return weight

    def aggregate(self, ticker: str) -> Optional[SentimentResidual]:
        """
        Compute composite SentimentResidual for a ticker from all registered signals.
        Returns None if no signals exist for ticker.
        """
        if ticker not in self._signal_history or not self._signal_history[ticker]:
            return None

        signals = self._signal_history[ticker]
        weights: Dict[str, float] = {}
        weighted_scores: Dict[str, float] = {}
        total_weight = 0.0

        # Compute per-signal weights and weighted values
        for sig in signals:
            sig_weight = self._compute_signal_weight(sig)
            total_weight += sig_weight
            weights[sig.name] = sig_weight
            weighted_scores[sig.name] = sig.value * sig_weight

        # Normalize weights to [0, 1] sum
        if total_weight > 0.0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
            composite = sum(weighted_scores.values()) / total_weight
        else:
            normalized_weights = {k: 1.0 / len(weights) for k in weights}
            composite = 0.0

        # Identify dominant signal (highest normalized weight)
        dominant = max(
            normalized_weights.items(), key=lambda x: x[1]
        )[0] if normalized_weights else None

        return SentimentResidual(
            ticker=ticker,
            composite_score=max(-1.0, min(1.0, composite)),  # Clamp to [-1, 1]
            component_scores=weighted_scores,
            raw_weights=normalized_weights,
            signal_count=len(signals),
            dominant_signal=dominant,
            timestamp=datetime.utcnow(),
        )

    def aggregate_batch(self, tickers: List[str]) -> Dict[str, SentimentResidual]:
        """
        Compute SentimentResidual for multiple tickers in one call.
        Returns dict mapping ticker -> SentimentResidual (omits tickers with no signals).
        """
        results = {}
        for ticker in tickers:
            residual = self.aggregate(ticker)
            if residual is not None:
                results[ticker] = residual
        return results

    def clear_history(self, ticker: Optional
