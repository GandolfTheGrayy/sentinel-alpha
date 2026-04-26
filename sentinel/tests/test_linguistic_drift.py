"""
Sentinel Test Suite — Linguistic Drift Detector Unit Tests

This module validates the Linguist agent's Linguistic Drift detector, which
tracks tone and certainty shifts in corporate communications over a rolling
30-day window. Drift scoring surfaces subtle changes in executive language
that may precede material market events before they are publicly acknowledged.

Tests use fixture text corpora representing:
  - Stable/confident language baselines (low drift expected)
  - Escalating hedging and uncertainty language (high drift expected)
  - Mixed / neutral transitions (mid-range drift expected)
  - Edge cases: empty inputs, single-document windows, identical documents

The detector under test lives at sentinel/linguist/linguistic_drift.py.
These tests are designed to run with `pytest` and require no external services
or API keys — all LLM calls in the detector must be mockable via monkeypatch.
"""

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — allow imports from project root without install
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixture corpora
# ---------------------------------------------------------------------------

CONFIDENT_TEXTS = [
    (
        "We are extremely pleased to report record-breaking revenue this quarter. "
        "Our pipeline is strong, execution has been flawless, and we are raising "
        "full-year guidance with confidence. Customer retention is at an all-time "
        "high and our market position has never been stronger."
    ),
    (
        "We delivered exceptional results driven by disciplined cost management "
        "and robust demand. Our balance sheet is healthy and we are accelerating "
        "strategic investments. We expect continued momentum into next quarter."
    ),
    (
        "Product adoption is exceeding expectations across all segments. "
        "We are on track to achieve our long-term targets. The fundamentals "
        "of our business remain solid and our competitive advantages are widening."
    ),
]

HEDGING_TEXTS = [
    (
        "We believe, subject to certain assumptions, that results may potentially "
        "approach prior guidance ranges, although we cannot rule out material "
        "headwinds. It is possible that some customers could reduce spend. "
        "We are cautiously monitoring several uncertainties."
    ),
    (
        "While we remain cautiously optimistic, we must acknowledge that "
        "macroeconomic conditions could adversely impact our outlook. "
        "We may need to reassess our targets depending on circumstances. "
        "There are risks we are unable to fully quantify at this time."
    ),
    (
        "We are unable to provide specific forward guidance given the current "
        "environment. Results could vary significantly from expectations. "
        "Management is evaluating strategic alternatives and cannot guarantee "
        "any particular outcome. Investors should consider these risk factors carefully."
    ),
]

MIXED_TEXTS = [
    (
        "Our core business performed well this quarter with strong unit economics. "
        "However, we are monitoring some potential headwinds in certain geographies. "
        "We remain confident in our strategy while acknowledging market uncertainties."
    ),
    (
        "Revenue growth was solid and margins improved year over year. "
        "We are cautiously watching macroeconomic indicators but believe "
        "our diversified model provides resilience. Guidance is maintained."
    ),
    (
        "Customer acquisition costs rose modestly but lifetime value remains high. "
        "We may adjust our marketing spend depending on Q3 signals. "
        "Overall we are pleased with our execution and remain on plan."
    ),
]

IDENTICAL_TEXTS = [
    "Our business is performing in line with expectations and guidance is unchanged."
] * 3

EMPTY_WINDOW: list[str] = []

SINGLE_TEXT = [
    "We are confident in our ability to deliver shareholder value this fiscal year."
]


# ---------------------------------------------------------------------------
# Stub the sentinel.linguist.linguistic_drift module so tests can run
# even before the real implementation exists, while still testing the
# contract the module must satisfy.
# ---------------------------------------------------------------------------

def _build_stub_module() -> types.ModuleType:
    """Build a minimal stub of linguistic_drift satisfying the public contract."""
    mod = types.ModuleType("sentinel.linguist.linguistic_drift")
    mod.__file__ = str(PROJECT_ROOT / "sentinel" / "linguist" / "linguistic_drift.py")

    class DriftResult:
        """Minimal stub result dataclass."""

        def __init__(
            self,
            drift_score: float,
            baseline_tone: str,
            current_tone: str,
            delta_summary: str,
            flagged: bool,
        ) -> None:
            self.drift_score = drift_score
            self.baseline_tone = baseline_tone
            self.current_tone = current_tone
            self.delta_summary = delta_summary
            self.flagged = flagged

        def __repr__(self) -> str:  # pragma: no cover
            return (
                f"DriftResult(score={self.drift_score:.3f}, "
                f"flagged={self.flagged})"
            )

    mod.DriftResult = DriftResult  # type: ignore[attr-defined]

    def compute_lexical_hedging_ratio(text: str) -> float:
        """
        Return ratio of hedging tokens to total tokens in *text*.

        Hedging tokens are words/phrases associated with uncertainty:
        'may', 'might', 'could', 'possibly', 'potentially', 'uncertain',
        'cautious', 'evaluate', 'cannot', 'unable', 'reassess', 'risk',
        'depend', 'subject to', 'approximately', 'believe', 'assume'.
        """
        HEDGE_TOKENS = {
            "may", "might", "could", "possibly", "potentially",
            "uncertain", "cautiously", "evaluate", "cannot", "unable",
            "reassess", "risk", "risks", "depend", "depending",
            "approximately", "believe", "assume", "assumptions",
            "cautious", "potentially", "subject",
        }
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        hedge_count = sum(1 for t in tokens if t.strip(".,;:()") in HEDGE_TOKENS)
        return hedge_count / len(tokens)

    def compute_window_drift(
        window_texts: list[str],
        flag_threshold: float = 0.15,
    ) -> "DriftResult":
        """
        Compute linguistic drift across a rolling window of texts.

        Compares the average hedging ratio of the first half of the window
        (baseline) against the second half (current). Returns a DriftResult
        with a normalised drift_score in [0, 1] and a flagged boolean when
        the absolute delta exceeds *flag_threshold*.
        """
        if not window_texts:
            return DriftResult(
                drift_score=0.0,
                baseline_tone="unknown",
                current_tone="unknown",
                delta_summary="No documents in window.",
                flagged=False,
            )
        if len(window_texts) == 1:
            ratio = compute_lexical_hedging_ratio(window_texts[0])
            tone = "hedging" if ratio > 0.05 else "confident"
            return DriftResult(
                drift_score=0.0,
                baseline_tone=tone,
                current_tone=tone,
                delta_summary="Single document — no drift measurable.",
                flagged=False,
            )

        mid = len(window_texts) // 2
        baseline_docs = window_texts[:mid]
        current_docs = window_texts[mid:]

        baseline_ratio = sum(
            compute_lexical_hedging_ratio(t) for t in baseline_docs
        ) / len(baseline_docs)
        current_ratio = sum(
            compute_lexical_hedging_ratio(t) for t in current_docs
        ) / len(current_docs)

        delta = current_ratio - baseline_ratio
        # Normalise to [0, 1] using a sigmoid-like clamp
        drift_score =
