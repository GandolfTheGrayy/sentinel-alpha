"""
Unit tests for the Linguistic Drift detector.

This module validates the drift-scoring logic that detects tone and language
shifts in company-specific sentiment signals (SEC filings, news, social media)
over time. Drift detection feeds into the Judge's confidence calibration,
flagging anomalous linguistic patterns that may precede market moves.

Tests use fixture text corpora to assert correct drift scoring, drift direction
classification, and edge cases (empty history, single snapshot, identical text).
"""

import pytest
from typing import List, Dict
import sqlite3
import tempfile
from pathlib import Path


class LinguisticDriftDetector:
    """Mock implementation of drift detector for testing."""

    def __init__(self, db_path: str):
        """Initialize detector with SQLite backing store."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create schema for storing text snapshots and drift scores."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS text_snapshots (
                id INTEGER PRIMARY KEY,
                company TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text BLOB NOT NULL,
                UNIQUE(company, timestamp, source)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_scores (
                id INTEGER PRIMARY KEY,
                company TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                score REAL NOT NULL,
                direction TEXT NOT NULL,
                magnitude REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def record_snapshot(
        self, company: str, timestamp: str, source: str, text: str
    ) -> None:
        """Record a text snapshot (filing, news, social) for drift tracking."""
        import hashlib

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO text_snapshots
            (company, timestamp, source, text_hash, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (company, timestamp, source, text_hash, text),
        )
        conn.commit()
        conn.close()

    def compute_drift(self, company: str, lookback_days: int = 30) -> Dict:
        """
        Compute linguistic drift for a company over lookback window.
        
        Returns dict with keys: score (0-1), direction (pos/neg/neutral), magnitude.
        """
        conn = sqlite3.connect(self.db_path)
        snapshots = conn.execute(
            """
            SELECT timestamp, text FROM text_snapshots
            WHERE company = ?
            ORDER BY timestamp DESC
            LIMIT 2
            """,
            (company,),
        ).fetchall()
        conn.close()

        if len(snapshots) < 2:
            return {"score": 0.0, "direction": "neutral", "magnitude": 0.0}

        recent_text = snapshots[0][1]
        older_text = snapshots[1][1]

        # Simple heuristic: measure keyword shifts
        recent_words = set(recent_text.lower().split())
        older_words = set(older_text.lower().split())

        if len(older_words) == 0:
            return {"score": 0.0, "direction": "neutral", "magnitude": 0.0}

        common = len(recent_words & older_words)
        jaccard = common / len(recent_words | older_words) if recent_words | older_words else 0.0
        drift_score = 1.0 - jaccard

        # Detect sentiment direction via keyword presence
        bullish_terms = {"growth", "strong", "exceed", "opportunity", "robust", "gain"}
        bearish_terms = {"decline", "challenge", "risk", "pressure", "weakness", "loss"}

        recent_bullish = len([w for w in recent_words if w in bullish_terms])
        recent_bearish = len([w for w in recent_words if w in bearish_terms])
        older_bullish = len([w for w in older_words if w in bullish_terms])
        older_bearish = len([w for w in older_words if w in bearish_terms])

        bullish_shift = recent_bullish - older_bullish
        bearish_shift = recent_bearish - older_bearish

        if bullish_shift > bearish_shift:
            direction = "positive"
        elif bearish_shift > bullish_shift:
            direction = "negative"
        else:
            direction = "neutral"

        magnitude = abs(bullish_shift - bearish_shift) / max(
            older_bullish + older_bearish, 1
        )

        return {
            "score": drift_score,
            "direction": direction,
            "magnitude": magnitude,
        }


@pytest.fixture
def detector():
    """Provide a temporary detector instance for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "drift.db")
        yield LinguisticDriftDetector(db_path)


@pytest.fixture
def fixture_texts() -> Dict[str, List[str]]:
    """Provide fixture text corpora for test assertions."""
    return {
        "stable": [
            "We maintain strong operational performance with steady revenue growth.",
            "We maintain steady operational performance with strong revenue growth.",
        ],
        "bullish_shift": [
            "Revenue declined and margins contracted amid market pressure.",
            "Revenue exceeded expectations with strong growth and margin expansion.",
        ],
        "bearish_shift": [
            "We achieved record profits with exceptional operational efficiency.",
            "We face significant headwinds, declining profitability and market weakness.",
        ],
        "high_drift": [
            "The company operates in the technology sector with digital infrastructure.",
            "Blockchain neural networks quantum cryptography decentralized metaverse protocols.",
        ],
    }


def test_drift_with_single_snapshot(detector: LinguisticDriftDetector) -> None:
    """Assert drift is neutral when only one snapshot exists."""
    detector.record_snapshot("ACME", "2025-01-01", "10-Q", "Revenue growth continues.")
    result = detector.compute_drift("ACME")
    assert result["score"] == 0.0
    assert result["direction"] == "neutral"
    assert result["magnitude"] == 0.0


def test_drift_with_stable_text(
    detector: LinguisticDriftDetector, fixture_texts: Dict
) -> None:
    """Assert low drift when text changes minimally."""
    texts = fixture_texts["stable"]
    detector.record_snapshot("STABLE", "2025-01-01", "news", texts[0])
    detector.record_snapshot("STABLE", "2025-01-08", "news", texts[1])
    result = detector.compute_drift("STABLE")
    assert result["score"] < 0.3, "Stable text should have low drift"
    assert result["direction"] == "neutral"


def test_drift_detects_bullish_shift(
    detector: LinguisticDriftDetector, fixture_texts: Dict
) -> None:
    """Assert positive drift direction when tone shifts bullish."""
    texts = fixture_texts["bullish_shift"]
    detector.record_snapshot("BULL", "2025-01-01", "8-K", texts[0])
    detector.record_snapshot("BULL", "2025-01-08", "8-K", texts[1])
    result = detector.compute_drift("BULL")
    assert result["direction"] == "positive"
    assert result["magnitude"] > 0.2


def test_drift_detects_bearish_shift(
    detector: LinguisticDriftDetector, fixture_texts: Dict
) -> None:
    """Assert negative drift direction when tone shifts bearish."""
    texts = fixture_texts["bearish_shift"]
    detector.record_snapshot("BEAR", "2025-01-01", "10-Q", texts[0])
    detector.record_snapshot("BEAR",
