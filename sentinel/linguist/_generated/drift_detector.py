"""
Linguistic Drift Detector for Sentinel Sentiment Engine.

Compares a company's current 10-Q filing language against a rolling 30-day
baseline of prior filings and news sentiment to detect significant tone shifts.
Flags increases in risk language, decreases in confidence markers, or sudden
shifts in dominant topics. Used by Judge to weight predictions when drift
is detected.

Integration: Called by sentinel/judge/predictor.py before final scoring.
Output: DriftSignal with severity (LOW/MEDIUM/HIGH), affected topics, and
        confidence metric for downstream weighting.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional
import sqlite3
from datetime import datetime, timedelta
import numpy as np


@dataclass
class DriftSignal:
    """Represents a detected linguistic drift anomaly."""
    ticker: str
    severity: str  # LOW, MEDIUM, HIGH
    drift_score: float  # 0.0 to 1.0
    affected_topics: list[str]
    baseline_avg_risk_tone: float
    current_risk_tone: float
    baseline_avg_confidence: float
    current_confidence: float
    explanation: str
    detected_at: str


class LinguisticDriftDetector:
    """
    Detects significant tone shifts in company filings by comparing
    current 10-Q language against a 30-day rolling baseline.
    """

    # Risk language markers (increase = negative drift)
    RISK_MARKERS = {
        "uncertainty": [
            "may", "might", "could", "potentially", "possibly", "uncertain",
            "risk", "risks", "risky", "threatened", "threat", "challenging",
            "decline", "declined", "declines", "downturn"
        ],
        "legal": [
            "litigation", "lawsuit", "legal", "regulatory", "compliance",
            "violation", "violation", "breach", "penalty", "investigation"
        ],
        "operational": [
            "supply chain", "disruption", "shortage", "delay", "delays",
            "inefficiency", "downtime", "outage", "failure", "failed"
        ],
        "financial": [
            "loss", "losses", "deficit", "liquidity", "debt", "bankruptcy",
            "insolvency", "impairment", "writedown", "margin pressure"
        ]
    }

    # Confidence markers (decrease = negative drift)
    CONFIDENCE_MARKERS = [
        "strong", "robust", "leading", "dominant", "competitive advantage",
        "strategic", "growth", "expanding", "momentum", "opportunity",
        "opportunity", "improved", "improvement", "improvement", "successful"
    ]

    # Anti-markers (increase = negative drift)
    ANTI_CONFIDENCE_MARKERS = [
        "weak", "challenged", "struggling", "declining", "decreased",
        "limited", "constrained", "barriers", "headwinds", "headwind"
    ]

    def __init__(self, db_path: str = "sentinel.db"):
        """Initialize drift detector with baseline storage."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite table for baseline storage."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS linguistic_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                baseline_date TEXT NOT NULL,
                risk_tone REAL,
                confidence_tone REAL,
                topic_distribution TEXT,
                raw_text_hash TEXT UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON linguistic_baselines(ticker, baseline_date)
        """)
        conn.commit()
        conn.close()

    def store_baseline(
        self,
        ticker: str,
        text: str,
        risk_tone: float,
        confidence_tone: float,
        topics: Optional[dict] = None
    ) -> None:
        """Store a filing snapshot for rolling baseline."""
        text_hash = str(hash(text) % (10 ** 8))
        topic_json = json.dumps(topics or {})
        
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO linguistic_baselines
                (ticker, baseline_date, risk_tone, confidence_tone, topic_distribution, raw_text_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                datetime.utcnow().isoformat(),
                risk_tone,
                confidence_tone,
                topic_json,
                text_hash
            ))
            conn.commit()
        finally:
            conn.close()

    def _score_risk_tone(self, text: str) -> float:
        """Calculate risk language density (0.0 to 1.0)."""
        text_lower = text.lower()
        risk_count = 0
        total_markers = sum(len(v) for v in self.RISK_MARKERS.values())
        
        for category_markers in self.RISK_MARKERS.values():
            for marker in category_markers:
                risk_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        
        # Normalize: cap at 1.0
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        risk_density = risk_count / (word_count / 100)  # Per 100 words
        return min(risk_density / 10.0, 1.0)  # Scale to [0, 1]

    def _score_confidence_tone(self, text: str) -> float:
        """Calculate confidence language density (0.0 to 1.0)."""
        text_lower = text.lower()
        confidence_count = 0
        anti_count = 0
        
        for marker in self.CONFIDENCE_MARKERS:
            confidence_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        
        for marker in self.ANTI_CONFIDENCE_MARKERS:
            anti_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.5  # Neutral default
        
        # Net confidence: positive markers - negative markers per 100 words
        net_markers = (confidence_count - anti_count) / (word_count / 100)
        confidence = (net_markers + 5.0) / 10.0  # Center around 0.5, scale to [0, 1]
        return max(0.0, min(confidence, 1.0))

    def _extract_topics(self, text: str) -> dict:
        """Extract dominant topics from text (stub: returns marker counts)."""
        text_lower = text.lower()
        topics = {}
        
        for category, markers in self.RISK_MARKERS.items():
            count = sum(
                len(re.findall(r'\b' + re.escape(m) + r'\b', text_lower))
                for m in markers
            )
            topics[category] = count
        
        return topics

    def detect_drift(
        self,
        ticker: str,
        current_text: str,
        days_lookback: int = 30
    ) -> Optional[DriftSignal]:
        """
        Compare current 10-Q text against 30-day baseline and flag drift.
        
        Returns DriftSignal if drift detected, None otherwise.
        """
        # Score current filing
        current_risk = self._score_risk_tone(current_text)
        current_conf = self._score_confidence_tone(current_text)
        current_topics = self._extract_topics(current_text)
        
        # Fetch baseline from last N days
        conn = sqlite3.connect(self.db_path)
        cutoff_date = (datetime.utc
