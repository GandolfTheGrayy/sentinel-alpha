"""
sentinel/scout/data_normalizer.py

Data normalization layer for the Sentinel Sentiment Engine's Scout agent.

This module defines the canonical SignalRecord schema and provides normalizer
functions that map heterogeneous scraper outputs (SEC filings, Reddit posts,
Hacker News threads, GitHub metrics, live price ticks) into a single unified
structure persisted in SQLite.

It sits between raw scraper outputs and all downstream consumers (Linguist,
Historian, Judge), guaranteeing that every signal entering the pipeline shares
the same shape, field semantics, and storage contract regardless of origin.

SQLite is used as the initial time-series store; the schema is designed to be
swap-ready for TimescaleDB by keeping all time columns as ISO-8601 UTC strings
and avoiding SQLite-specific extensions in the DDL.

Usage:
    from sentinel.scout.data_normalizer import (
        init_db, normalize_sec, normalize_reddit,
        normalize_hn, normalize_github, normalize_price,
        insert_signal, query_signals,
    )
    conn = init_db("sentinel.db")
    record = normalize_reddit(raw_praw_post)
    insert_signal(conn, record)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema version — bump when DDL changes to trigger migration warnings.
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Allowed source identifiers — keeps the source column an explicit enum.
# ---------------------------------------------------------------------------
SOURCE_SEC = "sec_edgar"
SOURCE_REDDIT = "reddit"
SOURCE_HN = "hacker_news"
SOURCE_GITHUB = "github"
SOURCE_PRICE = "price_tick"

VALID_SOURCES = {SOURCE_SEC, SOURCE_REDDIT, SOURCE_HN, SOURCE_GITHUB, SOURCE_PRICE}

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    """
    Canonical representation of every ingested data point in Sentinel.

    Fields
    ------
    signal_id : str
        UUID4 primary key, auto-generated if not supplied.
    source : str
        One of VALID_SOURCES.
    ticker : str
        Uppercase equity ticker this signal relates to (e.g. "AAPL").
        Use empty string "" when no ticker can be inferred.
    signal_type : str
        Coarse category: "filing" | "sentiment" | "dev_health" | "price".
    subtype : str
        Fine-grained label, e.g. "8-K" | "reddit_post" | "hn_comment" |
        "commit_velocity" | "ohlcv".
    title : str
        Human-readable headline or description (≤ 512 chars, truncated).
    body : str
        Full text payload or JSON-serialised numeric payload.
    url : str
        Canonical link to the original artefact; empty string if N/A.
    author : str
        Username, bot name, or data provider identifier.
    score : Optional[float]
        Raw engagement metric from source (upvotes, stars, volume, etc.).
        None when not applicable.
    sentiment_raw : Optional[float]
        Pre-computed sentiment from the source itself, if available (-1..1).
        None means "not yet scored" — Linguist will fill this later.
    extra : Dict[str, Any]
        Source-specific metadata that does not fit standard fields, stored
        as a JSON blob.
    ingested_at : str
        ISO-8601 UTC timestamp of when this record entered Sentinel.
    source_ts : str
        ISO-8601 UTC timestamp reported by the originating source.
        Falls back to ingested_at when the source provides no timestamp.
    content_hash : str
        SHA-256 of (source + url + body[:256]) for deduplication.
    """

    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    ticker: str = ""
    signal_type: str = ""
    subtype: str = ""
    title: str = ""
    body: str = ""
    url: str = ""
    author: str = ""
    score: Optional[float] = None
    sentiment_raw: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_ts: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Validate, truncate, and hash after construction."""
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source '{self.source}'. Must be one of {VALID_SOURCES}."
            )
        self.title = self.title[:512]
        if not self.source_ts:
            self.source_ts = self.ingested_at
        if not self.content_hash:
            self.content_hash = _compute_hash(self.source, self.url, self.body)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_hash(source: str, url: str, body: str) -> str:
    """Return SHA-256 hex digest used for deduplication."""
    payload = f"{source}||{url}||{body[:256]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _utc_iso(ts: Any) -> str:
    """
    Convert various timestamp formats to an ISO-8601 UTC string.

    Accepts: datetime objects, Unix epoch ints/floats, or existing ISO strings.
    Returns the ingestion time if conversion fails.
    """
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            pass
    if isinstance(ts, str) and ts:
        return ts  # trust caller for pre-formatted strings
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float, returning None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id       TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    ticker          TEXT NOT NULL DEFAULT '',
    signal_type     TEXT NOT NULL,
    subtype         TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    body            TEXT NOT NULL DEFAULT '',
    url             TEXT NOT NULL DEFAULT '',
    author          TEXT NOT NULL DEFAULT '',
    score           REAL,
    sentiment_raw   REAL,
    extra           TEXT NOT NULL DEFAULT '{}',
    ingested_at     TEXT NOT NULL,
    source_ts       TEXT NOT NULL,
    content_hash    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker      ON signals (ticker);
CREATE INDEX IF NOT EXISTS idx_signals_source      ON signals (source);
CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_source_ts   ON signals (source_ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_hash ON signals (content_hash);
"""


def init_db(db_path: str = "sentinel.db") -> sqlite3.Connection:
    """
    Open (or create) the SQLite database, apply DDL, and return the connection.
    """
    conn = sqlite3.connect(db_path, check_same_thread=
