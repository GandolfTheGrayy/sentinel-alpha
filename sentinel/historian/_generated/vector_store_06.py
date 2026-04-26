"""
sentinel/historian/vector_store.py

ChromaDB vector database setup and client wrapper for the Sentinel Historian Agent.

This module is the persistence backbone of the Historian pipeline. It initialises
a local, on-disk ChromaDB instance and exposes two typed collections:

  - "market_events"  : historical price-move episodes, tagged with ticker,
                       date, magnitude, and the narrative that preceded the move.
  - "sec_filings"    : 8-K / 10-Q filing excerpts, tagged with ticker, form
                       type, CIK, and filing date, used for Regulatory Whispers
                       cross-referencing.

The HistorianStore class wraps both collections behind a clean interface that
the rest of the Historian Agent (RAG query, confidence weighting, ingestion
pipeline) imports directly.  No other module should instantiate ChromaDB
directly — route everything through this wrapper so collection names and
embedding settings stay in one place.

Approved packages used: chromadb, numpy, sqlite3 (stdlib).
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_PERSIST_DIR: str = "data/historian/chromadb"

COLLECTION_MARKET_EVENTS: str = "market_events"
COLLECTION_SEC_FILINGS: str = "sec_filings"

# ChromaDB embedding function — use the built-in sentence-transformers default
# (all-MiniLM-L6-v2) so we stay within the approved package list while still
# getting decent semantic search quality.
_DEFAULT_EF = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

# ---------------------------------------------------------------------------
# Typed data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class MarketEventRecord:
    """A single historical market-move episode to be stored in the vector DB."""

    ticker: str
    event_date: str                    # ISO-8601 date string, e.g. "2024-01-15"
    headline: str                      # Short narrative summary (becomes the embedded text)
    price_change_pct: float            # Actual % price change on / after the event
    volume_spike: float                # Volume vs. 30-day average (1.0 = normal)
    sector: str = ""
    source: str = ""                   # e.g. "reuters", "bloomberg", "reddit"
    tags: List[str] = field(default_factory=list)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class SecFilingRecord:
    """An SEC filing excerpt to be stored in the vector DB."""

    ticker: str
    cik: str
    form_type: str                     # "8-K", "10-Q", "10-K", etc.
    filing_date: str                   # ISO-8601 date string
    accession_number: str
    excerpt: str                       # The text chunk to embed
    section_label: str = ""            # e.g. "Risk Factors", "MD&A"
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class QueryResult:
    """Structured result returned from a similarity search."""

    doc_id: str
    document: str
    distance: float
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# SQLite audit log (lightweight provenance trail)
# ---------------------------------------------------------------------------


def _init_audit_db(db_path: str) -> sqlite3.Connection:
    """Create (or open) the SQLite audit database and ensure schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            collection    TEXT    NOT NULL,
            doc_id        TEXT    NOT NULL,
            ticker        TEXT,
            ingested_at   TEXT    NOT NULL,
            source        TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS query_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            collection    TEXT    NOT NULL,
            query_text    TEXT    NOT NULL,
            n_results     INTEGER NOT NULL,
            queried_at    TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


class HistorianStore:
    """
    Typed wrapper around a local ChromaDB instance for the Sentinel Historian.

    Manages two collections (market_events, sec_filings) and an SQLite audit
    trail.  Designed to be instantiated once per process and shared across all
    Historian sub-modules.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        audit_db_path: Optional[str] = None,
    ) -> None:
        """Initialise ChromaDB client, collections, and the SQLite audit log."""
        self._persist_dir = persist_dir
        self._audit_db_path = audit_db_path or str(
            Path(persist_dir).parent / "audit.sqlite3"
        )

        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._market_events = self._client.get_or_create_collection(
            name=COLLECTION_MARKET_EVENTS,
            embedding_function=_DEFAULT_EF,
            metadata={
                "description": "Historical market-move episodes with narrative context",
                "hnsw:space": "cosine",
            },
        )

        self._sec_filings = self._client.get_or_create_collection(
            name=COLLECTION_SEC_FILINGS,
            embedding_function=_DEFAULT_EF,
            metadata={
                "description": "SEC 8-K/10-Q filing text excerpts for RAG retrieval",
                "hnsw:space": "cosine",
            },
        )

        self._audit = _init_audit_db(self._audit_db_path)
        logger.info(
            "HistorianStore ready — persist_dir=%s  audit_db=%s",
            persist_dir,
            self._audit_db_path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def market_events_count(self) -> int:
        """Return the number of documents in the market_events collection."""
        return self._market_events.count()

    @property
    def sec_filings_count(self) -> int:
        """Return the number of documents in the sec_filings collection."""
        return self._sec_filings.count()

    # ------------------------------------------------------------------
    # Ingestion — market events
    # ------------------------------------------------------------------

    def add_market_event(self, record: MarketEventRecord) -> str:
        """Embed and store a single MarketEventRecord; return its doc_id."""
        metadata: Dict[str, Any] = {
            "ticker": record.ticker,
            "event_date": record.event_date,
            "price_change_pct": record.price_change_pct,
            "volume_spike": record.volume_spike,
            "sector": record.sector,
            "source": record.source,
            "tags": ",".join(record.tags),
        }
        self._market_events.add(
            documents=[record.headline],
            metadatas=[metadata],
            ids=[record.doc_id],
        )
        self._log_ingest(
            collection=COLLECTION_MARKET_EVENTS,
            doc_id=record.doc_id,
            ticker=record.ticker,
            source=record.source,
