"""
Historical market event ingestion pipeline for Sentinel.

Reads a CSV of past market events (date, ticker, event type, description, outcome)
and embeds them into ChromaDB using Gemini embeddings. Supports incremental ingestion
with deduplication by (date, ticker, event_type) composite key.

Used by historian/rag_query.py to enrich sentiment predictions with historical context.
"""

import csv
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from google.generativeai import Client, embed_content

# ChromaDB persistence directory
CHROMADB_PATH = Path(__file__).parent.parent / "chroma_data"
CHROMADB_PATH.mkdir(parents=True, exist_ok=True)

# SQLite dedup log
DEDUP_DB = Path(__file__).parent.parent / "event_dedup.db"


def _init_dedup_db() -> sqlite3.Connection:
    """Initialize SQLite table for deduplication tracking."""
    conn = sqlite3.connect(DEDUP_DB)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_events (
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (date, ticker, event_type)
        )
        """
    )
    conn.commit()
    return conn


def _is_event_ingested(
    conn: sqlite3.Connection, date: str, ticker: str, event_type: str
) -> bool:
    """Check if event (date, ticker, event_type) already embedded."""
    cursor = conn.execute(
        "SELECT 1 FROM ingested_events WHERE date=? AND ticker=? AND event_type=?",
        (date, ticker, event_type),
    )
    return cursor.fetchone() is not None


def _mark_event_ingested(
    conn: sqlite3.Connection, date: str, ticker: str, event_type: str
) -> None:
    """Record event as embedded."""
    conn.execute(
        "INSERT INTO ingested_events (date, ticker, event_type, embedded_at) VALUES (?, ?, ?, ?)",
        (date, ticker, event_type, datetime.utcnow().isoformat()),
    )
    conn.commit()


def _embed_text(text: str, api_key: str) -> list[float]:
    """Embed text using Gemini embedding API."""
    client = Client(api_key=api_key)
    result = embed_content(
        model="models/embedding-001",
        content=text,
    )
    return result["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    collection_name: str = "market_events",
    dedup: bool = True,
) -> dict:
    """
    Ingest historical market events from CSV into ChromaDB.

    CSV must have columns: date, ticker, event_type, description, outcome

    Args:
        csv_path: Path to CSV file.
        collection_name: ChromaDB collection name.
        dedup: If True, skip events already ingested.

    Returns:
        Dictionary with keys: total_rows, ingested, skipped, errors, collection_id.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not set")

    # Initialize ChromaDB client and collection
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # Initialize dedup tracking
    dedup_conn = _init_dedup_db() if dedup else None

    stats = {"total_rows": 0, "ingested": 0, "skipped": 0, "errors": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows"] += 1
            try:
                date = row.get("date", "").strip()
                ticker = row.get("ticker", "").strip().upper()
                event_type = row.get("event_type", "").strip()
                description = row.get("description", "").strip()
                outcome = row.get("outcome", "").strip()

                if not all([date, ticker, event_type, description]):
                    stats["errors"] += 1
                    continue

                # Check dedup
                if dedup_conn and _is_event_ingested(dedup_conn, date, ticker, event_type):
                    stats["skipped"] += 1
                    continue

                # Embed description + outcome for context
                full_text = f"{event_type} on {date} for {ticker}: {description}. Outcome: {outcome}"
                embedding = _embed_text(full_text, gemini_key)

                # Add to ChromaDB
                collection.add(
                    ids=[f"{date}_{ticker}_{event_type}".replace(" ", "_")],
                    embeddings=[embedding],
                    documents=[full_text],
                    metadatas=[
                        {
                            "date": date,
                            "ticker": ticker,
                            "event_type": event_type,
                            "outcome": outcome,
                        }
                    ],
                )

                # Mark as ingested
                if dedup_conn:
                    _mark_event_ingested(dedup_conn, date, ticker, event_type)

                stats["ingested"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"Error ingesting row {stats['total_rows']}: {e}")

    if dedup_conn:
        dedup_conn.close()

    stats["collection_id"] = collection.id
    return stats


def query_historical_events(
    query_text: str,
    collection_name: str = "market_events",
    n_results: int = 5,
) -> list[dict]:
    """
    Retrieve historical events similar to query text.

    Args:
        query_text: Text to search for (e.g., ticker + event context).
        collection_name: ChromaDB collection to query.
        n_results: Number of top results to return.

    Returns:
        List of matching events with metadata and similarity scores.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY not set")

    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        return []

    # Embed query
    query_embedding = _embed_text(query_text, gemini_key)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results or not results.get("ids"):
        return []

    # Format results
    formatted = []
    for i, doc_id in enumerate(results["ids"][0]):
        formatted.append(
            {
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],  # Lower = more similar
            }
        )
    return formatted


def clear_collection(collection_name: str = "market_events") -> None:
    """Delete all events from a collection (useful for testing/reset)."""
    client = chromadb.PersistentClient(path=str(CHROMADB_PATH))
    try:
        client.delete_collection(name=collection_name)
