"""
Historical market event ingestion pipeline for Sentinel Sentiment Engine.

This module reads a CSV of past market events (e.g., earnings announcements,
regulatory filings, analyst downgrades) and embeds them into ChromaDB using
Gemini embeddings. The embedded corpus supports RAG queries in historian/rag_query.py
by providing historical context and similar past events to inform predictions.

The pipeline normalizes event metadata, deduplicates, and stores embeddings
with source attribution for post-mortem traceability.
"""

import os
import csv
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
import pandas as pd

# ============================================================================
# Configuration & Constants
# ============================================================================

EMBEDDING_MODEL = "text-embedding-004"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
CSV_ENCODING = "utf-8"
BATCH_SIZE = 100  # Embed in batches to avoid rate limits


# ============================================================================
# Event Ingestion Functions
# ============================================================================


def load_event_csv(filepath: str) -> list[dict]:
    """Load and validate market events from CSV file."""
    events = []
    try:
        df = pd.read_csv(filepath, encoding=CSV_ENCODING)
        required_cols = {"date", "ticker", "event_type", "description"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV missing required columns: {required_cols}")
        for _, row in df.iterrows():
            events.append(row.to_dict())
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV {filepath}: {e}")
    return events


def normalize_event(event: dict) -> dict:
    """Normalize event metadata and add derived fields."""
    normalized = {
        "date": event.get("date", "").strip(),
        "ticker": event.get("ticker", "").strip().upper(),
        "event_type": event.get("event_type", "").strip().lower(),
        "description": event.get("description", "").strip(),
        "impact": event.get("impact", "neutral").strip().lower(),
        "source": event.get("source", "unknown").strip(),
    }

    # Validate date format
    try:
        datetime.fromisoformat(normalized["date"])
    except ValueError:
        normalized["date"] = ""

    # Ensure ticker is valid (non-empty, alphanumeric + dash/dot)
    if not (normalized["ticker"] and normalized["ticker"].replace("-", "").replace(".", "").isalnum()):
        normalized["ticker"] = ""

    # Ensure description is non-empty
    if not normalized["description"]:
        raise ValueError("Event must have a non-empty description")

    return normalized


def deduplicate_events(events: list[dict]) -> list[dict]:
    """Remove duplicate events by content hash."""
    seen = set()
    unique = []
    for event in events:
        content = f"{event['date']},{event['ticker']},{event['event_type']},{event['description']}"
        hash_val = hashlib.md5(content.encode()).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            unique.append(event)
    return unique


def generate_event_embeddings(
    events: list[dict], api_key: Optional[str] = None
) -> list[tuple[dict, list[float]]]:
    """Generate Gemini embeddings for event descriptions."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)
    embedded_events = []

    for i in range(0, len(events), BATCH_SIZE):
        batch = events[i : i + BATCH_SIZE]
        texts = [
            f"Event: {e['event_type']} | Ticker: {e['ticker']} | Date: {e['date']} | {e['description']}"
            for e in batch
        ]

        try:
            response = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT",
            )
            embeddings = response["embedding"]
            for event, embedding in zip(batch, embeddings):
                embedded_events.append((event, embedding))
        except Exception as e:
            raise RuntimeError(f"Embedding failed for batch {i}: {e}")

    return embedded_events


def ingest_events_to_chromadb(
    embedded_events: list[tuple[dict, list[float]]], collection_name: str = "market_events"
) -> chromadb.Collection:
    """Store embedded events in ChromaDB with metadata."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for event, embedding in embedded_events:
        # Generate unique ID from event content
        content_hash = hashlib.md5(
            f"{event['date']},{event['ticker']},{event['event_type']}".encode()
        ).hexdigest()
        event_id = f"event_{content_hash[:12]}"

        ids.append(event_id)
        embeddings.append(embedding)
        documents.append(event["description"])
        metadatas.append(
            {
                "date": event["date"],
                "ticker": event["ticker"],
                "event_type": event["event_type"],
                "impact": event["impact"],
                "source": event["source"],
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )

    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as e:
        raise RuntimeError(f"ChromaDB upsert failed: {e}")

    return collection


def ingest_csv_pipeline(
    csv_filepath: str,
    collection_name: str = "market_events",
    api_key: Optional[str] = None,
) -> dict:
    """
    End-to-end pipeline: load CSV → normalize → deduplicate → embed → store.
    Returns a summary dict with counts and status.
    """
    summary = {
        "status": "failed",
        "csv_file": csv_filepath,
        "loaded_count": 0,
        "normalized_count": 0,
        "deduplicated_count": 0,
        "embedded_count": 0,
        "stored_count": 0,
        "collection_name": collection_name,
        "error": None,
    }

    try:
        # Load & validate CSV
        raw_events = load_event_csv(csv_filepath)
        summary["loaded_count"] = len(raw_events)

        # Normalize
        normalized_events = []
        for event in raw_events:
            try:
                normalized = normalize_event(event)
                normalized_events.append(normalized)
            except ValueError as e:
                # Skip invalid events with warning
                print(f"Warning: Skipping invalid event: {e}")
                continue
        summary["normalized_count"] = len(normalized_events)

        # Deduplicate
        unique_events = deduplicate_events(normalized_events)
        summary["deduplicated_count"] = len(unique_events)

        if not unique_events:
            summary["error"] = "No valid events after normalization/deduplication"
            return summary

        # Embed
        embedded_events = generate_event_embeddings(unique_events, api_key=api_key)
        summary["embedded_count"] = len(embedded_events)

        # Store in ChromaDB
        collection = ingest_events_to_chromadb(embedded_
