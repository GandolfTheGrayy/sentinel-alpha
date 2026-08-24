"""
Historical market event ingestion pipeline for Sentinel.

Reads a CSV of past market events (earnings announcements, regulatory filings,
macroeconomic releases, etc.) and embeds them into ChromaDB for RAG-based
historical context lookup. Events are stored with metadata (date, ticker, event_type,
source) to enable filtered retrieval during prediction synthesis.

Pairs with sentinel/historian/rag_query.py to power the Historian pillar.
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load market events from CSV; expects columns: date, ticker, event_type, headline, source."""
    events = []
    if not os.path.exists(csv_path):
        print(f"[event_ingestion] CSV not found: {csv_path}")
        return events
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get("date") and row.get("ticker"):
                events.append({
                    "date": row["date"],
                    "ticker": row["ticker"].upper(),
                    "event_type": row.get("event_type", "unknown"),
                    "headline": row.get("headline", ""),
                    "source": row.get("source", ""),
                })
    
    print(f"[event_ingestion] Loaded {len(events)} events from {csv_path}")
    return events


def init_chroma_client(db_dir: Optional[str] = None) -> chromadb.Client:
    """Initialize ChromaDB client; defaults to ./chroma_data if db_dir not specified."""
    if db_dir is None:
        db_dir = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_data")
    
    Path(db_dir).mkdir(parents=True, exist_ok=True)
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_dir,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def embed_and_store_events(
    client: chromadb.Client,
    events: list[dict],
    collection_name: str = "market_events",
) -> None:
    """Embed events using Gemini and store in ChromaDB collection."""
    if not events:
        print("[event_ingestion] No events to ingest.")
        return
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    doc_ids = []
    documents = []
    metadatas = []
    
    for idx, event in enumerate(events):
        doc_id = f"{event['ticker']}_{event['date']}_{idx}"
        doc_text = f"{event['event_type']}: {event['headline']} (Source: {event['source']})"
        
        doc_ids.append(doc_id)
        documents.append(doc_text)
        metadatas.append({
            "date": event["date"],
            "ticker": event["ticker"],
            "event_type": event["event_type"],
            "source": event["source"],
        })
    
    # ChromaDB auto-embeds with default Sentence Transformers; no external API call needed here.
    collection.add(
        ids=doc_ids,
        documents=documents,
        metadatas=metadatas,
    )
    
    print(f"[event_ingestion] Stored {len(events)} events in '{collection_name}' collection.")


def ingest_events_workflow(
    csv_path: str,
    db_dir: Optional[str] = None,
    collection_name: str = "market_events",
) -> chromadb.Client:
    """
    End-to-end workflow: load CSV, initialize ChromaDB, embed and store events.
    Returns the ChromaDB client for downstream RAG queries.
    """
    print(f"[event_ingestion] Starting workflow for {csv_path}")
    
    events = load_events_from_csv(csv_path)
    if not events:
        print("[event_ingestion] No events loaded; skipping ChromaDB ingest.")
        return init_chroma_client(db_dir)
    
    client = init_chroma_client(db_dir)
    embed_and_store_events(client, events, collection_name)
    
    print("[event_ingestion] Workflow complete.")
    return client


if __name__ == "__main__":
    # Example usage: point to a sample CSV and ingest.
    sample_csv = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "sample_events.csv"
    )
    ingest_events_workflow(sample_csv)
