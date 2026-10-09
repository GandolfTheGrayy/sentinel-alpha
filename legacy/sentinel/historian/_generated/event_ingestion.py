"""
Historical market event ingestion pipeline for Sentinel Sentiment Engine.

This module reads a CSV of past market events (with date, ticker, event type, description)
and embeds them into ChromaDB using a default embedding function. The embedded events
serve as the knowledge base for the Historian RAG pipeline, enabling similarity searches
when analyzing new sentiment signals against historical patterns.

Integration:
  - Called by historian/spine.py during initialization and periodic refreshes.
  - ChromaDB collection is queried by historian/rag_interface.py for event lookup.
"""

import csv
import os
import sqlite3
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings


def initialize_chromadb(persist_dir: str = "./data/chromadb") -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage."""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir,
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def get_or_create_collection(client: chromadb.Client, collection_name: str = "market_events") -> chromadb.Collection:
    """Get or create a ChromaDB collection for historical market events."""
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Historical market events with embeddings"}
    )
    return collection


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load historical market events from CSV file."""
    events = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Event CSV not found: {csv_path}")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "date": row.get("date", ""),
                "ticker": row.get("ticker", "").upper(),
                "event_type": row.get("event_type", ""),
                "description": row.get("description", ""),
            })
    
    return events


def embed_and_ingest_events(
    collection: chromadb.Collection,
    events: list[dict],
    batch_size: int = 100
) -> None:
    """Embed events using ChromaDB's default embedding and ingest into collection."""
    if not events:
        return
    
    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        
        ids = [f"{e['ticker']}_{e['date']}_{j}" for j, e in enumerate(batch)]
        documents = [f"{e['event_type']}: {e['description']}" for e in batch]
        metadatas = [
            {
                "date": e["date"],
                "ticker": e["ticker"],
                "event_type": e["event_type"],
            }
            for e in batch
        ]
        
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


def ingest_historical_events(
    csv_path: str,
    chromadb_persist_dir: str = "./data/chromadb",
    collection_name: str = "market_events",
) -> chromadb.Collection:
    """
    Main ingestion pipeline: load CSV events, initialize ChromaDB, and embed.
    
    Args:
        csv_path: Path to CSV file with columns: date, ticker, event_type, description
        chromadb_persist_dir: Directory for ChromaDB persistence
        collection_name: Name of the ChromaDB collection
    
    Returns:
        ChromaDB collection ready for RAG queries.
    """
    events = load_events_from_csv(csv_path)
    client = initialize_chromadb(chromadb_persist_dir)
    collection = get_or_create_collection(client, collection_name)
    embed_and_ingest_events(collection, events)
    
    return collection


if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/historical_events.csv"
    collection = ingest_historical_events(csv_file)
    print(f"✓ Ingested {collection.count()} events into ChromaDB collection.")
