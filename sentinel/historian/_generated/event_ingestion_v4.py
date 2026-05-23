"""
Historical market event ingestion pipeline for Sentinel.

Reads a CSV of past market events (e.g., earnings misses, FDA rejections, 
regulatory announcements) and embeds them into ChromaDB using a simple 
embedding function. Serves as the corpus for RAG lookups in historian/rag_query.py.

The pipeline:
  1. Loads CSV with columns: date, ticker, event_type, description, outcome (up/down/neutral)
  2. Generates embeddings via Gemini's embedding API
  3. Stores embeddings + metadata in ChromaDB for similarity search
  4. Provides query interface to find analogous historical events by semantic similarity
"""

import csv
import os
import sqlite3
from datetime import datetime
from typing import Optional

import chromadb
import google.generativeai as genai
import pandas as pd


SENTINEL_DB_DIR = os.environ.get("SENTINEL_DB_DIR", "./sentinel_data")
EVENTS_CSV_PATH = os.environ.get("EVENTS_CSV_PATH", "./sentinel_data/historical_events.csv")
CHROMA_PATH = os.path.join(SENTINEL_DB_DIR, "chroma_events")


def init_chroma_client() -> chromadb.Client:
    """Initialize and return a ChromaDB client configured for event storage."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client


def get_or_create_collection(
    client: chromadb.Client, collection_name: str = "market_events"
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for historical market events."""
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Historical market events with semantic embeddings"},
        )
    return collection


def generate_embedding(text: str) -> list[float]:
    """Generate a vector embedding for text using Gemini's embedding model."""
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return response["embedding"]


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load and parse CSV file of historical market events."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Events CSV not found at {csv_path}")

    events = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def ingest_events(
    csv_path: str = EVENTS_CSV_PATH,
    chroma_collection: Optional[chromadb.Collection] = None,
    skip_embeddings: bool = False,
) -> dict:
    """
    Ingest historical events from CSV into ChromaDB with embeddings.

    Args:
        csv_path: Path to CSV file with columns: date, ticker, event_type, description, outcome
        chroma_collection: ChromaDB collection to store events (creates if None)
        skip_embeddings: If True, use placeholder embeddings (for testing)

    Returns:
        Dictionary with ingestion stats: {'total': int, 'embedded': int, 'errors': list}
    """
    if chroma_collection is None:
        client = init_chroma_client()
        chroma_collection = get_or_create_collection(client)

    events = load_events_from_csv(csv_path)
    stats = {"total": len(events), "embedded": 0, "errors": []}

    for idx, event in enumerate(events):
        try:
            # Validate required fields
            required = ["date", "ticker", "event_type", "description"]
            for field in required:
                if field not in event or not event[field]:
                    raise ValueError(f"Missing or empty field: {field}")

            # Build document text for embedding
            doc_text = (
                f"{event['ticker']} {event['event_type']}: {event['description']} "
                f"({event.get('outcome', 'unknown')})"
            )

            # Generate embedding
            if skip_embeddings:
                embedding = [0.0] * 768  # Placeholder for testing
            else:
                embedding = generate_embedding(doc_text)

            # Store in ChromaDB with metadata
            event_id = f"{event['ticker']}_{event['date']}_{idx}"
            chroma_collection.add(
                ids=[event_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[
                    {
                        "date": event["date"],
                        "ticker": event["ticker"],
                        "event_type": event["event_type"],
                        "outcome": event.get("outcome", "unknown"),
                    }
                ],
            )
            stats["embedded"] += 1

        except Exception as e:
            stats["errors"].append({"event_idx": idx, "error": str(e)})

    return stats


def query_similar_events(
    query_text: str,
    chroma_collection: Optional[chromadb.Collection] = None,
    n_results: int = 5,
    skip_embeddings: bool = False,
) -> list[dict]:
    """
    Query ChromaDB for historical events similar to query_text.

    Args:
        query_text: Description of the event to find analogues for
        chroma_collection: ChromaDB collection (creates if None)
        n_results: Number of similar events to return
        skip_embeddings: If True, return empty results (for testing)

    Returns:
        List of similar events with metadata and distances.
    """
    if chroma_collection is None:
        client = init_chroma_client()
        chroma_collection = get_or_create_collection(client)

    if skip_embeddings:
        return []

    query_embedding = generate_embedding(query_text)
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=n_results)

    events = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0] if results["documents"] else [],
            results["metadatas"][0] if results["metadatas"] else [],
            results["distances"][0] if results["distances"] else [],
        ):
            events.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

    return events


def clear_collection(chroma_collection: Optional[chromadb.Collection] = None) -> None:
    """Delete all documents from the events collection (for testing/resets)."""
    if chroma_collection is None:
        client = init_chroma_client()
        chroma_collection = get_or_create_collection(client)

    # Retrieve all IDs and delete
    all_data = chroma_collection.get()
    if all_data and all_data["ids"]:
        chroma_collection.delete(ids=all_data["ids"])


if __name__ == "__main__":
    # Example: ingest events from CSV
    client = init_chroma_client()
    collection = get_or_create_collection(client)

    print("Ingesting historical events from CSV...")
    stats = ingest_events(chroma_collection=collection)
    print(f"Ingestion complete: {stats['embedded']}/{stats['total']} events embedded")
    if stats["errors"]:
        print(f"Errors: {stats['errors']}")

    # Example: query similar events
    sample_query = "FDA approval delay for pharmaceutical candidate"
    print(f"\nQuerying for events similar to: {sample_query}")
    similar
