"""
Historical market event ingestion pipeline for Sentinel.

Reads CSV files containing past market events (e.g., earnings surprises, regulatory
announcements, macroeconomic shifts) and embeds them into ChromaDB using Gemini's
embedding model. This enriches the RAG vector store with historical context,
allowing the Judge to calibrate predictions against similar past scenarios.

Integrates with sentinel/historian/rag_query.py for unified vector lookups.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Optional

import chromadb
from google.generativeai import embed_content
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))


def load_event_csv(csv_path: str) -> list[dict[str, str]]:
    """Load historical events from CSV file into memory."""
    events = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    events.append(row)
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {csv_path}", file=sys.stderr)
    return events


def embed_text(text: str) -> Optional[list[float]]:
    """Embed text using Gemini embedding model."""
    try:
        result = embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return result["embedding"]
    except Exception as e:
        print(f"Error embedding text: {e}", file=sys.stderr)
        return None


def ingest_events_to_chromadb(
    csv_path: str,
    db_path: str = "./sentinel_history.db",
    collection_name: str = "market_events",
) -> int:
    """
    Ingest CSV events into ChromaDB with embeddings.
    
    Returns the number of events successfully ingested.
    """
    events = load_event_csv(csv_path)
    if not events:
        print("No events to ingest.", file=sys.stderr)
        return 0

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    ingested_count = 0
    for idx, event in enumerate(events):
        # Construct a descriptive text from event fields
        event_text = " | ".join([f"{k}: {v}" for k, v in event.items()])
        
        # Generate embedding
        embedding = embed_text(event_text)
        if embedding is None:
            print(f"Skipping event {idx} due to embedding failure.", file=sys.stderr)
            continue

        # Create a unique ID for the event
        event_id = f"event_{idx}"
        
        # Upsert into collection
        try:
            collection.upsert(
                ids=[event_id],
                embeddings=[embedding],
                documents=[event_text],
                metadatas=[event],
            )
            ingested_count += 1
        except Exception as e:
            print(f"Error upserting event {idx}: {e}", file=sys.stderr)

    print(f"Ingested {ingested_count}/{len(events)} events into ChromaDB.", file=sys.stderr)
    return ingested_count


def validate_csv_format(csv_path: str) -> bool:
    """Validate that CSV has required columns (at minimum: date, ticker, event_type)."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return False
            required = {"date", "ticker", "event_type"}
            return required.issubset(set(reader.fieldnames))
    except Exception:
        return False


def main() -> None:
    """CLI entry point for ingesting historical events."""
    if len(sys.argv) < 2:
        print(
            "Usage: python event_ingester.py <csv_path> [db_path] [collection_name]",
            file=sys.stderr
        )
        sys.exit(1)

    csv_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "./sentinel_history.db"
    collection_name = sys.argv[3] if len(sys.argv) > 3 else "market_events"

    # Validate CSV
    if not validate_csv_format(csv_path):
        print(
            f"Error: CSV at {csv_path} must contain 'date', 'ticker', and 'event_type' columns.",
            file=sys.stderr
        )
        sys.exit(1)

    # Ingest
    count = ingest_events_to_chromadb(csv_path, db_path, collection_name)
    print(f"Successfully ingested {count} events.", file=sys.stderr)


if __name__ == "__main__":
    main()
