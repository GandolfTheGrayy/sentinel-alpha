"""
Historical Market Event Ingestion Pipeline for Sentinel.

This module reads past market events from a CSV file and embeds them into
ChromaDB using a simple embedding function. Events are indexed by ticker,
date, and event type for efficient RAG lookups during prediction scoring.

The ingestion pipeline:
  1. Loads events from CSV (expected columns: date, ticker, event_type, description)
  2. Generates embeddings via Gemini's embedding API
  3. Stores vectors + metadata in ChromaDB collection
  4. Provides query interface for Historian RAG synthesis

Used by: sentinel/historian/rag_query.py (historical context lookup)
"""

import csv
import os
from pathlib import Path
from typing import Optional
import sqlite3

import chromadb
from google.generativeai import embed_content
import google.generativeai as genai


def initialize_event_db(db_path: str = "sentinel_events.db") -> sqlite3.Connection:
    """Initialize SQLite database for event metadata."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_events (
            id INTEGER PRIMARY KEY,
            date TEXT NOT NULL,
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            source TEXT,
            impact_direction TEXT,
            UNIQUE(date, ticker, event_type)
        )
    """)
    conn.commit()
    return conn


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load market events from CSV file."""
    events = []
    if not Path(csv_path).exists():
        return events
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date") and row.get("ticker") and row.get("event_type"):
                events.append({
                    "date": row["date"],
                    "ticker": row["ticker"].upper(),
                    "event_type": row["event_type"],
                    "description": row.get("description", ""),
                    "source": row.get("source", "historical"),
                    "impact_direction": row.get("impact_direction", "neutral"),
                })
    return events


def embed_event_description(description: str) -> list[float]:
    """Generate embedding for event description using Gemini."""
    response = embed_content(
        model="models/embedding-001",
        content=description,
    )
    return response["embedding"]


def ingest_events_to_chromadb(
    events: list[dict],
    chroma_path: str = ".chroma",
    collection_name: str = "market_events",
) -> chromadb.Collection:
    """Ingest events into ChromaDB with embeddings and metadata."""
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    for i, event in enumerate(events):
        event_id = f"{event['date']}_{event['ticker']}_{event['event_type']}"
        description = event.get("description", "")
        
        if not description:
            description = f"{event['event_type']} for {event['ticker']} on {event['date']}"
        
        try:
            embedding = embed_event_description(description)
            collection.add(
                ids=[event_id],
                embeddings=[embedding],
                metadatas=[{
                    "date": event["date"],
                    "ticker": event["ticker"],
                    "event_type": event["event_type"],
                    "source": event.get("source", "historical"),
                    "impact_direction": event.get("impact_direction", "neutral"),
                }],
                documents=[description],
            )
        except Exception as e:
            print(f"Error embedding event {event_id}: {e}")
            continue
    
    return collection


def store_events_in_sqlite(
    events: list[dict],
    db_path: str = "sentinel_events.db",
) -> None:
    """Store event metadata in SQLite for fast filtering."""
    conn = initialize_event_db(db_path)
    cursor = conn.cursor()
    
    for event in events:
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO market_events
                (date, ticker, event_type, description, source, impact_direction)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event["date"],
                event["ticker"],
                event["event_type"],
                event.get("description", ""),
                event.get("source", "historical"),
                event.get("impact_direction", "neutral"),
            ))
        except sqlite3.IntegrityError:
            pass
    
    conn.commit()
    conn.close()


def query_events_by_ticker(
    ticker: str,
    collection: chromadb.Collection,
    db_path: str = "sentinel_events.db",
    limit: int = 10,
) -> list[dict]:
    """Retrieve historical events for a ticker from SQLite."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT date, ticker, event_type, description, source, impact_direction
        FROM market_events
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """, (ticker.upper(), limit))
    
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def ingest_pipeline(
    csv_path: str,
    chroma_path: str = ".chroma",
    db_path: str = "sentinel_events.db",
    collection_name: str = "market_events",
) -> tuple[chromadb.Collection, sqlite3.Connection]:
    """
    Full ingestion pipeline: load CSV → embed → store in ChromaDB + SQLite.
    
    Returns: (ChromaDB collection, SQLite connection)
    """
    print(f"Loading events from {csv_path}...")
    events = load_events_from_csv(csv_path)
    print(f"Loaded {len(events)} events.")
    
    if not events:
        print("No events to ingest.")
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(name=collection_name)
        conn = initialize_event_db(db_path)
        return collection, conn
    
    print(f"Storing {len(events)} events in SQLite...")
    store_events_in_sqlite(events, db_path)
    
    print(f"Embedding and ingesting into ChromaDB...")
    collection = ingest_events_to_chromadb(events, chroma_path, collection_name)
    
    print(f"Ingestion complete. {len(events)} events indexed.")
    conn = sqlite3.connect(db_path)
    return collection, conn


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "historical_events.csv"
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    collection, conn = ingest_pipeline(csv_file)
    print(f"ChromaDB collection: {collection.name}")
    print(f"Total docs in collection: {collection.count()}")
    conn.close()
