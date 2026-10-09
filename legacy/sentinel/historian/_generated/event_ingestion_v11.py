"""
Historical market event ingestion pipeline for Sentinel.

Reads past market events from CSV, embeds them via Gemini's embedding API,
and stores vectors in ChromaDB for RAG lookup. Supports event deduplication,
metadata tagging (company, event_type, date), and incremental updates.

Integrates with sentinel/historian/rag_query.py for event-aware predictions.
"""

import csv
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
import pandas as pd

# Initialize Gemini for embeddings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def load_events_csv(csv_path: str) -> list[dict]:
    """Load market events from CSV file."""
    events = []
    if not os.path.exists(csv_path):
        return events
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                events.append(row)
    return events


def embed_text(text: str) -> Optional[list[float]]:
    """Embed text using Gemini embedding API."""
    if not GEMINI_API_KEY:
        return None
    
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def deduplicate_events(events: list[dict], db_path: str = ":memory:") -> list[dict]:
    """Remove duplicate events using SQLite dedup table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seen_events (
            event_hash TEXT PRIMARY KEY,
            event_date TEXT,
            company TEXT,
            event_type TEXT
        )
    """)
    conn.commit()
    
    unique_events = []
    for event in events:
        event_hash = hash(
            f"{event.get('date', '')}-{event.get('company', '')}-{event.get('title', '')}"
        )
        event_hash_str = str(abs(event_hash))
        
        cursor.execute("SELECT 1 FROM seen_events WHERE event_hash = ?", (event_hash_str,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO seen_events (event_hash, event_date, company, event_type) VALUES (?, ?, ?, ?)",
                (event_hash_str, event.get("date"), event.get("company"), event.get("event_type"))
            )
            unique_events.append(event)
    
    conn.commit()
    conn.close()
    return unique_events


def prepare_event_documents(events: list[dict]) -> list[tuple[str, str, dict]]:
    """Convert events to (id, text, metadata) tuples for ChromaDB."""
    documents = []
    for i, event in enumerate(events):
        event_id = f"event_{event.get('date', 'unknown')}_{i}"
        
        text = f"{event.get('title', 'Unknown Event')}. {event.get('description', '')}. Company: {event.get('company', 'N/A')}."
        
        metadata = {
            "date": event.get("date", ""),
            "company": event.get("company", ""),
            "event_type": event.get("event_type", "general"),
            "source": event.get("source", "historical"),
        }
        
        documents.append((event_id, text, metadata))
    
    return documents


def ingest_events_to_chromadb(
    csv_path: str,
    chromadb_path: str = "./chroma_events",
    collection_name: str = "market_events"
) -> tuple[int, int]:
    """
    Ingest historical events from CSV into ChromaDB.
    
    Returns (total_ingested, embedded_count).
    """
    # Load and deduplicate
    events = load_events_csv(csv_path)
    print(f"Loaded {len(events)} events from CSV.")
    
    unique_events = deduplicate_events(events)
    print(f"After deduplication: {len(unique_events)} unique events.")
    
    # Prepare documents
    documents = prepare_event_documents(unique_events)
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    embedded_count = 0
    for event_id, text, metadata in documents:
        embedding = embed_text(text)
        
        if embedding:
            collection.upsert(
                ids=[event_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
            embedded_count += 1
        else:
            # Fallback: add without embedding for keyword search
            collection.upsert(
                ids=[event_id],
                metadatas=[metadata],
                documents=[text]
            )
    
    print(f"Ingested {embedded_count}/{len(documents)} events with embeddings.")
    return len(documents), embedded_count


def query_similar_events(
    query_text: str,
    chromadb_path: str = "./chroma_events",
    collection_name: str = "market_events",
    n_results: int = 3
) -> list[dict]:
    """Query ChromaDB for events similar to input text."""
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    query_embedding = embed_text(query_text)
    if query_embedding is None:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    similar = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i, event_id in enumerate(results["ids"][0]):
            similar.append({
                "id": event_id,
                "text": results["documents"][0][i] if results["documents"] else "",
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0
            })
    
    return similar


def update_events_incremental(
    new_events_csv: str,
    chromadb_path: str = "./chroma_events",
    collection_name: str = "market_events"
) -> int:
    """Incrementally add new events without re-embedding the entire corpus."""
    new_events = load_events_csv(new_events_csv)
    unique_new = deduplicate_events(new_events)
    
    documents = prepare_event_documents(unique_new)
    
    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    added = 0
    for event_id, text, metadata in documents:
        embedding = embed_text(text)
        if embedding:
            collection.upsert(
                ids=[event_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text]
            )
        else:
            collection.upsert(
                ids=[event_id],
                metadatas=[metadata],
                documents=[text]
