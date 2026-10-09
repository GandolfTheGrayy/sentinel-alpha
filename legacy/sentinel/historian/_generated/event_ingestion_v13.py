"""
Sentinel Historian: Historical Market Event Ingestion Pipeline

This module ingests past market events from a CSV file and embeds them into
ChromaDB using Gemini's embedding API. Events are stored with metadata
(ticker, date, event_type) for retrieval during RAG queries.

Workflow:
  1. Load CSV with columns: ticker, date, event_type, description, impact
  2. Generate embeddings via Gemini's embedding-1 model
  3. Store vectors + metadata in ChromaDB collection
  4. Enable Historian RAG module to retrieve similar historical events

Used by: sentinel/historian/rag_query.py (context enrichment)
"""

import csv
import os
import sqlite3
from datetime import datetime
from typing import Optional

import chromadb
import google.generativeai as genai
import numpy as np

# Initialize Gemini for embeddings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load market events from CSV file (ticker, date, event_type, description, impact)."""
    events = []
    if not os.path.isfile(csv_path):
        print(f"Warning: CSV file {csv_path} not found; skipping ingestion.")
        return events

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                events.append(row)
    print(f"Loaded {len(events)} events from {csv_path}")
    return events


def embed_text_gemini(text: str) -> Optional[list[float]]:
    """Generate embedding for text using Gemini's embedding-1 model."""
    if not GEMINI_API_KEY:
        print("Warning: GEMINI_API_KEY not set; returning zero vector.")
        return [0.0] * 768

    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return result["embedding"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None


def ingest_events_to_chromadb(
    events: list[dict], collection_name: str = "historical_events"
) -> chromadb.Collection:
    """
    Embed and store market events in ChromaDB.
    
    Returns ChromaDB collection with vectors, metadata, and document IDs.
    """
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    for i, event in enumerate(events):
        ticker = event.get("ticker", "UNKNOWN")
        date_str = event.get("date", "")
        event_type = event.get("event_type", "")
        description = event.get("description", "")
        impact = event.get("impact", "neutral")

        # Construct embedding text
        embed_text = f"{ticker} {event_type} {date_str} {description}"
        vector = embed_text_gemini(embed_text)

        if vector is None:
            print(f"Skipping event {i} due to embedding failure")
            continue

        doc_id = f"event_{ticker}_{date_str}_{i}"
        metadata = {
            "ticker": ticker,
            "date": date_str,
            "event_type": event_type,
            "impact": impact,
        }

        collection.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[description],
            metadatas=[metadata],
        )

    print(f"Ingested {len(events)} events into ChromaDB collection '{collection_name}'")
    return collection


def load_or_ingest_events(
    csv_path: str, collection_name: str = "historical_events"
) -> chromadb.Collection:
    """
    Convenience function: load CSV and ingest to ChromaDB in one call.
    
    Returns ChromaDB collection ready for RAG queries.
    """
    events = load_events_from_csv(csv_path)
    if not events:
        print("No events loaded; returning empty ChromaDB collection.")
        client = chromadb.EphemeralClient()
        return client.get_or_create_collection(name=collection_name)

    return ingest_events_to_chromadb(events, collection_name=collection_name)


def query_events_by_embedding(
    collection: chromadb.Collection, query_text: str, n_results: int = 5
) -> Optional[dict]:
    """
    Query ChromaDB collection for similar historical events using embedding.
    
    Returns dict with 'ids', 'documents', 'distances', 'metadatas'.
    """
    query_embedding = embed_text_gemini(query_text)
    if query_embedding is None:
        print("Failed to embed query text")
        return None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    return results


if __name__ == "__main__":
    # Example: ingest a sample events CSV
    sample_csv = "data/historical_events.csv"
    col = load_or_ingest_events(sample_csv)
    
    # Example query
    query_result = query_events_by_embedding(
        col, "Apple earnings miss Q3", n_results=3
    )
    if query_result:
        print("Query results:")
        print(f"  IDs: {query_result['ids']}")
        print(f"  Documents: {query_result['documents']}")
        print(f"  Metadatas: {query_result['metadatas']}")
