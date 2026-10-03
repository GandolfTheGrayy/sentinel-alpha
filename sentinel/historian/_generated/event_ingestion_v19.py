"""
Historical market event ingestion pipeline for Sentinel.

Reads past market events (earnings surprises, regulatory actions, etc.) from CSV,
embeds them into ChromaDB using Gemini embeddings, and indexes them for RAG lookups.
Integrates with sentinel/historian/rag_query.py for historical context retrieval.
"""

import csv
import os
from pathlib import Path
from typing import Optional
import sqlite3

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np


def load_events_from_csv(csv_path: str) -> list[dict]:
    """Load market events from a CSV file with columns: date, ticker, event_type, description, impact."""
    events = []
    if not os.path.exists(csv_path):
        return events
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get("ticker") and row.get("description"):
                events.append(row)
    
    return events


def embed_text_gemini(text: str, api_key: Optional[str] = None) -> list[float]:
    """Embed a single text string using Gemini embeddings API."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    genai.configure(api_key=api_key)
    
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY"
        )
        return result["embedding"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return [0.0] * 768  # fallback zero vector


def ingest_events_to_chromadb(
    events: list[dict],
    db_path: str = "sentinel_events.db",
    collection_name: str = "market_events"
) -> chromadb.Collection:
    """
    Ingest a list of events into ChromaDB with Gemini embeddings.
    
    Returns the ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name)
    
    for idx, event in enumerate(events):
        event_id = f"{event.get('ticker', 'UNKNOWN')}_{event.get('date', 'NODATE')}_{idx}"
        
        # Build document text from event fields
        doc_text = (
            f"Date: {event.get('date', 'N/A')} | "
            f"Ticker: {event.get('ticker', 'N/A')} | "
            f"Type: {event.get('event_type', 'N/A')} | "
            f"Description: {event.get('description', 'N/A')} | "
            f"Impact: {event.get('impact', 'N/A')}"
        )
        
        # Embed the document
        embedding = embed_text_gemini(doc_text)
        
        # Add to ChromaDB
        collection.add(
            ids=[event_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[{
                "date": event.get("date", ""),
                "ticker": event.get("ticker", "").upper(),
                "event_type": event.get("event_type", ""),
                "impact": event.get("impact", "")
            }]
        )
    
    return collection


def query_events_by_similarity(
    query_text: str,
    collection: chromadb.Collection,
    n_results: int = 5
) -> list[dict]:
    """
    Query ChromaDB collection for similar historical events using semantic search.
    
    Returns list of dicts with 'document', 'metadata', and 'distance' keys.
    """
    query_embedding = embed_text_gemini(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    output = []
    if results and results.get("documents"):
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "document": doc,
                "metadata": metadata,
                "distance": distance
            })
    
    return output


def initialize_default_events_db(csv_path: str, db_path: str = "sentinel_events.db") -> None:
    """
    One-shot ingestion: read CSV file and populate ChromaDB collection.
    
    Call this once at startup or during initialization.
    """
    events = load_events_from_csv(csv_path)
    if events:
        ingest_events_to_chromadb(events, db_path=db_path)
        print(f"Ingested {len(events)} events into {db_path}")
    else:
        print(f"No events found in {csv_path}")


if __name__ == "__main__":
    # Example usage: load events from a CSV and ingest them
    example_csv = "market_events.csv"  # Place this file in project root
    
    if os.path.exists(example_csv):
        initialize_default_events_db(example_csv)
        
        # Test query
        client = chromadb.PersistentClient(path="sentinel_events.db")
        collection = client.get_collection("market_events")
        
        test_query = "Apple earnings miss"
        results = query_events_by_similarity(test_query, collection, n_results=3)
        
        print(f"\nQuery: '{test_query}'")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['document'][:100]}... (distance: {result['distance']:.3f})")
    else:
        print(f"{example_csv} not found. Create it with columns: date,ticker,event_type,description,impact")
