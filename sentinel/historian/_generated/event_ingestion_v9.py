"""
Sentinel Historian — Historical Market Event Ingestion Pipeline

Reads CSV files containing historical market events (earnings, acquisitions,
regulatory actions, etc.) and embeds them into ChromaDB for RAG retrieval.
Used by historian/rag_query.py to provide historical context when predicting
future price movements.

Flow:
  1. Load CSV (ticker, date, event_type, description, impact_direction)
  2. Generate embeddings via Gemini (high-volume text encoding)
  3. Upsert into ChromaDB collection with metadata (ticker, date, type)
  4. Return collection handle for rag_query.py to retrieve similar events
"""

import os
import csv
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np


def load_event_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load historical events from CSV file with columns: ticker, date, event_type, description, impact_direction."""
    events = []
    if not os.path.exists(csv_path):
        return events
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get("ticker") and row.get("description"):
                events.append({
                    "ticker": row.get("ticker", "").strip().upper(),
                    "date": row.get("date", "").strip(),
                    "event_type": row.get("event_type", "OTHER").strip().upper(),
                    "description": row.get("description", "").strip(),
                    "impact_direction": row.get("impact_direction", "NEUTRAL").strip().upper(),
                })
    return events


def embed_event_texts(texts: List[str]) -> List[List[float]]:
    """Embed list of event descriptions using Gemini embedding model."""
    if not texts:
        return []
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set; cannot generate embeddings.")
    
    genai.configure(api_key=api_key)
    embeddings = []
    
    for text in texts:
        if not text or len(text.strip()) == 0:
            embeddings.append([0.0] * 768)
            continue
        
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = result.get("embedding", [0.0] * 768)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Warning: embedding failed for text '{text[:50]}...': {e}")
            embeddings.append([0.0] * 768)
    
    return embeddings


def initialize_chromadb(db_path: str = "./chroma_data") -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client


def ingest_events_to_chromadb(
    events: List[Dict[str, Any]],
    collection_name: str = "market_events",
    db_path: str = "./chroma_data"
) -> chromadb.Collection:
    """
    Ingest historical events into ChromaDB with embeddings and metadata.
    
    Returns the collection handle for querying.
    """
    if not events:
        print("No events to ingest.")
        client = initialize_chromadb(db_path)
        return client.get_or_create_collection(name=collection_name)
    
    texts = [
        f"{e['ticker']} {e['event_type']}: {e['description']} (Impact: {e['impact_direction']})"
        for e in events
    ]
    print(f"Embedding {len(texts)} events...")
    embeddings = embed_event_texts(texts)
    
    client = initialize_chromadb(db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = []
    metadatas = []
    for i, event in enumerate(events):
        event_id = f"{event['ticker']}_{event.get('date', 'unknown').replace('-', '')}_{i}"
        ids.append(event_id)
        metadatas.append({
            "ticker": event["ticker"],
            "date": event.get("date", "unknown"),
            "event_type": event["event_type"],
            "impact_direction": event["impact_direction"],
            "description": event["description"][:500],
        })
    
    print(f"Upserting {len(ids)} events into collection '{collection_name}'...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Ingestion complete. Collection now has {collection.count()} events.")
    return collection


def query_similar_events(
    collection: chromadb.Collection,
    query_text: str,
    ticker: Optional[str] = None,
    n_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB collection for events similar to query_text, optionally filtered by ticker.
    
    Returns list of dicts with 'id', 'document', 'metadata', 'distance'.
    """
    if not query_text or len(query_text.strip()) == 0:
        return []
    
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"ticker": ticker} if ticker else None
        )
        
        output = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                output.append({
                    "id": doc_id,
                    "document": results["documents"][0][i] if i < len(results.get("documents", [[]])[0]) else "",
                    "metadata": results["metadatas"][0][i] if i < len(results.get("metadatas", [[]])[0]) else {},
                    "distance": results["distances"][0][i] if i < len(results.get("distances", [[]])[0]) else 1.0,
                })
        return output
    except Exception as e:
        print(f"Error querying collection: {e}")
        return []


def build_event_corpus(csv_path: str, db_path: str = "./chroma_data") -> chromadb.Collection:
    """
    End-to-end pipeline: load CSV, embed events, ingest into ChromaDB, return collection.
    
    Typical usage: corpus = build_event_corpus("historical_events.csv")
    """
    print(f"Loading events from {csv_path}...")
    events = load_event_csv(csv_path)
    print(f"Loaded {len(events)} events.")
    
    if not events:
        print("No events loaded; returning empty collection.")
        client = initialize_chromadb(db_path)
        return client.get_or_create_collection(name="market_events")
    
    collection = ingest_events_to_chromadb(events, db_path=db_path)
    return collection


if __name__ == "__main__":
    csv_file = "historical_events.csv"
    print(f"Building event corpus from {csv_file}...")
    col = build_event_corpus(csv_file)
    
    query = "APPLE earnings beat revenue expectations"
    print(f"\
