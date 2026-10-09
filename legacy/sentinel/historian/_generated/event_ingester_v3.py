"""
Historical market event ingestion pipeline for Sentinel.

This module reads past market events from a CSV file and embeds them into
ChromaDB using Gemini's embedding API. Events are indexed by ticker and
event type, enabling RAG lookups in historian/rag_query.py to surface
relevant historical precedents when scoring price movements.

Typical workflow:
  1. Call load_events_from_csv(csv_path) to parse the CSV
  2. Call embed_and_ingest(events, collection) to vectorize and store
  3. Historian RAG queries will find similar events via similarity search
"""

import os
import csv
from typing import List, Dict, Any, Optional
import chromadb
import google.generativeai as genai


def load_events_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """Load market events from a CSV file and return as list of dicts."""
    events = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and any(row.values()):  # Skip empty rows
                events.append(row)
    
    return events


def embed_text(text: str, api_key: Optional[str] = None) -> List[float]:
    """Generate embedding vector for a text snippet using Gemini API."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    genai.configure(api_key=api_key)
    
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    return response['embedding']


def embed_and_ingest(
    events: List[Dict[str, Any]],
    collection: chromadb.Collection,
    ticker_field: str = "ticker",
    event_field: str = "event_description",
    date_field: str = "date"
) -> int:
    """
    Embed events and ingest into ChromaDB collection.
    
    Returns the count of events successfully ingested.
    """
    if not events:
        return 0
    
    ingested = 0
    
    for idx, event in enumerate(events):
        ticker = event.get(ticker_field, "UNKNOWN").upper()
        description = event.get(event_field, "")
        date_str = event.get(date_field, "")
        
        if not description:
            continue
        
        # Create a rich text for embedding: concatenate key fields
        embedding_text = f"{ticker} {date_str}: {description}"
        
        # Generate embedding
        embedding = embed_text(embedding_text)
        
        # Create a unique document ID
        doc_id = f"event_{ticker}_{idx}_{date_str}".replace(" ", "_").replace("/", "-")
        
        # Store metadata
        metadata = {
            "ticker": ticker,
            "date": date_str,
            "event_type": event.get("event_type", "unknown"),
            "source": event.get("source", "csv"),
        }
        
        # Ingest into ChromaDB
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[metadata]
        )
        
        ingested += 1
    
    return ingested


def initialize_event_collection(
    db_path: str = "./sentinel_events.db",
    collection_name: str = "historical_events"
) -> chromadb.Collection:
    """Initialize or retrieve a ChromaDB collection for historical events."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def ingest_csv_into_chromadb(
    csv_path: str,
    db_path: str = "./sentinel_events.db",
    collection_name: str = "historical_events"
) -> Dict[str, Any]:
    """
    End-to-end pipeline: load CSV, embed, and ingest into ChromaDB.
    
    Returns a summary dict with event count, ingestion status, and DB path.
    """
    events = load_events_from_csv(csv_path)
    collection = initialize_event_collection(db_path, collection_name)
    ingested = embed_and_ingest(events, collection)
    
    return {
        "total_events": len(events),
        "ingested": ingested,
        "db_path": db_path,
        "collection_name": collection_name,
        "status": "success" if ingested > 0 else "no_events"
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python event_ingester.py <csv_path> [db_path]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    db_file = sys.argv[2] if len(sys.argv) > 2 else "./sentinel_events.db"
    
    result = ingest_csv_into_chromadb(csv_file, db_file)
    print(f"Ingestion result: {result}")
