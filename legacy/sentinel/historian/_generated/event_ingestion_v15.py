"""
Sentinel historian event ingestion pipeline.

Reads historical market events from CSV, embeds them into ChromaDB using Gemini's
embedding API, and provides lookup utilities for the RAG pipeline. This module
transforms raw event data into queryable vector embeddings for contextual
analysis during prediction calibration.
"""

import csv
import os
from pathlib import Path
from typing import Optional
import json

import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CHROMA_DB_PATH = Path(__file__).parent.parent / ".chroma_db"


def init_chroma_collection(collection_name: str = "market_events") -> chromadb.Collection:
    """Initialize or retrieve ChromaDB collection for market events."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(CHROMA_DB_PATH),
        anonymized_telemetry=False,
    )
    client = chromadb.Client(settings)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def embed_text_gemini(text: str) -> Optional[list[float]]:
    """Embed text using Gemini embedding API."""
    if not GEMINI_API_KEY:
        return None
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="SEMANTIC_SIMILARITY",
        )
        return response["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def ingest_events_from_csv(csv_path: str, collection: Optional[chromadb.Collection] = None) -> int:
    """Ingest market events from CSV file into ChromaDB."""
    if collection is None:
        collection = init_chroma_collection()
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    ingested_count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file is empty or malformed")
        
        for idx, row in enumerate(reader):
            # Expect columns: date, ticker, event_type, description, impact_direction
            date = row.get("date", "").strip()
            ticker = row.get("ticker", "").strip()
            event_type = row.get("event_type", "").strip()
            description = row.get("description", "").strip()
            impact_direction = row.get("impact_direction", "neutral").strip().lower()
            
            if not all([date, ticker, event_type, description]):
                print(f"Skipping row {idx + 2}: missing required fields")
                continue
            
            # Create embedding-friendly text
            full_text = f"{ticker} {event_type}: {description} (date: {date})"
            embedding = embed_text_gemini(full_text)
            
            if embedding is None:
                print(f"Failed to embed row {idx + 2}, skipping")
                continue
            
            doc_id = f"{ticker}_{date}_{idx}"
            metadata = {
                "date": date,
                "ticker": ticker,
                "event_type": event_type,
                "impact_direction": impact_direction,
            }
            
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[description],
                metadatas=[metadata],
            )
            ingested_count += 1
    
    return ingested_count


def query_events(query_text: str, collection: Optional[chromadb.Collection] = None, n_results: int = 5) -> list[dict]:
    """Query ChromaDB for similar historical events."""
    if collection is None:
        collection = init_chroma_collection()
    
    query_embedding = embed_text_gemini(query_text)
    if query_embedding is None:
        return []
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    events = []
    if results["ids"] and len(results["ids"]) > 0:
        for doc_id, distance, metadata, document in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0],
        ):
            events.append({
                "doc_id": doc_id,
                "similarity_score": 1 - distance,
                "metadata": metadata,
                "description": document,
            })
    
    return events


def get_events_by_ticker(ticker: str, collection: Optional[chromadb.Collection] = None) -> list[dict]:
    """Retrieve all events for a specific ticker."""
    if collection is None:
        collection = init_chroma_collection()
    
    results = collection.get(
        where={"ticker": ticker},
    )
    
    events = []
    if results["ids"]:
        for doc_id, metadata, document in zip(
            results["ids"],
            results["metadatas"],
            results["documents"],
        ):
            events.append({
                "doc_id": doc_id,
                "metadata": metadata,
                "description": document,
            })
    
    return events


def export_collection_stats(collection: Optional[chromadb.Collection] = None) -> dict:
    """Export collection statistics for monitoring."""
    if collection is None:
        collection = init_chroma_collection()
    
    count = collection.count()
    return {
        "total_events": count,
        "collection_name": collection.name,
        "embedding_model": "models/embedding-001",
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python event_ingestion.py <csv_path> [query_text]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    col = init_chroma_collection()
    
    print(f"Ingesting events from {csv_file}...")
    count = ingest_events_from_csv(csv_file, col)
    print(f"✓ Ingested {count} events")
    
    stats = export_collection_stats(col)
    print(f"Collection stats: {json.dumps(stats, indent=2)}")
    
    if len(sys.argv) > 2:
        query = sys.argv[2]
        print(f"\nQuerying: '{query}'")
        matches = query_events(query, col, n_results=3)
        for match in matches:
            print(f"  - {match['metadata']['ticker']} ({match['metadata']['date']}): "
                  f"{match['description'][:60]}... [score: {match['similarity_score']:.3f}]")
