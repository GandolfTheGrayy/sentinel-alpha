"""
Sentinel Historian — Historical Market Event Ingestion Pipeline.

Reads a CSV of past market events (earnings surprises, regulatory actions,
competitor news, etc.) and embeds them into ChromaDB using Gemini's embedding
API. Enables RAG queries to surface historical precedent when analyzing
current sentiment signals.

Part of the Sentinel Sentiment Engine's historian pillar: provides
time-series context for confidence scoring and anomaly detection.
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
import google.generativeai as genai
from chromadb.config import Settings


def _init_chromadb_client() -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage in sentinel/data."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=str(data_dir / "chroma_events"),
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def _init_gemini() -> None:
    """Initialize Gemini API client from environment."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)


def embed_text(text: str) -> list[float]:
    """Embed a single text string using Gemini's embedding model."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return response["embedding"]


def ingest_events_from_csv(csv_path: str, collection_name: str = "market_events") -> dict[str, Any]:
    """
    Read historical market events from CSV and embed into ChromaDB.
    
    Expected CSV columns: date, ticker, event_type, description, impact_direction.
    Returns a dict with ingestion stats: {total_read, total_embedded, errors}.
    """
    _init_gemini()
    client = _init_chromadb_client()
    
    # Delete collection if exists to allow re-ingestion.
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Historical market events with embeddings"}
    )
    
    stats = {"total_read": 0, "total_embedded": 0, "errors": []}
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                stats["total_read"] += 1
                
                try:
                    # Validate required fields.
                    date_str = row.get("date", "").strip()
                    ticker = row.get("ticker", "").strip().upper()
                    event_type = row.get("event_type", "").strip()
                    description = row.get("description", "").strip()
                    impact = row.get("impact_direction", "").strip()
                    
                    if not all([date_str, ticker, event_type, description]):
                        stats["errors"].append(
                            f"Row {row_idx}: Missing required field (date, ticker, event_type, description)"
                        )
                        continue
                    
                    # Validate date format.
                    try:
                        event_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        stats["errors"].append(
                            f"Row {row_idx}: Invalid date format '{date_str}' (expected YYYY-MM-DD)"
                        )
                        continue
                    
                    # Build embedding text: combine structured fields for context.
                    embedding_text = f"{ticker} {event_type} on {date_str}: {description}"
                    
                    # Embed via Gemini.
                    embedding = embed_text(embedding_text)
                    
                    # Store in ChromaDB with metadata.
                    doc_id = f"{ticker}_{event_date.timestamp()}_{row_idx}"
                    collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[description],
                        metadatas=[{
                            "date": date_str,
                            "ticker": ticker,
                            "event_type": event_type,
                            "impact_direction": impact or "neutral",
                            "embedding_text": embedding_text,
                            "ingested_at": datetime.utcnow().isoformat(),
                        }]
                    )
                    stats["total_embedded"] += 1
                
                except Exception as e:
                    stats["errors"].append(f"Row {row_idx}: {str(e)}")
    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    return stats


def query_historical_events(
    query_text: str,
    collection_name: str = "market_events",
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Query historical events by semantic similarity.
    
    Returns a list of dicts with keys: {id, description, metadata, distance}.
    """
    _init_gemini()
    client = _init_chromadb_client()
    
    collection = client.get_collection(name=collection_name)
    query_embedding = embed_text(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    # Flatten ChromaDB's nested result structure.
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "id": results["ids"][0][i],
            "description": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    
    return formatted


def get_collection_stats(collection_name: str = "market_events") -> dict[str, Any]:
    """Fetch stats about the ChromaDB collection (count, metadata sample)."""
    _init_gemini()
    client = _init_chromadb_client()
    
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        
        # Sample 1 document if collection is non-empty.
        sample = None
        if count > 0:
            peek = collection.get(limit=1)
            if peek["ids"]:
                sample = {
                    "id": peek["ids"][0],
                    "metadata": peek["metadatas"][0],
                }
        
        return {
            "collection_name": collection_name,
            "total_documents": count,
            "sample": sample,
        }
    except Exception as e:
        return {
            "collection_name": collection_name,
            "error": str(e),
            "total_documents": 0,
        }


if __name__ == "__main__":
    # Example usage: ingest from a sample CSV.
    sample_csv = Path(__file__).parent.parent.parent / "data" / "sample_events.csv"
    
    if sample_csv.exists():
        print(f"Ingesting events from {sample_csv}...")
        stats = ingest_events_from_csv(str(sample_csv))
        print(f"Ingestion complete: {json.dumps(stats, indent=2)}")
        
        print("\nCollection stats:")
        cstats = get_collection_stats()
        print(json.dumps(cstats, indent=2))
        
        print("\nSample query: 'earnings
