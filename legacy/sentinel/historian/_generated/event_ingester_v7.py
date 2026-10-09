"""
Historical market event ingestion pipeline for Sentinel.

Reads a CSV of past market events (e.g., earnings surprises, regulatory actions,
macroeconomic shocks) and embeds them into ChromaDB via Gemini embeddings.
These embeddings enable RAG lookup during prediction to contextualize current
sentiment signals against similar historical precedents.

This module is part of the historian pillar and feeds the vector store that
rag_query.py queries during judge reasoning.
"""

import csv
import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
import google.generativeai as genai


def _init_chromadb(db_path: str = "sentinel/data/chromadb") -> chromadb.Client:
    """Initialize ChromaDB client with persistent storage at db_path."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=db_path,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def _embed_text(text: str) -> list[float]:
    """Embed a single text string using Gemini embeddings."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )
    return result["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    collection_name: str = "historical_events",
    db_path: str = "sentinel/data/chromadb",
) -> dict:
    """
    Ingest historical market events from CSV into ChromaDB.
    
    Expected CSV columns: date, ticker, event_type, description, impact_direction, impact_magnitude.
    Each row is embedded and stored with metadata for later RAG retrieval.
    Returns a summary dict with row counts and any errors encountered.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    client = _init_chromadb(db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Historical market events for RAG context"}
    )
    
    ingested = 0
    errors = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV is empty or malformed")
        
        required_fields = {"date", "ticker", "event_type", "description"}
        if not required_fields.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"CSV missing required columns. Expected: {required_fields}, "
                f"Got: {set(reader.fieldnames or [])}"
            )
        
        for row_idx, row in enumerate(reader, start=2):
            try:
                date = row.get("date", "").strip()
                ticker = row.get("ticker", "").strip().upper()
                event_type = row.get("event_type", "").strip()
                description = row.get("description", "").strip()
                impact_direction = row.get("impact_direction", "neutral").strip().lower()
                impact_magnitude = row.get("impact_magnitude", "0").strip()
                
                if not all([date, ticker, event_type, description]):
                    errors.append(f"Row {row_idx}: missing required fields")
                    continue
                
                # Construct document text for embedding
                doc_text = (
                    f"Date: {date}. Ticker: {ticker}. Event: {event_type}. "
                    f"Details: {description}. Impact: {impact_direction} ({impact_magnitude})."
                )
                
                # Embed the document
                embedding = _embed_text(doc_text)
                
                # Create unique ID from date, ticker, and event type
                doc_id = f"{date}_{ticker}_{event_type.replace(' ', '_')}"
                
                # Add to collection
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc_text],
                    metadatas=[{
                        "date": date,
                        "ticker": ticker,
                        "event_type": event_type,
                        "impact_direction": impact_direction,
                        "impact_magnitude": impact_magnitude,
                    }]
                )
                ingested += 1
                
            except Exception as e:
                errors.append(f"Row {row_idx}: {str(e)}")
    
    # Persist the collection
    client.persist()
    
    return {
        "ingested": ingested,
        "errors": errors,
        "collection_name": collection_name,
        "db_path": db_path,
    }


def query_similar_events(
    query_text: str,
    collection_name: str = "historical_events",
    db_path: str = "sentinel/data/chromadb",
    n_results: int = 5,
) -> list[dict]:
    """
    Query ChromaDB for historical events similar to query_text using embeddings.
    
    Returns a list of dicts with keys: id, document, metadata, distance.
    """
    client = _init_chromadb(db_path)
    collection = client.get_collection(name=collection_name)
    
    query_embedding = _embed_text(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    # Flatten results into a list of dicts for easier consumption
    output = []
    if results and results.get("ids"):
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i] if results.get("documents") else None,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    
    return output


def clear_collection(
    collection_name: str = "historical_events",
    db_path: str = "sentinel/data/chromadb",
) -> None:
    """Delete all documents from a collection (useful for re-ingestion)."""
    client = _init_chromadb(db_path)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection may not exist


if __name__ == "__main__":
    # Example usage: ingest a sample CSV and query
    sample_csv = "sentinel/data/sample_events.csv"
    
    if Path(sample_csv).exists():
        print(f"Ingesting events from {sample_csv}...")
        result = ingest_events_from_csv(sample_csv)
        print(f"✓ Ingested {result['ingested']} events")
        if result["errors"]:
            print(f"⚠ Encountered {len(result['errors'])} errors")
            for err in result["errors"][:5]:
                print(f"  - {err}")
        
        # Test a query
        print("\nQuerying similar events to: 'Apple earnings surprise'...")
        hits = query_similar_events("Apple earnings surprise", n_results=3)
        for hit in hits:
            print(f"  • {hit['metadata'].get('date')} {hit['metadata'].get('ticker')}: "
                  f"{hit['metadata'].get('event_type')}")
    else:
        print(f"Sample CSV not found at {sample_csv}")
        print("To use this module, provide a CSV with columns:")
