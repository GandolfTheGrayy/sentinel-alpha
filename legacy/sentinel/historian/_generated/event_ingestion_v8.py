"""
Event Ingestion Pipeline for Sentinel Historian.

Reads historical market events from CSV and embeds them into ChromaDB
using Gemini's embedding API. Provides lookup and retrieval of similar
past events to contextualize current predictions.

This module enables the RAG pipeline to ground predictions in analogous
historical market movements and sentiment patterns.
"""

import csv
import os
from typing import Optional
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import numpy as np


# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def get_chromadb_client() -> chromadb.Client:
    """Return a persistent ChromaDB client configured for Sentinel."""
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./data/chromadb",
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def embed_text_gemini(text: str) -> list[float]:
    """Embed a text string using Gemini's embedding model."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set; cannot embed text.")
    
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return response["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    collection_name: str = "market_events",
    overwrite: bool = False,
) -> int:
    """
    Ingest historical market events from a CSV file into ChromaDB.
    
    Expected CSV columns: date, ticker, event_type, description, outcome.
    Returns the count of events ingested.
    """
    client = get_chromadb_client()
    
    # Create or get collection
    if overwrite:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    
    ingested_count = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            try:
                date = row.get("date", "").strip()
                ticker = row.get("ticker", "").strip()
                event_type = row.get("event_type", "").strip()
                description = row.get("description", "").strip()
                outcome = row.get("outcome", "").strip()
                
                if not all([date, ticker, description]):
                    print(f"Warning: Skipping row {row_idx + 1} (missing required fields)")
                    continue
                
                # Construct embedding document
                doc_text = (
                    f"Date: {date}. Ticker: {ticker}. Event: {event_type}. "
                    f"Description: {description}. Outcome: {outcome}."
                )
                
                # Embed the document
                embedding = embed_text_gemini(doc_text)
                
                # Upsert into ChromaDB
                doc_id = f"{ticker}_{date}_{row_idx}"
                collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[doc_text],
                    metadatas=[{
                        "date": date,
                        "ticker": ticker,
                        "event_type": event_type,
                        "outcome": outcome,
                    }],
                )
                
                ingested_count += 1
            
            except Exception as e:
                print(f"Error processing row {row_idx + 1}: {e}")
                continue
    
    return ingested_count


def query_similar_events(
    query_text: str,
    collection_name: str = "market_events",
    n_results: int = 5,
) -> list[dict]:
    """
    Query ChromaDB for similar historical events by embedding the query.
    
    Returns a list of dicts with keys: id, document, metadata, distance.
    """
    client = get_chromadb_client()
    
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Collection '{collection_name}' not found.")
        return []
    
    # Embed query
    query_embedding = embed_text_gemini(query_text)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )
    
    # Flatten results into a clean list
    output = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            output.append({
                "id": doc_id,
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    
    return output


def get_collection_stats(collection_name: str = "market_events") -> dict:
    """Return metadata and size stats for a ChromaDB collection."""
    client = get_chromadb_client()
    
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        return {
            "collection_name": collection_name,
            "doc_count": count,
            "status": "ok",
        }
    except ValueError:
        return {
            "collection_name": collection_name,
            "doc_count": 0,
            "status": "not_found",
        }


if __name__ == "__main__":
    # Example: ingest events from a sample CSV
    sample_csv = "./data/historical_events.csv"
    
    if os.path.exists(sample_csv):
        count = ingest_events_from_csv(sample_csv, overwrite=True)
        print(f"Ingested {count} events into ChromaDB.")
        
        stats = get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Example query
        query = "Major earnings miss caused sharp selloff in tech stock"
        results = query_similar_events(query, n_results=3)
        print(f"\nTop 3 similar events for query '{query}':")
        for r in results:
            print(f"  - {r['document'][:100]}... (distance: {r['distance']:.3f})")
    else:
        print(f"Sample CSV not found at {sample_csv}")
        print("Usage: python event_ingestion.py")
        print("  Requires: ./data/historical_events.csv with columns: date, ticker, event_type, description, outcome")
