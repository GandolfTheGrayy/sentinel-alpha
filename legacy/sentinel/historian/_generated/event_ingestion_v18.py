"""
Historical market event ingestion pipeline for Sentinel.

Reads a CSV of past market events (dates, tickers, event types, descriptions)
and embeds them into ChromaDB using Gemini's embedding API. This corpus enables
the RAG historian to retrieve similar past events when analyzing current signals,
grounding predictions in historical precedent and anomaly detection.

Role: Historian pillar — prepares the event knowledge base that rag_query.py
queries against during per-ticker analysis.
"""

import csv
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
import google.generativeai as genai
from chromadb.config import Settings


def _initialize_chroma_client(persist_dir: str = "data/chroma_db") -> chromadb.Client:
    """Initialize ChromaDB client with persistence."""
    os.makedirs(persist_dir, exist_ok=True)
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_dir,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


def _get_or_create_collection(
    client: chromadb.Client, collection_name: str = "market_events"
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for market events."""
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_text_gemini(text: str, api_key: Optional[str] = None) -> list[float]:
    """Embed text using Gemini embedding API."""
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
    )
    return response["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    chroma_client: chromadb.Client,
    collection_name: str = "market_events",
    api_key: Optional[str] = None,
) -> int:
    """
    Ingest market events from CSV and embed them into ChromaDB.

    Expected CSV columns: date, ticker, event_type, description, source.
    Returns count of successfully ingested events.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    collection = _get_or_create_collection(chroma_client, collection_name)
    ingested = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or invalid")

        for row_idx, row in enumerate(reader, start=2):
            try:
                date_str = row.get("date", "").strip()
                ticker = row.get("ticker", "").strip().upper()
                event_type = row.get("event_type", "").strip()
                description = row.get("description", "").strip()
                source = row.get("source", "unknown").strip()

                if not all([date_str, ticker, event_type, description]):
                    print(f"Warning: Row {row_idx} missing required fields, skipping")
                    continue

                # Validate date format
                try:
                    event_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    print(f"Warning: Row {row_idx} has invalid date '{date_str}', skipping")
                    continue

                # Create embedding text
                embedding_text = (
                    f"Event: {event_type}\n"
                    f"Ticker: {ticker}\n"
                    f"Date: {date_str}\n"
                    f"Description: {description}"
                )

                # Embed via Gemini
                embedding = _embed_text_gemini(embedding_text, api_key)

                # Create unique doc ID
                doc_id = f"{ticker}_{date_str}_{event_type.replace(' ', '_')}_{row_idx}"

                # Add to ChromaDB
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[description],
                    metadatas=[
                        {
                            "ticker": ticker,
                            "date": date_str,
                            "event_type": event_type,
                            "source": source,
                            "timestamp": event_date.isoformat(),
                        }
                    ],
                )

                ingested += 1

            except Exception as e:
                print(f"Error processing row {row_idx}: {e}")
                continue

    collection.persist()
    return ingested


def query_similar_events(
    query_text: str,
    ticker: Optional[str] = None,
    n_results: int = 5,
    chroma_client: Optional[chromadb.Client] = None,
    collection_name: str = "market_events",
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Query ChromaDB for similar historical events using semantic search.

    Returns list of dicts with keys: id, document, metadata, distance.
    """
    if chroma_client is None:
        chroma_client = _initialize_chroma_client()

    collection = _get_or_create_collection(chroma_client, collection_name)

    # Embed the query
    query_embedding = _embed_text_gemini(query_text, api_key)

    # Search
    where_filter = None
    if ticker:
        where_filter = {"ticker": ticker}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    # Format results
    formatted = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for idx, doc_id in enumerate(results["ids"][0]):
            formatted.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "distance": results["distances"][0][idx] if results["distances"] else None,
                }
            )

    return formatted


def clear_collection(
    chroma_client: chromadb.Client,
    collection_name: str = "market_events",
) -> None:
    """Delete all documents from a ChromaDB collection."""
    collection = _get_or_create_collection(chroma_client, collection_name)
    # Get all IDs and delete them
    data = collection.get()
    if data and data.get("ids"):
        collection.delete(ids=data["ids"])
    collection.persist()


if __name__ == "__main__":
    # Example usage: ingest sample events CSV and demo query
    sample_csv = "data/sample_events.csv"

    # Create sample CSV if it doesn't exist
    if not os.path.exists(sample_csv):
        os.makedirs("data", exist_ok=True)
        with open(sample_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["date", "ticker", "event_type", "description", "source"]
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "date": "2023-01-15",
                        "ticker": "AAPL",
                        "event_type": "Earnings Miss",
                        "description": "Apple reported Q4 revenue below analyst expectations due to iPhone demand softness",
                        "source": "SEC 8-K",
                    },
                    {
                        "date": "
