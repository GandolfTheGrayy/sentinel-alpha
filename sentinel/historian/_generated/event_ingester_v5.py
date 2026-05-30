"""
Sentinel Historian — Historical Market Event Ingestion Pipeline.

This module reads a CSV of historical market events (e.g., earnings surprises,
regulatory announcements, product launches) and embeds them into ChromaDB for
RAG-based historical context lookup. Events are vectorized using Gemini's
embedding API and stored with metadata (date, ticker, event_type, source).

Integrated into the historian pillar to provide temporal context during
prediction synthesis in judge/predictor.py.
"""

import csv
import os
from typing import Optional
from datetime import datetime
import chromadb
import google.generativeai as genai
import pandas as pd


def initialize_event_collection(db_path: str = "sentinel_events.db") -> chromadb.Collection:
    """Initialize or fetch ChromaDB collection for historical events."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="market_events",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def embed_text_gemini(text: str) -> list[float]:
    """Embed a text string using Gemini embedding API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="SEMANTIC_SIMILARITY"
    )
    return response["embedding"]


def ingest_events_from_csv(
    csv_path: str,
    collection: chromadb.Collection,
    batch_size: int = 10
) -> dict[str, int]:
    """Ingest historical market events from CSV and embed into ChromaDB.
    
    CSV columns: date (YYYY-MM-DD), ticker, event_type, title, description, source.
    Returns count of successfully ingested events.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = {"date", "ticker", "event_type", "title", "description", "source"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing columns. Required: {required_cols}")
    
    ingested = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        try:
            date_str = row["date"]
            datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            skipped += 1
            continue
        
        ticker = str(row["ticker"]).upper()
        event_type = str(row["event_type"]).lower()
        title = str(row["title"])
        description = str(row["description"])
        source = str(row["source"])
        
        embedding_text = f"{title}. {description}"
        embedding = embed_text_gemini(embedding_text)
        
        event_id = f"{ticker}_{date_str}_{event_type}_{idx}"
        
        collection.add(
            ids=[event_id],
            embeddings=[embedding],
            metadatas=[{
                "ticker": ticker,
                "date": date_str,
                "event_type": event_type,
                "title": title,
                "source": source
            }],
            documents=[embedding_text]
        )
        
        ingested += 1
        
        if (ingested + skipped) % batch_size == 0:
            print(f"Progress: {ingested} ingested, {skipped} skipped")
    
    return {"ingested": ingested, "skipped": skipped}


def query_historical_events(
    collection: chromadb.Collection,
    query_text: str,
    ticker: Optional[str] = None,
    limit: int = 5
) -> list[dict]:
    """Query ChromaDB for historical events similar to query_text.
    
    Returns list of dicts with keys: id, ticker, date, event_type, title, source, distance.
    """
    embedding = embed_text_gemini(query_text)
    
    where_filter = None
    if ticker:
        where_filter = {"ticker": {"$eq": ticker.upper()}}
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=limit,
        where=where_filter
    )
    
    events = []
    if results and results["ids"] and len(results["ids"]) > 0:
        for i, event_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else None
            events.append({
                "id": event_id,
                "ticker": meta.get("ticker"),
                "date": meta.get("date"),
                "event_type": meta.get("event_type"),
                "title": meta.get("title"),
                "source": meta.get("source"),
                "distance": distance
            })
    
    return events


def main():
    """Demo: ingest sample events CSV and run a test query."""
    sample_csv = "sample_events.csv"
    
    if not os.path.exists(sample_csv):
        print(f"Creating sample CSV: {sample_csv}")
        sample_data = [
            ["date", "ticker", "event_type", "title", "description", "source"],
            ["2024-01-15", "AAPL", "earnings", "Q1 Earnings Beat", "Apple reported EPS of 2.18, beating estimates by 0.12.", "SEC"],
            ["2024-02-20", "TSLA", "product_launch", "Cybertruck Delivery Event", "Tesla delivered first Cybertrucks to customers.", "News"],
            ["2024-03-10", "MSFT", "regulatory", "Cloud Antitrust Review", "DOJ opened antitrust investigation into Microsoft cloud practices.", "Reuters"],
        ]
        with open(sample_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(sample_data)
        print(f"Sample CSV created: {sample_csv}")
    
    collection = initialize_event_collection()
    print("Initialized ChromaDB collection.")
    
    result = ingest_events_from_csv(sample_csv, collection)
    print(f"Ingestion complete: {result}")
    
    query_results = query_historical_events(collection, "Apple earnings surprise", ticker="AAPL", limit=3)
    print(f"Query results for 'Apple earnings surprise':")
    for event in query_results:
        print(f"  - {event['title']} ({event['date']}, dist={event['distance']:.3f})")


if __name__ == "__main__":
    main()
