"""
Historical market event ingestion pipeline for Sentinel.

Reads past market events from CSV, embeds them using Gemini's embedding API,
and stores them in ChromaDB for RAG retrieval. Complements live sentiment signals
with anchored historical context: "When did similar events happen? How did markets react?"

Part of sentinel/historian/ — the RAG backbone that grounds predictions in precedent.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any

import chromadb
import google.generativeai as genai
import numpy as np
import pandas as pd

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CHROMA_DB_PATH = Path("sentinel/data/chroma_db")
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)


def load_events_csv(csv_path: str) -> pd.DataFrame:
    """Load market events from CSV (date, ticker, event_type, description, price_change_pct)."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Event CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def embed_text_with_gemini(text: str) -> list[float]:
    """Embed text using Gemini embedding API; return embedding vector."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set; cannot embed text.")
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return response["embedding"]
    except Exception as e:
        raise RuntimeError(f"Gemini embedding failed: {e}")


def ingest_events_to_chroma(
    csv_path: str,
    collection_name: str = "market_events"
) -> dict[str, Any]:
    """
    Ingest CSV events into ChromaDB, embedding each event description.

    Returns metadata: count ingested, collection name, DB path.
    """
    df = load_events_csv(csv_path)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    count = 0
    for idx, row in df.iterrows():
        event_id = f"{row.get('ticker', 'UNKNOWN')}_{row.get('date', 'NODATE')}_{idx}"
        description = row.get("description", "")
        event_type = row.get("event_type", "UNKNOWN")
        date_str = row.get("date", "")
        ticker = row.get("ticker", "")
        price_change = row.get("price_change_pct", 0.0)

        text_to_embed = f"{event_type} {description}"
        embedding = embed_text_with_gemini(text_to_embed)

        metadata = {
            "date": date_str,
            "ticker": ticker,
            "event_type": event_type,
            "price_change_pct": float(price_change),
        }

        collection.add(
            ids=[event_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[description]
        )
        count += 1

    return {
        "count_ingested": count,
        "collection_name": collection_name,
        "db_path": str(CHROMA_DB_PATH),
    }


def query_similar_events(
    query_text: str,
    collection_name: str = "market_events",
    n_results: int = 5
) -> list[dict[str, Any]]:
    """
    Query ChromaDB for historical events similar to query_text; return top N with metadata.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set; cannot query.")

    query_embedding = embed_text_with_gemini(query_text)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    output = []
    if results and results.get("ids") and len(results["ids"]) > 0:
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
            document = results["documents"][0][i] if results.get("documents") else ""
            distance = results["distances"][0][i] if results.get("distances") else None

            output.append({
                "id": doc_id,
                "document": document,
                "metadata": metadata,
                "similarity_distance": distance,
            })

    return output


def get_collection_stats(collection_name: str = "market_events") -> dict[str, Any]:
    """Retrieve count and metadata about a ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=collection_name)
    count = collection.count()
    return {
        "collection_name": collection_name,
        "total_events": count,
        "db_path": str(CHROMA_DB_PATH),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python event_ingester.py <csv_path> [collection_name]")
        sys.exit(1)

    csv_file = sys.argv[1]
    coll_name = sys.argv[2] if len(sys.argv) > 2 else "market_events"

    result = ingest_events_to_chroma(csv_file, coll_name)
    print(json.dumps(result, indent=2))

    stats = get_collection_stats(coll_name)
    print(json.dumps(stats, indent=2))
