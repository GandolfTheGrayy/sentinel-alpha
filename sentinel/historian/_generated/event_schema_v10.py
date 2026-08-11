"""
Sentinel Historian Event Schema.

Defines dataclasses for MarketEvent, HistoricalMatch, and ConfidenceReport
used throughout the Historian layer (RAG pipeline, event lookup, confidence
weighting). These schemas standardize communication between Scout (data
ingestion), Linguist (sentiment analysis), and Judge (prediction + post-mortem).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List


@dataclass
class MarketEvent:
    """
    Represents a discrete market-moving event (earnings, filing, headline, etc.).
    
    Used by Scout to ingest raw events and by Historian to store/retrieve
    from ChromaDB vector DB.
    """
    
    event_id: str
    """Unique identifier (e.g., 'AAPL-20240115-earnings')."""
    
    ticker: str
    """Stock ticker symbol (uppercase)."""
    
    event_type: str
    """Category: 'earnings', 'sec_8k', 'sec_10q', 'news', 'reddit', 'github'."""
    
    headline: str
    """Short event description or title."""
    
    body: str
    """Full event text for embedding and retrieval."""
    
    timestamp: datetime
    """When the event occurred or was published."""
    
    source_url: Optional[str] = None
    """Original source link if available."""
    
    sentiment_label: Optional[str] = None
    """Pre-labeled sentiment: 'bullish', 'bearish', 'neutral', or None if unlabeled."""
    
    raw_metadata: dict = field(default_factory=dict)
    """Extra fields: {'earnings_eps_beat': True, 'sec_form_type': '8-K', ...}."""
    
    embedding_vector: Optional[List[float]] = None
    """Dense embedding for vector similarity search (set by Historian)."""


@dataclass
class HistoricalMatch:
    """
    Result of a RAG similarity search against the ChromaDB event corpus.
    
    Returned by Historian when querying for events similar to current
    market conditions. Used by Judge to calibrate predictions.
    """
    
    event: MarketEvent
    """The matched historical event."""
    
    similarity_score: float
    """Cosine similarity [0.0, 1.0]; higher = more relevant."""
    
    days_ago: int
    """How many calendar days ago this event occurred."""
    
    subsequent_return: Optional[float] = None
    """Market return (%) in the N days after this event (from historical label)."""
    
    match_context: Optional[str] = None
    """Free-text reason why this match was returned (e.g., 'similar earnings miss')."""
    
    confidence_weight: float = 1.0
    """Multiplier for this match's influence on final prediction [0.0, 2.0]."""


@dataclass
class ConfidenceReport:
    """
    Quantified confidence metrics for a single-ticker prediction.
    
    Synthesized by Historian (RAG confidence + event recency + corpus coverage)
    and refined by Judge (consensus across baselines, anomaly flags).
    
    Used for post-mortem calibration and uncertainty quantification.
    """
    
    ticker: str
    """Stock ticker."""
    
    prediction_date: datetime
    """Date the prediction was made."""
    
    base_confidence: float
    """Initial confidence from RAG corpus analysis [0.0, 1.0]."""
    
    rag_matches_count: int
    """Number of historical matches found for this ticker."""
    
    avg_match_similarity: float
    """Mean similarity score of top K matches [0.0, 1.0]."""
    
    recency_penalty: float
    """Discount if recent matches are sparse or old [0.0, 1.0]; 1.0 = no penalty."""
    
    linguistic_confidence: Optional[float] = None
    """Certainty score from Linguist sentiment analysis [0.0, 1.0]."""
    
    baseline_consensus: Optional[float] = None
    """Fraction of Judge baselines agreeing on direction [0.0, 1.0]."""
    
    anomaly_flags: List[str] = field(default_factory=list)
    """Red flags: ['unusual_volume', 'earnings_within_7d', 'low_liquidity', ...]."""
    
    final_confidence: Optional[float] = None
    """Post-mortem: calibrated confidence after actual outcome observed."""
    
    notes: Optional[str] = None
    """Free-text reasoning or caveats."""


@dataclass
class EmbeddingBatch:
    """
    Container for bulk embedding operations in Historian.
    
    Used when syncing multiple MarketEvent objects with the vector DB.
    """
    
    events: List[MarketEvent]
    """Events to embed."""
    
    model: str = "google-generativeai-embedding"
    """Embedding model identifier."""
    
    batch_id: str = ""
    """Correlation ID for logging."""
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    """When the batch was created."""


@dataclass
class HistorianConfig:
    """
    Configuration for the Historian RAG layer.
    
    Loaded from YAML and passed to ChromaDB initialization, embedding
    parameters, and similarity thresholds.
    """
    
    chromadb_path: str
    """Local path to ChromaDB persistent storage."""
    
    collection_name: str = "market_events"
    """Primary collection name in ChromaDB."""
    
    embedding_model: str = "models/embedding-001"
    """Gemini embedding model identifier."""
    
    similarity_threshold: float = 0.3
    """Minimum cosine similarity to return a match [0.0, 1.0]."""
    
    top_k_matches: int = 5
    """Number of historical matches to retrieve per query."""
    
    recency_window_days: int = 365
    """How far back to weight matches more heavily."""
    
    min_corpus_size: int = 50
    """Warn if fewer than this many events indexed for a ticker."""
    
    vector_db_batch_size: int = 100
    """Chunk size for bulk inserts into ChromaDB."""
