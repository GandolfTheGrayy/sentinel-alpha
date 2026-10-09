"""Historian: Gemini text embeddings (768-dim) with on-disk seed cache."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL = "models/text-embedding-004"


def _genai():
    """Lazy import + configure google-generativeai."""
    import google.generativeai as genai
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=key)
    return genai


def embed(texts: list[str]) -> np.ndarray:
    """Return (n, 768) float32 array of embeddings for the given texts."""
    g = _genai()
    out: list[list[float]] = []
    for t in texts:
        r = g.embed_content(model=MODEL, content=t)
        out.append(r["embedding"])
    return np.asarray(out, dtype=np.float32)


def cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a 1-D query vector and rows of `matrix`."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q


def load_or_build(cache_path: Path, texts: list[str]) -> np.ndarray:
    """Load cached seed embeddings or build + cache them on first call."""
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mat = embed(texts)
    np.save(cache_path, mat)
    return mat
