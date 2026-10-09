"""Historian: embedding-based RAG over seed market events.

Falls back to keyword overlap if Gemini embeddings are unavailable
(missing key, network failure, etc.) so the pipeline is never blocked
on the Historian.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sentinel.historian.embeddings import cosine, embed, load_or_build

SEED: list[dict] = [
    {"date": "2008-09-15", "event": "Lehman Brothers bankruptcy filing — credit markets seize", "sp500_5d_pct": -22.0, "tags": ["credit", "bank", "collapse", "liquidity"]},
    {"date": "2020-03-12", "event": "COVID circuit breaker — broad equity sell-off, supply chain fears", "sp500_5d_pct": -12.0, "tags": ["pandemic", "supply chain", "panic", "circuit breaker"]},
    {"date": "2022-11-10", "event": "FTX collapse — crypto contagion, risk-off then sharp rebound", "sp500_5d_pct": 5.5, "tags": ["crypto", "fraud", "contagion"]},
    {"date": "2023-03-10", "event": "Silicon Valley Bank failure — regional bank stress", "sp500_5d_pct": -2.7, "tags": ["bank", "deposit", "regional", "liquidity"]},
    {"date": "2023-05-04", "event": "Apple earnings — cautious supply chain commentary, guidance hedged", "sp500_5d_pct": 1.8, "tags": ["earnings", "apple", "supply chain", "hedged", "guidance"]},
    {"date": "2024-08-05", "event": "Yen carry unwind — VIX spike, tech sell-off", "sp500_5d_pct": -3.0, "tags": ["volatility", "tech", "carry trade"]},
]

CACHE = Path("docs/_cache/seed_embeddings.npy")
STOP = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "to", "for", "with", "is", "are", "was", "were", "be", "been"}


def _seed_text(ev: dict) -> str:
    """Concatenate event + tags into one embedding-friendly string."""
    return f"{ev['event']} | tags: {', '.join(ev.get('tags', []))}"


def _keyword_score(text: str, ev: dict) -> int:
    """Fallback overlap score (count of shared non-stop words)."""
    words = {w.lower().strip(".,;:!?\"'()") for w in text.split()} - STOP
    bag = {w.lower().strip(".,;:!?\"'()") for w in (ev["event"] + " " + " ".join(ev.get("tags", []))).split()} - STOP
    return len(words & bag)


def query(text: str, k: int = 3) -> list[dict]:
    """Return top-k seed events by embedding similarity, with keyword fallback."""
    try:
        seed_texts = [_seed_text(e) for e in SEED]
        seed_mat = load_or_build(CACHE, seed_texts)
        q_vec = embed([text])[0]
        sims = cosine(q_vec, seed_mat)
        order = np.argsort(-sims)[:k]
        out = []
        for i in order:
            score = float(sims[int(i)])
            if score < 0.35:
                continue
            out.append({**SEED[int(i)], "match_score": round(score, 3), "method": "embedding"})
        return out
    except Exception:
        scored = [(_keyword_score(text, ev), ev) for ev in SEED]
        scored.sort(key=lambda x: -x[0])
        return [{**ev, "match_score": s, "method": "keyword"} for s, ev in scored[:k] if s > 0]
