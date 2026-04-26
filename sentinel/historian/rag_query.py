"""Historian: keyword-overlap RAG against a hand-curated seed of market events.

Stand-in for ChromaDB until the vector pipeline is wired. Keeps interface stable.
"""
from __future__ import annotations

SEED: list[dict] = [
    {"date": "2008-09-15", "event": "Lehman Brothers bankruptcy filing — credit markets seize", "sp500_5d_pct": -22.0, "tags": ["credit", "bank", "collapse", "liquidity"]},
    {"date": "2020-03-12", "event": "COVID circuit breaker — broad equity sell-off, supply chain fears", "sp500_5d_pct": -12.0, "tags": ["pandemic", "supply chain", "panic", "circuit breaker"]},
    {"date": "2022-11-10", "event": "FTX collapse — crypto contagion, risk-off then sharp rebound", "sp500_5d_pct": 5.5, "tags": ["crypto", "fraud", "contagion"]},
    {"date": "2023-03-10", "event": "Silicon Valley Bank failure — regional bank stress", "sp500_5d_pct": -2.7, "tags": ["bank", "deposit", "regional", "liquidity"]},
    {"date": "2023-05-04", "event": "Apple earnings — cautious supply chain commentary, guidance hedged", "sp500_5d_pct": 1.8, "tags": ["earnings", "apple", "supply chain", "hedged", "guidance"]},
    {"date": "2024-08-05", "event": "Yen carry unwind — VIX spike, tech sell-off", "sp500_5d_pct": -3.0, "tags": ["volatility", "tech", "carry trade"]},
]

STOP = {"the", "a", "an", "and", "or", "but", "of", "in", "on", "to", "for", "with", "is", "are", "was", "were", "be", "been"}


def query(text: str, k: int = 3) -> list[dict]:
    """Return top-k seed events with the highest keyword overlap with `text`."""
    words = {w.lower().strip(".,;:!?\"'()") for w in text.split()} - STOP
    scored: list[tuple[int, dict]] = []
    for ev in SEED:
        bag = set(ev["event"].lower().split()) | set(ev["tags"])
        bag = {w.strip(".,;:!?\"'()") for w in bag} - STOP
        scored.append((len(words & bag), ev))
    scored.sort(key=lambda x: -x[0])
    return [{**ev, "match_score": s} for s, ev in scored[:k] if s > 0]
