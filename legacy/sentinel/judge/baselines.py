"""Judge: non-LLM baseline strategies tracked alongside Claude predictions.

If Claude can't beat these, the system isn't generating real signal.
"""
from __future__ import annotations

STRATEGIES = ["always_up", "always_neutral", "momentum"]


def predict(strategy: str, ticker: str, recent_pct: float) -> dict:
    """Return a baseline prediction shaped like Claude's predict() output."""
    if strategy == "always_up":
        return {"direction": "up", "magnitude_pct": 1.0, "confidence": 50, "rationale": "always-up baseline"}
    if strategy == "always_neutral":
        return {"direction": "neutral", "magnitude_pct": 0.0, "confidence": 50, "rationale": "always-neutral baseline"}
    if strategy == "momentum":
        d = "up" if recent_pct > 0.5 else "down" if recent_pct < -0.5 else "neutral"
        return {"direction": d, "magnitude_pct": round(abs(recent_pct) * 0.5, 2), "confidence": 50, "rationale": f"5d momentum follow ({recent_pct:+.1f}%)"}
    raise ValueError(f"unknown baseline strategy: {strategy}")
