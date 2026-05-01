"""Judge: resolve a due prediction by comparing forecast to actual move."""
from __future__ import annotations

import yfinance as yf


def resolve(pred: dict) -> dict:
    """Mutate `pred` in place with actual outcome fields, return it."""
    if pred.get("resolved"):
        return pred
    p0 = pred.get("price_at_prediction")
    if not p0:
        pred["resolved"] = True
        pred["error"] = "no baseline price"
        return pred
    try:
        h = yf.Ticker(pred["ticker"]).history(period="10d", auto_adjust=True)
        if h.empty:
            return pred
        p1 = float(h["Close"].iloc[-1])
        actual_pct = (p1 - p0) / p0 * 100.0
        actual_dir = "up" if actual_pct > 0.5 else "down" if actual_pct < -0.5 else "neutral"
        pred["actual_pct"] = round(actual_pct, 2)
        pred["actual_direction"] = actual_dir
        pred["correct_direction"] = pred.get("direction", "neutral") == actual_dir
        pred["magnitude_error"] = round(abs(abs(actual_pct) - float(pred.get("magnitude_pct", 0.0))), 2)
        pred["price_at_resolution"] = round(p1, 2)
        pred["resolved"] = True
    except Exception as exc:
        pred["error"] = str(exc)[:120]
    return pred
