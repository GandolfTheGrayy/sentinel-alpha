"""Supabase client wrapper — Sentinel's source of truth for predictions, portfolios, positions, trades."""
from __future__ import annotations

import os
from datetime import datetime, timezone

from supabase import Client, create_client


def client() -> Client:
    """Return an authenticated Supabase client (service role)."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY not set")
    return create_client(url, key)


def _normalize_prediction(p: dict) -> dict:
    """Map a pipeline prediction dict into a Supabase predictions row."""
    return {
        "id": p["id"],
        "made_on": p.get("made"),
        "ticker": p["ticker"],
        "strategy": p.get("strategy", "claude"),
        "horizon_days": int(p.get("horizon_days", 5)),
        "direction": p.get("direction"),
        "magnitude_pct": p.get("magnitude_pct"),
        "confidence": p.get("confidence"),
        "rationale": p.get("rationale"),
        "headline": p.get("headline"),
        "publisher": p.get("publisher"),
        "filing": p.get("filing"),
        "evidence": p.get("evidence"),
        "price_at_prediction": p.get("price_at_prediction"),
        "resolves_on": p.get("resolves_on"),
        "resolved": bool(p.get("resolved", False)),
        "resolved_on": p.get("resolved_on"),
        "actual_pct": p.get("actual_pct"),
        "actual_direction": p.get("actual_direction"),
        "correct_direction": p.get("correct_direction"),
        "magnitude_error": p.get("magnitude_error"),
        "postmortem": p.get("postmortem"),
    }


def upsert_predictions(records: list[dict]) -> None:
    """Insert or update prediction records by id."""
    if not records:
        return
    c = client()
    rows = [_normalize_prediction(r) for r in records]
    c.table("predictions").upsert(rows, on_conflict="id").execute()


def get_unresolved_predictions(strategy: str | None = None) -> list[dict]:
    """Return predictions where resolved=false, optionally filtered by strategy."""
    c = client()
    q = c.table("predictions").select("*").eq("resolved", False)
    if strategy:
        q = q.eq("strategy", strategy)
    return q.execute().data or []


def get_recent_predictions(strategy: str | None = None, since_date: str | None = None, limit: int = 500) -> list[dict]:
    """Return recent predictions, newest first."""
    c = client()
    q = c.table("predictions").select("*").order("made_on", desc=True).limit(limit)
    if strategy:
        q = q.eq("strategy", strategy)
    if since_date:
        q = q.gte("made_on", since_date)
    return q.execute().data or []


def get_portfolio(pid: str) -> dict | None:
    """Fetch a portfolio by id ('agent' or 'human')."""
    c = client()
    r = c.table("portfolios").select("*").eq("id", pid).single().execute()
    return r.data


def update_portfolio_cash(pid: str, new_cash: float) -> None:
    """Set the cash field on a portfolio."""
    c = client()
    c.table("portfolios").update({"cash": new_cash}).eq("id", pid).execute()


def get_open_positions(pid: str, ticker: str | None = None) -> list[dict]:
    """Return open positions for a portfolio (optionally for one ticker)."""
    c = client()
    q = c.table("positions").select("*").eq("portfolio_id", pid).eq("closed", False)
    if ticker:
        q = q.eq("ticker", ticker)
    return q.order("entry_time", desc=False).execute().data or []


def open_position(pid: str, ticker: str, shares: float, price: float, thesis: str | None = None, prediction_id: str | None = None) -> dict:
    """Insert a new open position lot and a corresponding buy trade."""
    c = client()
    row = {
        "portfolio_id": pid,
        "ticker": ticker,
        "shares": shares,
        "entry_price": price,
        "thesis": thesis,
        "prediction_id": prediction_id,
    }
    r = c.table("positions").insert(row).execute()
    c.table("trades").insert({"portfolio_id": pid, "ticker": ticker, "action": "buy", "shares": shares, "price": price, "thesis": thesis, "prediction_id": prediction_id}).execute()
    return (r.data or [row])[0]


def close_position(position_id: int, price: float, shares: float, reason: str) -> None:
    """Mark a position closed and log the matching sell trade."""
    c = client()
    pos = c.table("positions").select("portfolio_id, ticker, prediction_id, thesis").eq("id", position_id).single().execute().data or {}
    c.table("positions").update({
        "closed": True,
        "exit_price": price,
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "exit_reason": reason,
    }).eq("id", position_id).execute()
    c.table("trades").insert({
        "portfolio_id": pos.get("portfolio_id"),
        "ticker": pos.get("ticker"),
        "action": "sell",
        "shares": shares,
        "price": price,
        "thesis": f"auto-close: {reason}",
        "prediction_id": pos.get("prediction_id"),
    }).execute()
