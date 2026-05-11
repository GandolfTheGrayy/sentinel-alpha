"""Sentinel agent trader — auto-manage the 'agent' portfolio.

Runs hourly during US market hours. On each tick:
  - Reads recent un-acted-on Claude predictions
  - Opens positions for high-conviction directional calls
  - Closes positions whose underlying prediction resolved (or hit stop-loss)

Conservative defaults — only opens long positions on 'up' direction
predictions with confidence >= MIN_CONF. Never shorts.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

import requests

from sentinel.storage import db

PORTFOLIO_ID = "agent"
POSITION_SIZE_USD = 100.0
MIN_CONF = 60
STOP_LOSS_PCT = -8.0
TAKE_PROFIT_PCT = 12.0
MAX_OPEN_POSITIONS = 8
FINNHUB = "https://finnhub.io/api/v1"


def _finnhub_price(ticker: str) -> float | None:
    """Fetch current price from Finnhub. Returns None on failure."""
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(f"{FINNHUB}/quote", params={"symbol": ticker, "token": key}, timeout=10)
        r.raise_for_status()
        return float(r.json().get("c") or 0) or None
    except Exception:
        return None


def _is_market_hours() -> bool:
    """Roughly check if US equity market is open (weekday 13:30-21:00 UTC)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes <= 21 * 60


def _open_for_prediction(pred: dict) -> bool:
    """Open an agent position for one Claude prediction if eligible. Returns True on action."""
    if pred.get("direction") != "up" or int(pred.get("confidence") or 0) < MIN_CONF:
        return False
    ticker = pred["ticker"]
    existing = db.get_open_positions(PORTFOLIO_ID, ticker)
    if existing:
        return False
    price = _finnhub_price(ticker)
    if not price or price <= 0:
        print(f"  skip {ticker}: no price")
        return False
    portfolio = db.get_portfolio(PORTFOLIO_ID) or {}
    cash = float(portfolio.get("cash") or 0)
    if cash < POSITION_SIZE_USD:
        print(f"  skip {ticker}: insufficient cash (${cash:.2f})")
        return False
    shares = round(POSITION_SIZE_USD / price, 6)
    cost = shares * price
    db.update_portfolio_cash(PORTFOLIO_ID, cash - cost)
    db.open_position(
        PORTFOLIO_ID, ticker, shares, price,
        thesis=f"auto: conf={pred.get('confidence')} mag={pred.get('magnitude_pct')}%",
        prediction_id=pred.get("id"),
    )
    print(f"  OPEN {ticker} {shares} @ ${price:.2f} (cost ${cost:.2f}, conf {pred.get('confidence')})")
    return True


def _close_position(pos: dict, reason: str, price: float | None = None) -> bool:
    """Close a single position, refund cash."""
    ticker = pos["ticker"]
    price = price if price is not None else _finnhub_price(ticker)
    if not price or price <= 0:
        print(f"  cannot close {ticker}: no price")
        return False
    shares = float(pos["shares"])
    proceeds = shares * price
    portfolio = db.get_portfolio(PORTFOLIO_ID) or {}
    cash = float(portfolio.get("cash") or 0)
    db.close_position(pos["id"], price, shares, reason)
    db.update_portfolio_cash(PORTFOLIO_ID, cash + proceeds)
    entry = float(pos["entry_price"])
    pnl = (price - entry) / entry * 100.0
    print(f"  CLOSE {ticker} ({reason}): {pnl:+.2f}% (proceeds ${proceeds:.2f})")
    return True


def manage_positions() -> tuple[int, int]:
    """Sweep open positions, close anything past horizon or hit stop/take levels."""
    closed_count = 0
    opens = db.get_open_positions(PORTFOLIO_ID)
    today_iso = date.today().isoformat()
    for pos in opens:
        ticker = pos["ticker"]
        price = _finnhub_price(ticker)
        if not price:
            continue
        entry = float(pos["entry_price"])
        pnl_pct = (price - entry) / entry * 100.0
        # Stop loss
        if pnl_pct <= STOP_LOSS_PCT:
            if _close_position(pos, "stop_loss", price):
                closed_count += 1
            continue
        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            if _close_position(pos, "take_profit", price):
                closed_count += 1
            continue
        # Underlying prediction resolved?
        pred_id = pos.get("prediction_id")
        if pred_id:
            preds = db.get_recent_predictions(limit=1000)
            pred = next((p for p in preds if p.get("id") == pred_id), None)
            if pred and pred.get("resolved") and (pred.get("resolved_on") or "") <= today_iso:
                if _close_position(pos, "prediction_resolved", price):
                    closed_count += 1
    return len(opens), closed_count


def open_new_positions() -> int:
    """Open positions on high-conviction Claude predictions not yet acted on."""
    opens = db.get_open_positions(PORTFOLIO_ID)
    if len(opens) >= MAX_OPEN_POSITIONS:
        print(f"  skip opens: at cap ({MAX_OPEN_POSITIONS})")
        return 0
    open_pred_ids = {p.get("prediction_id") for p in opens if p.get("prediction_id")}
    since = date.today().isoformat()
    candidates = db.get_recent_predictions(strategy="claude", since_date=since, limit=50)
    opened = 0
    for pred in candidates:
        if len(opens) + opened >= MAX_OPEN_POSITIONS:
            break
        if pred.get("id") in open_pred_ids or pred.get("resolved"):
            continue
        if _open_for_prediction(pred):
            opened += 1
    return opened


def main() -> int:
    """Entry point — manage existing positions, then open new ones."""
    if not all(os.environ.get(k) for k in ("FINNHUB_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY")):
        print("ERROR: missing FINNHUB / SUPABASE keys", file=sys.stderr)
        return 1
    print(f"Agent trader — {datetime.now(timezone.utc).isoformat()}")
    if not _is_market_hours():
        print("  market closed, sweeping positions only")
    open_count, closed_count = manage_positions()
    opened = open_new_positions() if _is_market_hours() else 0
    print(f"Summary: {open_count} open before · {closed_count} closed · {opened} new")
    return 0


if __name__ == "__main__":
    sys.exit(main())
