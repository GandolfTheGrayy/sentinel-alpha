"""Event-driven minute backtester built on the shared Trader step.

Runs one or more variants over the bars cached in MarketData (no network), filling
entries at the next minute's open and exits intrabar, with the configured slippage and
fees. Returns the same trade records the live engine produces.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from sentinel.config import Settings
from sentinel.data.market import MarketData
from sentinel.engine.broker import SimBroker
from sentinel.engine.ledger import Ledger
from sentinel.engine.trader import Trader
from sentinel.learn.metrics import compute_metrics
from sentinel.strategies.base import Strategy, get_family
from sentinel.util.clock import UTC


def run_backtest(settings: Settings, md: MarketData, family: str, params: dict[str, Any], *, days: int, end: datetime | None = None,
                 markets: list[str] | None = None, variant_id: str | None = None, sleeve: float | None = None, start_equity: float | None = None,
                 collect_curve: bool = True) -> dict[str, Any]:
    """Backtest a single family/params over the last `days` of cached bars ending at `end`."""
    cls = get_family(family)
    strat: Strategy = cls({**params, **({"markets": markets} if markets else {})})
    vid = variant_id or f"{family}#bt"
    variant = {"id": vid, "family": family, "status": "active", "allocation": 1.0, "is_control": cls.is_control, "params": strat.p}
    equity0 = float(start_equity or settings.broker.starting_equity)
    broker = SimBroker(equity0, settings.costs)
    ledger = Ledger(None, persist=False)
    events: list[dict] = []
    trader = Trader(settings, md, broker, ledger, [variant], {vid: strat}, backtest=True,
                    on_event=lambda kind, msg, level, data: events.append({"kind": kind, "message": msg, "level": level}) if level != "info" else None,
                    sleeve_override=sleeve if sleeve is not None else equity0 * 0.10)
    # relax the daily halt for a sleeve-sized backtest: losses are measured per sleeve, not per account
    end = end or datetime.now(UTC)
    start = end - timedelta(days=days)
    symbols = md.symbols_for(strat.market_list())
    minute_ts = _minute_grid(md, symbols, int(start.timestamp()), int(end.timestamp()))
    curve: list[dict[str, Any]] = []
    last_curve_day = None
    for i, ts in enumerate(minute_ts):
        now_ts = int(ts) + 60  # the minute bar [ts, ts+60) has closed
        trader.step(now_ts)
        if collect_curve:
            day = (now_ts // 3600)
            if day != last_curve_day:
                last_curve_day = day
                pnl = sum(t["pnl"] for t in ledger.trades) + ledger.unrealized(trader.state.last_prices)
                curve.append({"ts": datetime.fromtimestamp(now_ts, UTC).isoformat().replace("+00:00", "Z"), "pnl_cum": round(pnl, 2)})
    # flatten whatever is still open at the end so results are comparable
    final_ts = int(minute_ts[-1]) + 60 if len(minute_ts) else int(end.timestamp())
    trader.flatten_all(final_ts, "kill")
    trades = ledger.trades
    m = compute_metrics(trades, sleeve=trader.sleeve_override or equity0)
    return {
        "family": family, "params": strat.p, "variant_id": vid, "days": days, "start": start.isoformat(), "end": end.isoformat(),
        "n": m["n"], "win_rate": m["win_rate"], "expectancy_r": m["expectancy_r"], "expectancy_ci": m["expectancy_ci"], "profit_factor": m["profit_factor"],
        "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"], "pnl": m["pnl"], "avg_hold_minutes": m["avg_hold_minutes"],
        "by_exit_reason": m["by_exit_reason"], "equity_curve": curve, "trades": trades, "warnings": [e["message"] for e in events][:20],
    }


def _minute_grid(md: MarketData, symbols: list[str], start_ts: int, end_ts: int) -> np.ndarray:
    arrays = []
    for s in symbols:
        base = md.store.base(s, "1m")
        if base.empty:
            continue
        a = base.index.values.astype("datetime64[s]").astype("int64")
        arrays.append(a[(a >= start_ts) & (a < end_ts)])
    if not arrays:
        return np.array([], dtype="int64")
    return np.unique(np.concatenate(arrays))


def walk_forward(settings: Settings, md: MarketData, family: str, params: dict[str, Any], *, days: int, splits: int, end: datetime | None = None,
                 markets: list[str] | None = None) -> dict[str, Any]:
    """Split the window into `splits` consecutive segments and report per-segment + pooled stats."""
    end = end or datetime.now(UTC)
    seg_days = max(3, days // max(1, splits))
    segments = []
    all_trades: list[dict] = []
    for i in range(splits):
        seg_end = end - timedelta(days=seg_days * (splits - 1 - i))
        r = run_backtest(settings, md, family, params, days=seg_days, end=seg_end, markets=markets, collect_curve=False)
        segments.append({"end": seg_end.isoformat(), "n": r["n"], "expectancy_r": r["expectancy_r"], "pnl": r["pnl"], "win_rate": r["win_rate"]})
        all_trades += r["trades"]
    m = compute_metrics(all_trades, sleeve=settings.broker.starting_equity * 0.10)
    positive = sum(1 for s in segments if (s["expectancy_r"] or 0) > 0)
    return {"family": family, "params": params, "days": days, "splits": splits, "segments": segments, "segments_positive": positive,
            "n": m["n"], "expectancy_r": m["expectancy_r"], "expectancy_ci": m["expectancy_ci"], "win_rate": m["win_rate"], "profit_factor": m["profit_factor"],
            "sharpe": m["sharpe"], "max_dd_pct": m["max_dd_pct"], "pnl": m["pnl"], "trades": all_trades}
