"""The shared trading step used by both the live engine and the backtester.

One `step(now_ts)` = mark-to-market + generic exits for every open lot, daily risk
checks, then strategy cycles for every variant whose bar (or time trigger) just closed,
turning signals into sized orders.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Callable

import numpy as np

from sentinel.config import Settings
from sentinel.data.features import snapshot
from sentinel.data.market import MarketData, is_crypto
from sentinel.engine.broker import Broker, Fill
from sentinel.engine.exits import Bar, check_exit, update_marks
from sentinel.engine.ledger import Ledger
from sentinel.engine.risk import size_order
from sentinel.store.db import to_iso
from sentinel.strategies.base import Context, IndicatorCache, Signal, Strategy
from sentinel.util.clock import EARLY_CLOSES, NY, TF_SECONDS, UTC, next_open_close, session_at, session_bounds, to_ny

TRADING_STATUSES = ("active", "incubating", "probation")


def passes_filters(filters: list[dict], feats: dict[str, float]) -> bool:
    """Entry filters attached to a variant's params. A missing feature never blocks."""
    for f in filters:
        name = f.get("feature")
        if name not in feats:
            continue
        x = float(feats[name])
        op = f.get("op")
        vals = f.get("values") or []
        val = f.get("value")
        ok = True
        if op == "in":
            ok = any(abs(x - float(v)) < 1e-9 for v in vals)
        elif op == "not_in":
            ok = not any(abs(x - float(v)) < 1e-9 for v in vals)
        elif op == "lt" and val is not None:
            ok = x < float(val)
        elif op == "le" and val is not None:
            ok = x <= float(val)
        elif op == "gt" and val is not None:
            ok = x > float(val)
        elif op == "ge" and val is not None:
            ok = x >= float(val)
        elif op == "between" and len(vals) == 2:
            ok = float(vals[0]) <= x <= float(vals[1])
        if not ok:
            return False
    return True


@dataclass
class TraderState:
    paused: bool = False            # no new entries (manual)
    halted: bool = False            # flattened, nothing happens until resumed
    entries_paused_day: str | None = None
    day_date: str | None = None
    day_start_equity: float = 0.0
    last_equity: float = 0.0
    last_prices: dict[str, float] = field(default_factory=dict)
    errors: list[tuple[int, str]] = field(default_factory=list)


class Trader:
    def __init__(self, settings: Settings, md: MarketData, broker: Broker, ledger: Ledger, variants: list[dict[str, Any]], strategies: dict[str, Strategy],
                 *, backtest: bool = False, sim: bool = False, on_event: Callable[[str, str, str, dict], None] | None = None, sleeve_override: float | None = None):
        self.s = settings
        self.md = md
        self.broker = broker
        self.ledger = ledger
        self.variants = {v["id"]: v for v in variants}
        self.strategies = strategies
        self.backtest = backtest
        # in backtests and in --sim the clock is virtual: every price lookup must be bounded by now_ts
        self.bounded = backtest or sim
        self.cache = IndicatorCache()
        self.state = TraderState()
        self.on_event = on_event or (lambda kind, msg, level, data: None)
        self.sleeve_override = sleeve_override
        self._rng = {vid: np.random.default_rng(zlib.crc32(vid.encode())) for vid in self.variants}
        self._last_bucket: dict[tuple[str, str], int] = {}
        self._fired: dict[tuple[str, str], str] = {}
        self._pending: list[tuple[str, Signal, dict]] = []
        self.ref_symbols = {"equities": settings.universe.equities[:1], "crypto": settings.universe.crypto[:1]}

    # ---- helpers ------------------------------------------------------------------------
    def event(self, kind: str, message: str, level: str = "info", data: dict | None = None) -> None:
        self.on_event(kind, message, level, data or {})

    def equity(self) -> float:
        return float(self.broker.account()["equity"])

    def sleeve_equity(self, variant: dict[str, Any], equity: float) -> float:
        if self.sleeve_override is not None:
            return self.sleeve_override
        return equity * float(variant.get("allocation") or 0.0)

    def tradable_variants(self) -> list[dict[str, Any]]:
        return [v for v in self.variants.values() if v["status"] in TRADING_STATUSES and v["id"] in self.strategies]

    def _minute_bar(self, symbol: str, now_ts: int) -> Bar | None:
        """The last 1-minute bar fully closed by now_ts."""
        b = self.md.store.minute_bar(symbol, now_ts)
        return Bar(*b) if b else None

    def _session_last_minute(self, now_ts: int) -> bool:
        ny = to_ny(datetime.fromtimestamp(now_ts, UTC))
        b = session_bounds(ny.date())
        if not b:
            return False
        close_local = to_ny(b[1])
        last_min = (close_local - timedelta(minutes=1)).time()
        return ny.time() >= last_min

    # ---- main step ----------------------------------------------------------------------
    def step(self, now_ts: int) -> None:
        now = datetime.fromtimestamp(now_ts, UTC)
        session = session_at(now)
        if self.backtest:
            self._fill_pending(now_ts)
        self._refresh_prices(now_ts)
        self._day_rollover(now)
        self._exits(now_ts, session)
        if self.state.halted:
            return
        self._risk_checks(now_ts)
        equity = self.equity()
        for v in self.tradable_variants():
            strat = self.strategies[v["id"]]
            try:
                self._run_variant(v, strat, now_ts, now, session, equity)
            except Exception as exc:  # noqa: BLE001
                self.state.errors.append((now_ts, f"{v['id']}: {exc}"))
                self.event("system", f"variant {v['id']} failed: {exc}", "error")

    # ---- prices / marks -----------------------------------------------------------------
    def _refresh_prices(self, now_ts: int) -> None:
        prices: dict[str, float] = {}
        symbols = set(l["symbol"] for l in self.ledger.lots.values())
        if not self.backtest:
            symbols |= set(self.md.symbols)
        for s in symbols:
            p = self.md.last_price(s, end_ts=now_ts if self.bounded else None)
            if p:
                prices[s] = p
        self.state.last_prices = prices
        if hasattr(self.broker, "set_marks"):
            self.broker.set_marks(prices)

    def _day_rollover(self, now: datetime) -> None:
        d = to_ny(now).date().isoformat()
        if self.state.day_date != d:
            self.state.day_date = d
            self.state.day_start_equity = self.equity()
            self.state.entries_paused_day = None
            self.cache.clear()

    # ---- exits --------------------------------------------------------------------------
    def _exits(self, now_ts: int, session: str) -> None:
        last_min = self._session_last_minute(now_ts)
        for lot in list(self.ledger.lots.values()):
            bar = self._minute_bar(lot["symbol"], now_ts)
            if bar is None or bar.ts < lot.get("entry_ts_epoch", 0):
                continue
            update_marks(lot, bar)
            crypto = is_crypto(lot["symbol"])
            hit = check_exit(lot, bar, now_ts, intrabar=self.backtest, session_last_minute=(last_min and not crypto))
            if hit:
                reason, px = hit
                self._close(lot, now_ts, px, reason)
            elif not self.backtest and now_ts % 300 < 60:
                self.ledger.sync_marks(lot)

    def _close(self, lot: dict[str, Any], now_ts: int, ref_price: float, reason: str) -> dict[str, Any] | None:
        side = "sell" if lot["side"] == "long" else "buy"
        try:
            fill = self.broker.market_order(lot["symbol"], side, float(lot["qty"]), ref_price=ref_price, extended_hours=self.s.risk.extended_hours,
                                            client_id=f"sx-{lot['id']}-{now_ts}"[:48])
        except Exception as exc:  # noqa: BLE001
            self.event("risk", f"close order failed for {lot['symbol']} ({lot['variant_id']}): {exc}", "error")
            fill = None
        if fill is None:
            if self.backtest:
                return None
            # cannot reach the broker: record at reference price so the ledger stays consistent
            fill = Fill(lot["symbol"], side, float(lot["qty"]), float(ref_price), 0.0, datetime.fromtimestamp(now_ts, UTC), "unfilled")
            reason = "kill" if reason == "kill" else reason
        trade = self.ledger.close(lot, now_ts, fill.price, reason, fees=fill.fees, is_backtest=self.backtest)
        self.event("exit", f"{lot['variant_id']} {reason} {lot['symbol']} {lot['side']} {fill.qty:g} @ {fill.price:.4g} -> {trade['pnl']:+.2f} ({trade['pnl_r']:+.2f}R)",
                   "info", {"trade": {k: trade[k] for k in ("id", "variant_id", "symbol", "side", "pnl", "pnl_r", "exit_reason")}})
        return trade

    def flatten_all(self, now_ts: int, reason: str = "kill") -> int:
        n = 0
        for lot in list(self.ledger.lots.values()):
            px = self.state.last_prices.get(lot["symbol"]) or float(lot["entry_price"])
            if self._close(lot, now_ts, px, reason):
                n += 1
        return n

    # ---- risk ---------------------------------------------------------------------------
    def _risk_checks(self, now_ts: int) -> None:
        eq = self.equity()
        self.state.last_equity = eq
        start = self.state.day_start_equity or eq
        if start <= 0:
            return
        dd = (start - eq) / start * 100.0
        if dd >= self.s.risk.daily_loss_halt_pct and not self.state.halted:
            n = self.flatten_all(now_ts, "kill")
            self.state.halted = True
            self.event("risk", f"daily loss {dd:.2f}% >= halt limit; flattened {n} lots and halted", "error", {"dd_pct": dd})
        elif dd >= self.s.risk.daily_loss_pause_pct and self.state.entries_paused_day != self.state.day_date:
            self.state.entries_paused_day = self.state.day_date
            self.event("risk", f"daily loss {dd:.2f}% >= pause limit; no new entries today", "warn", {"dd_pct": dd})

    def entries_allowed(self) -> bool:
        return not (self.state.paused or self.state.halted or self.state.entries_paused_day == self.state.day_date)

    # ---- cycles -------------------------------------------------------------------------
    def _bucket_start(self, market: str, tf: str, now_ts: int) -> int | None:
        for ref in self.ref_symbols.get(market, []):
            b = self.md.store.last_closed_bucket(ref, tf, now_ts)
            if b is not None:
                return b
        return None

    def _run_variant(self, v: dict[str, Any], strat: Strategy, now_ts: int, now: datetime, session: str, equity: float) -> None:
        tf = strat.tf()
        markets = [m for m in strat.market_list() if self.md.symbols_for([m])]
        for market in markets:
            symbols = self.md.symbols_for([market])
            if market == "equities" and strat.regular_session_only and session != "regular":
                continue
            if strat.trigger == "bar":
                b = self._bucket_start(market, tf, now_ts)
                if b is None:
                    continue
                key = (v["id"], market)
                if self._last_bucket.get(key) == b:
                    continue
                if market == "equities" and not self._bucket_complete(market, tf, b, now_ts):
                    continue
                self._last_bucket[key] = b
                self._cycle(v, strat, now_ts, session, symbols, equity, market)
            else:
                ny = to_ny(now)
                for trig in strat.trigger:
                    h, m = (int(x) for x in trig.split(":"))
                    key = (v["id"], f"{market}:{trig}")
                    trig_dt = datetime.combine(ny.date(), time(h, m), tzinfo=NY)
                    if ny < trig_dt or (ny - trig_dt) > timedelta(minutes=20) or self._fired.get(key) == ny.date().isoformat():
                        continue
                    self._fired[key] = ny.date().isoformat()
                    self._cycle(v, strat, now_ts, session, symbols, equity, market)

    def _bucket_complete(self, market: str, tf: str, bucket_start: int, now_ts: int) -> bool:
        """Live mode only: make sure the final minute of the bucket has arrived (data lag guard)."""
        if self.backtest or tf == "1d":
            return True
        ref = self.ref_symbols[market][0]
        base = self.md.store.base(ref, "1m")
        if base.empty:
            return False
        last_min = int(base.index[-1].timestamp())
        bucket_end = bucket_start + TF_SECONDS[tf]
        # the bucket may be a short session-end bucket; accept if the last minute bar is the session's last minute
        return last_min >= bucket_end - 60 or self._session_last_minute(last_min)

    def _cycle(self, v: dict[str, Any], strat: Strategy, now_ts: int, session: str, symbols: list[str], equity: float, market: str) -> None:
        allow_short = bool(strat.shortable and market == "equities" and self.s.risk.allow_shorts_equities)
        ctx = Context(md=self.md, ts=now_ts, tf=strat.tf(), symbols=symbols, open_lots=self.ledger.open_lots(v["id"]), session=session,
                      rng=self._rng[v["id"]], cache=self.cache, variant_id=v["id"], allow_short=allow_short)
        # strategy-managed exits first
        for lot in [l for l in ctx.open_lots if l["symbol"] in symbols]:
            df = ctx.frame(lot["symbol"])
            if df.empty:
                continue
            try:
                reason = strat.manage(ctx, lot, df)
            except Exception as exc:  # noqa: BLE001
                self.event("system", f"{v['id']} manage failed: {exc}", "warn")
                reason = None
            if reason:
                px = self.state.last_prices.get(lot["symbol"]) or float(df["c"].iloc[-1])
                self._close(lot, now_ts, px, reason)
        if not self.entries_allowed():
            return
        ctx.open_lots = self.ledger.open_lots(v["id"])
        signals = strat.on_cycle(ctx)
        for sig in signals:
            self._handle_signal(v, strat, sig, now_ts, equity, market)

    def _handle_signal(self, v: dict[str, Any], strat: Strategy, sig: Signal, now_ts: int, equity: float, market: str) -> None:
        price = self.state.last_prices.get(sig.symbol) or self.md.last_price(sig.symbol, end_ts=now_ts if self.bounded else None)
        if not price:
            return
        err = sig.validate(price)
        if err:
            self.event("signal", f"{v['id']} rejected signal on {sig.symbol}: {err}", "warn")
            return
        if sig.side == "short" and (is_crypto(sig.symbol) or not self.s.risk.allow_shorts_equities or not strat.shortable):
            return
        if not is_crypto(sig.symbol):
            sess = session_at(datetime.fromtimestamp(now_ts, UTC))
            if sess != "regular" and not (self.s.risk.extended_hours and sess in ("pre", "post")):
                return
        held = self.ledger.side_held(sig.symbol)
        if held and held != sig.side:
            self.event("signal", f"{v['id']} {sig.side} {sig.symbol} blocked: another variant is {held}", "info", {"conflict": True})
            return
        my_lots = self.ledger.open_lots(v["id"])
        if any(l["symbol"] == sig.symbol for l in my_lots):
            return
        if len(my_lots) >= self.s.risk.max_open_per_variant:
            return
        sleeve = self.sleeve_equity(v, equity)
        if self.ledger.variant_drawdown_pct(v["id"], sleeve) >= self.s.risk.variant_drawdown_pause_pct and not v.get("is_control"):
            return
        gross = self.ledger.gross_exposure(self.state.last_prices)
        res = size_order(price, sig.stop, sleeve, equity, gross, sig.symbol, self.s.risk)
        if res.qty <= 0:
            self.event("signal", f"{v['id']} {sig.side} {sig.symbol} not sized: {res.reason}", "info")
            return
        feats = snapshot(self.md, sig.symbol, strat.tf(), now_ts,
                         benchmark=self.s.universe.benchmark_crypto if is_crypto(sig.symbol) else self.s.universe.benchmark_equity,
                         extra={"signal_strength": sig.strength, "n_open_positions": float(len(self.ledger.lots)),
                                "sleeve_dd_pct": self.ledger.variant_drawdown_pct(v["id"], sleeve), **sig.extra})
        filters = strat.p.get("filters") or []
        if filters and not passes_filters(filters, feats):
            self.event("signal", f"{v['id']} {sig.side} {sig.symbol} filtered out", "info", {"filtered": True})
            return
        if self.backtest:
            self._pending.append((v["id"], sig, {"qty": res.qty, "features": feats, "ts": now_ts}))
        else:
            self._enter(v["id"], sig, res.qty, feats, now_ts, ref_price=price)

    def _fill_pending(self, now_ts: int) -> None:
        """Backtest only: fill signals queued at the previous step at the open of the bar that just closed."""
        pend, self._pending = self._pending, []
        for vid, sig, info in pend:
            bar = self._minute_bar(sig.symbol, now_ts)
            if bar is None or bar.ts != now_ts - 60:
                if now_ts - info["ts"] <= 300:  # keep waiting a few minutes for a bar (e.g. illiquid minute)
                    self._pending.append((vid, sig, info))
                continue
            if self.ledger.side_held(sig.symbol) not in (None, sig.side):
                continue
            self._enter(vid, sig, info["qty"], info["features"], now_ts - 60, ref_price=bar.o)

    def _enter(self, vid: str, sig: Signal, qty: float, feats: dict, now_ts: int, *, ref_price: float) -> None:
        v = self.variants[vid]
        strat = self.strategies[vid]
        side = "buy" if sig.side == "long" else "sell"
        try:
            fill = self.broker.market_order(sig.symbol, side, qty, ref_price=ref_price, extended_hours=self.s.risk.extended_hours, client_id=f"se-{vid}-{now_ts}"[:48].replace("#", "-"))
        except Exception as exc:  # noqa: BLE001
            self.event("risk", f"entry order failed {vid} {sig.symbol}: {exc}", "error")
            return
        if fill is None:
            self.event("signal", f"{vid} {sig.side} {sig.symbol} not filled", "warn")
            return
        entry = fill.price
        stop = sig.stop
        if (sig.side == "long" and stop >= entry) or (sig.side == "short" and stop <= entry):
            stop = entry * (0.995 if sig.side == "long" else 1.005)
        tf_secs = TF_SECONDS[strat.tf()]
        max_hold_ts = None
        if sig.max_hold_bars:
            max_hold_ts = now_ts + int(sig.max_hold_bars) * tf_secs
        if "overnight_exit_min" in sig.extra:
            o, _ = next_open_close(datetime.fromtimestamp(now_ts, UTC) + timedelta(minutes=1))
            max_hold_ts = int(o.timestamp()) + int(sig.extra["overnight_exit_min"]) * 60
        lot = {
            "variant_id": vid, "family": v["family"], "symbol": sig.symbol, "side": sig.side, "qty": fill.qty,
            "entry_ts": to_iso(datetime.fromtimestamp(now_ts, UTC)), "entry_price": entry, "stop": stop, "target": sig.target,
            "max_hold_ts": to_iso(datetime.fromtimestamp(max_hold_ts, UTC)) if max_hold_ts else None,
            "flat_at_session_end": bool(sig.flat_at_session_end or (strat.intraday_only and not is_crypto(sig.symbol))),
            "trail_atr": sig.trail_atr, "risk_per_unit": abs(entry - stop), "reason": sig.reason, "features": feats,
            "signal": {"stop": sig.stop, "target": sig.target, "max_hold_bars": sig.max_hold_bars, "strength": sig.strength},
            "broker_order_id": fill.order_id, "fees": fill.fees,
        }
        lot = self.ledger.open(lot)
        self.event("fill", f"{vid} {sig.side} {sig.symbol} {fill.qty:g} @ {entry:.4g} stop {stop:.4g}" + (f" target {sig.target:.4g}" if sig.target else "") + f" - {sig.reason}",
                   "info", {"lot_id": lot["id"], "variant_id": vid, "symbol": sig.symbol, "side": sig.side, "qty": fill.qty, "price": entry})
