"""The live engine: a minute loop around Trader with polling, snapshots and scheduled jobs.

Two clocks: real time (Alpaca paper/live, or sim at real speed) and a fast simulated
clock (`--sim --speed N` advances N simulated minutes per real second) so a full week
of synthetic trading can be observed in minutes.
"""
from __future__ import annotations

import asyncio
import time as _time
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable

from sentinel.config import Settings, alpaca_keys
from sentinel.data.market import MarketData
from sentinel.data.synthetic import SyntheticProvider
from sentinel.engine.broker import AlpacaBroker, SimBroker
from sentinel.engine.ledger import Ledger
from sentinel.engine.population import instantiate, load_evolved_families, seed_population
from sentinel.engine.trader import Trader
from sentinel.store.db import Database, from_iso, now_iso, to_iso
from sentinel.util.clock import UTC, next_fire, next_open_close, session_at


class Engine:
    def __init__(self, settings: Settings, db: Database, *, sim: bool = False, sim_speed: float = 30.0, sim_start: datetime | None = None, log: Callable[[str], None] = print):
        self.s = settings
        self.db = db
        self.sim = sim or settings.mode == "sim"
        self.sim_speed = sim_speed
        self.log = log
        self.started_at = _time.time()
        self.tick_count = 0
        self.last_tick: str | None = None
        self.running = False
        self._stop = asyncio.Event()
        self.listeners: list[Callable[[str, dict], None]] = []
        self.jobs: dict[str, Callable[[], Any]] = {}
        self._next_fire: dict[str, datetime] = {}
        self._job_specs: dict[str, str] = {}
        self._interval_jobs: dict[str, tuple[int, float]] = {}
        self._cached_status: dict[str, Any] = {}
        self.budget_provider: Callable[[], dict[str, Any]] | None = None
        self.population_dirty = False
        self.loop: asyncio.AbstractEventLoop | None = None
        self._sim_now: datetime = (sim_start or (datetime.now(UTC) - timedelta(days=7))).replace(second=0, microsecond=0)
        if self.sim and not sim_start:
            saved = db.kv_get("sim_clock")
            if saved:
                self._sim_now = from_iso(saved)  # resume the virtual clock where the last sim run stopped

        # ---- wiring ------------------------------------------------------------------------
        if self.sim:
            self.provider = SyntheticProvider(clock=self.now)
            self.broker = SimBroker(settings.broker.starting_equity, settings.costs, clock=self.now)
        else:
            key, secret = alpaca_keys()
            if not key or not secret:
                raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing (put them in .env) or run with --sim")
            from sentinel.data.alpaca_data import AlpacaProvider

            self.provider = AlpacaProvider(key, secret, feed=settings.data.feed, extended_hours=settings.risk.extended_hours)
            self.broker = AlpacaBroker(key, secret, paper=(settings.broker.mode != "live"))
            self.broker.register_crypto(settings.universe.crypto)
        self.md = MarketData(db, self.provider, settings.universe.equities, settings.universe.crypto, feed=settings.data.feed)
        load_evolved_families(settings, log=self.log)
        seed_population(db, settings, log=self.log)
        self.ledger = Ledger(db, persist=True)
        self.trader: Trader | None = None
        self.reload_population()

    # ---- clock --------------------------------------------------------------------------
    def now(self) -> datetime:
        return self._sim_now if self.sim else datetime.now(UTC)

    # ---- population ---------------------------------------------------------------------
    def reload_population(self) -> None:
        variants = self.db.variants(include_retired=False)
        strategies = instantiate(variants, log=self.log)
        state = self.trader.state if self.trader else None
        self.trader = Trader(self.s, self.md, self.broker, self.ledger, variants, strategies, backtest=False, sim=self.sim, on_event=self._on_event)
        if state is not None:
            self.trader.state = state

    def _on_event(self, kind: str, message: str, level: str, data: dict) -> None:
        ts = to_iso(self.now())
        row = self.db.add_event(kind, message, level, data, ts=ts)
        self.log(f"[{ts}] {level.upper():5} {kind:10} {message}")
        self._emit("event", row)

    def _emit(self, typ: str, payload: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(typ, payload)
            except Exception:  # noqa: BLE001
                pass

    def emit_threadsafe(self, typ: str, payload: dict) -> None:
        """Emit from a worker thread (jobs) onto the engine's loop."""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._emit, typ, payload)
        else:
            self._emit(typ, payload)

    def record_event(self, kind: str, message: str, level: str = "info", data: dict | None = None) -> dict:
        """Thread-safe event helper for jobs: persists, logs and streams the event."""
        row = self.db.add_event(kind, message, level, data or {}, ts=to_iso(self.now()))
        self.log(f"[{row['ts']}] {level.upper():5} {kind:10} {message}")
        self.emit_threadsafe("event", row)
        return row

    # ---- jobs ---------------------------------------------------------------------------
    def schedule(self, name: str, spec: str, fn: Callable[[], Any]) -> None:
        """Run fn at the America/New_York wall-clock spec ('16:35' daily or 'Sun 10:00')."""
        self.jobs[name] = fn
        self._job_specs[name] = spec
        self._next_fire[name] = next_fire(spec, self.now())

    def schedule_every(self, name: str, minutes: int, fn: Callable[[], Any]) -> None:
        self.jobs[name] = fn
        self._interval_jobs[name] = (minutes, self.now().timestamp())

    def next_fire_of(self, name: str) -> datetime | None:
        return self._next_fire.get(name)

    def _run_jobs(self, now: datetime) -> None:
        for name, spec in list(self._job_specs.items()):
            if now >= self._next_fire[name]:
                self._next_fire[name] = next_fire(spec, now)
                self._safe_job(name)
        for name, (minutes, last) in list(self._interval_jobs.items()):
            if now.timestamp() - last >= minutes * 60:
                self._interval_jobs[name] = (minutes, now.timestamp())
                self._safe_job(name)

    def _safe_job(self, name: str) -> None:
        try:
            if name not in self._interval_jobs:
                self.log(f"job {name} starting")
            self.jobs[name]()
        except Exception as exc:  # noqa: BLE001
            self.db.add_event("system", f"job {name} failed: {exc}", "error", {"trace": traceback.format_exc()[-1500:]}, ts=to_iso(self.now()))

    # ---- lifecycle ----------------------------------------------------------------------
    def bootstrap(self) -> None:
        now = self.now()
        self.log(f"mode={self.s.mode} broker={self.broker.name} now={now:%Y-%m-%d %H:%M} UTC; loading bars")
        self.md.load_from_db(self.s.data.backfill_days_minute, self.s.data.backfill_days_daily, now=now)
        self.md.backfill(self.s.data.backfill_days_minute, self.s.data.backfill_days_daily, now=now, log=self.log)
        self.md.last_poll_ts = int(now.timestamp())
        if not self.sim:
            self.reconcile()
        self.db.add_event("system", f"engine started ({self.s.mode}); {len(self.trader.tradable_variants())} tradable variants", "info", ts=to_iso(now))

    def reconcile(self) -> None:
        """Compare the ledger's lots with the broker's net positions and warn about any drift (never auto-fixes)."""
        try:
            broker_pos = self.broker.positions()
        except Exception as exc:  # noqa: BLE001
            self.db.add_event("system", f"reconcile skipped: broker positions unavailable ({exc})", "warn", ts=to_iso(self.now()))
            return
        ledger_pos: dict[str, float] = {}
        for lot in self.ledger.lots.values():
            ledger_pos[lot["symbol"]] = ledger_pos.get(lot["symbol"], 0.0) + float(lot["qty"]) * (1.0 if lot["side"] == "long" else -1.0)
        drift = []
        for sym in set(broker_pos) | set(ledger_pos):
            b, l = broker_pos.get(sym, 0.0), ledger_pos.get(sym, 0.0)
            if abs(b - l) > max(1e-6, abs(b) * 0.001):
                drift.append(f"{sym}: broker {b:g} vs ledger {l:g}")
        if drift:
            self.db.add_event("risk", "position drift between broker and ledger: " + "; ".join(drift[:10]), "warn", {"drift": drift}, ts=to_iso(self.now()))
        else:
            self.db.add_event("system", f"reconciled {len(ledger_pos)} ledger symbols with the broker", "info", ts=to_iso(self.now()))

    async def run(self) -> None:
        self.running = True
        self.loop = asyncio.get_running_loop()
        await asyncio.to_thread(self.bootstrap)  # keep the API responsive while history loads
        try:
            while not self._stop.is_set():
                if self.sim:
                    self._sim_now += timedelta(minutes=1)
                    self.tick(self._sim_now)
                    await asyncio.sleep(max(0.0, 1.0 / max(self.sim_speed, 0.01)))
                else:
                    now = datetime.now(UTC)
                    # wake ~15s after each minute boundary so the previous minute's bar exists
                    target = (now.replace(second=15, microsecond=0) + (timedelta(minutes=1) if now.second >= 15 else timedelta(0)))
                    await asyncio.sleep(max(0.5, (target - now).total_seconds()))
                    self.tick(datetime.now(UTC))
        finally:
            self.running = False

    def stop(self) -> None:
        self._stop.set()

    def tick(self, now: datetime) -> None:
        now_ts = int(now.timestamp()) // 60 * 60
        if self.population_dirty:
            self.population_dirty = False
            self.reload_population()
        try:
            self.md.poll(now=now)
        except Exception as exc:  # noqa: BLE001
            self.db.add_event("data", f"poll failed: {exc}", "warn", ts=to_iso(now))
        if not self.sim and self.tick_count % 1 == 0:
            self.md.refresh_prices([l["symbol"] for l in self.ledger.lots.values()] or self.md.symbols[:5])
        try:
            self.trader.step(now_ts)
        except Exception as exc:  # noqa: BLE001
            self.db.add_event("system", f"step failed: {exc}", "error", {"trace": traceback.format_exc()[-1500:]}, ts=to_iso(now))
        self.tick_count += 1
        self.last_tick = to_iso(now)
        if now_ts % 300 == 0:
            self._snapshot(now)
        self._run_jobs(now)
        self._emit("tick", self.status())

    # ---- status -------------------------------------------------------------------------
    def _snapshot(self, now: datetime) -> None:
        acct = self.broker.account()
        bench = self._benchmark_value(now)
        per_variant = {vid: round(p, 2) for vid, p in self.ledger.variant_pnl.items()}
        self.db.add_equity_snapshot(to_iso(now), round(acct["equity"], 2), round(acct["cash"], 2), bench, per_variant)
        if self.sim:
            self.db.kv_set("sim_clock", to_iso(now))
        self._emit("equity", {"ts": to_iso(now), "equity": round(acct["equity"], 2), "cash": round(acct["cash"], 2), "benchmark": bench})

    def _benchmark_value(self, now: datetime) -> float | None:
        """Buy-and-hold benchmark rebased to the starting equity at the first snapshot."""
        sym = self.s.universe.benchmark_equity or self.s.universe.benchmark_crypto
        px = self.md.last_price(sym, end_ts=int(now.timestamp()) if self.sim else None)
        if not px:
            return None
        base = self.db.kv_get("benchmark_base")
        if not base:
            base = {"price": px, "equity": float(self.broker.account()["equity"])}
            self.db.kv_set("benchmark_base", base)
        return round(base["equity"] * px / base["price"], 2)

    def status(self) -> dict[str, Any]:
        now = self.now()
        acct = self.broker.account()
        st = self.trader.state
        start_eq = self.db.kv_get("starting_equity")
        if not start_eq:
            start_eq = float(acct["equity"])
            self.db.kv_set("starting_equity", start_eq)
        day_start = st.day_start_equity or acct["equity"]
        week_row = self.db.one("SELECT equity FROM equity_snapshots WHERE ts>=? ORDER BY ts", (to_iso(now - timedelta(days=7)),))
        week_start = float(week_row["equity"]) if week_row else float(start_eq)
        prices = st.last_prices
        gross = self.ledger.gross_exposure(prices)
        try:
            o, c = next_open_close(now)
        except Exception:  # noqa: BLE001
            o = c = now
        from sentinel.engine.population import population_counts

        return {
            "mode": self.s.mode,
            "engine": {"running": self.running, "paused": st.paused, "halted": st.halted, "last_tick": self.last_tick, "uptime_s": int(_time.time() - self.started_at),
                       "tick_count": self.tick_count, "errors_1h": len([e for e in st.errors if e[0] > now.timestamp() - 3600]), "sim_time": to_iso(now) if self.sim else None},
            "market": {"equities_open": session_at(now) == "regular", "next_open": to_iso(o), "next_close": to_iso(c), "session": session_at(now), "crypto_open": True},
            "account": {"equity": round(acct["equity"], 2), "cash": round(acct["cash"], 2),
                        "day_pnl": round(acct["equity"] - day_start, 2), "day_pnl_pct": round((acct["equity"] / day_start - 1) * 100, 3) if day_start else 0.0,
                        "week_pnl": round(acct["equity"] - week_start, 2), "week_pnl_pct": round((acct["equity"] / week_start - 1) * 100, 3) if week_start else 0.0,
                        "total_pnl": round(acct["equity"] - start_eq, 2), "total_pnl_pct": round((acct["equity"] / start_eq - 1) * 100, 3) if start_eq else 0.0,
                        "gross_exposure_pct": round(gross / acct["equity"] * 100, 2) if acct["equity"] else 0.0, "open_positions": len(self.ledger.lots),
                        "starting_equity": round(float(start_eq), 2)},
            "population": population_counts(self.db),
            "budget": self.budget_provider() if self.budget_provider else self._cached_status.get("budget", {}),
            "universe": {"equities": self.s.universe.equities, "crypto": self.s.universe.crypto},
        }

    def set_budget_status(self, b: dict[str, Any]) -> None:
        self._cached_status["budget"] = b

    # ---- controls -----------------------------------------------------------------------
    def pause(self) -> None:
        self.trader.state.paused = True
        self.db.add_event("system", "entries paused by operator", "warn", ts=to_iso(self.now()))

    def resume(self) -> None:
        self.trader.state.paused = False
        self.trader.state.halted = False
        self.trader.state.entries_paused_day = None
        self.db.add_event("system", "engine resumed by operator", "info", ts=to_iso(self.now()))

    def halt(self) -> None:
        n = self.trader.flatten_all(int(self.now().timestamp()), "kill")
        self.trader.state.halted = True
        self.db.add_event("system", f"halted by operator; flattened {n} lots", "warn", ts=to_iso(self.now()))

    def flatten(self) -> int:
        n = self.trader.flatten_all(int(self.now().timestamp()), "kill")
        self.db.add_event("system", f"flattened {n} lots by operator", "warn", ts=to_iso(self.now()))
        return n
