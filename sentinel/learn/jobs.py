"""Background learning jobs, serialised on one worker thread so they never block trading.

Every job that needs bars builds its own MarketData snapshot from the database, so the
engine's in-memory store is never read from another thread.
"""
from __future__ import annotations

import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Callable

from sentinel.config import Settings
from sentinel.data.market import MarketData
from sentinel.engine.runner import Engine
from sentinel.learn.attribution import build_report, digest_md
from sentinel.learn.budget import BudgetGovernor
from sentinel.learn.claude_lab import decide_proposal, run_session
from sentinel.learn.optimizer import run_optimizer
from sentinel.learn.tournament import evaluate_all, run_tournament
from sentinel.store.db import Database, now_iso, to_iso
from sentinel.util.clock import UTC


def snapshot_md(db: Database, settings: Settings, now: datetime | None = None) -> MarketData:
    md = MarketData(db, provider=None, equities=settings.universe.equities, crypto=settings.universe.crypto)  # type: ignore[arg-type]
    md.load_from_db(settings.data.backfill_days_minute, settings.data.backfill_days_daily, now=now or datetime.now(UTC))
    return md


class JobRunner:
    def __init__(self, engine: Engine, db: Database, settings: Settings):
        self.engine = engine
        self.db = db
        self.s = settings
        self.gov = BudgetGovernor(db, settings)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-job")
        self._lock = threading.Lock()
        self.current: str | None = None
        self.queued: list[str] = []
        self.last_result: dict[str, Any] = {}

    # ---- plumbing ---------------------------------------------------------------------
    def submit(self, name: str, fn: Callable[[], Any]) -> Future:
        with self._lock:
            self.queued.append(name)

        def wrapped() -> Any:
            with self._lock:
                self.current = name
                if name in self.queued:
                    self.queued.remove(name)
            try:
                res = fn()
                self.last_result[name] = {"ok": True, "at": now_iso()}
                return res
            except Exception as exc:  # noqa: BLE001
                self.last_result[name] = {"ok": False, "at": now_iso(), "error": str(exc)}
                self.engine.record_event("system", f"job {name} failed: {exc}", "error", {"trace": traceback.format_exc()[-1500:]})
                raise
            finally:
                with self._lock:
                    self.current = None

        return self._pool.submit(wrapped)

    def wire_schedule(self) -> None:
        sc = self.s.schedules
        e = self.engine
        e.schedule_every("metrics", sc.metrics_every_minutes, lambda: self.submit("metrics", self.metrics_job))
        e.schedule("tournament", sc.tournament, lambda: self.submit("tournament", self.tournament_job))
        e.schedule("attribution", sc.attribution, lambda: self.submit("attribution", self.attribution_job))
        e.schedule("optimizer", sc.optimizer, lambda: self.submit("optimizer", self.optimizer_job))
        e.schedule("analyst", sc.analyst, lambda: self.submit("analyst", lambda: self.research_job("analyst")))
        e.schedule("strategist", sc.strategist, lambda: self.submit("strategist", lambda: self.research_job("strategist")))
        e.budget_provider = lambda: self.gov.status(e.now(), next_runs={"analyst": to_iso(e.next_fire_of("analyst")) if e.next_fire_of("analyst") else None,
                                                                         "strategist": to_iso(e.next_fire_of("strategist")) if e.next_fire_of("strategist") else None})

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"current": self.current, "queued": list(self.queued), "last": self.last_result}

    # ---- jobs ---------------------------------------------------------------------------
    def _equity(self) -> float:
        try:
            return float(self.engine.broker.account()["equity"])
        except Exception:  # noqa: BLE001
            return self.s.broker.starting_equity

    def metrics_job(self) -> dict[str, Any]:
        m = evaluate_all(self.db, self.s, self._equity())
        return {"variants": len(m)}

    def tournament_job(self) -> dict[str, Any]:
        res = run_tournament(self.db, self.s, self._equity(), now=self.engine.now())
        self.engine.population_dirty = True
        for c in res["changes"]:
            self.engine.emit_threadsafe("event", {"ts": now_iso(), "level": "info", "kind": "tournament", "message": f"{c['variant_id']}: {c['from']} -> {c['to']} ({c['reason']})", "data": c})
        return res

    def attribution_job(self) -> dict[str, Any]:
        now = self.engine.now()
        since = to_iso(now - timedelta(days=self.s.learn.attribution_window_days))
        trades = self.db.trades(limit=20000, since=since)
        report = build_report(trades, window_days=self.s.learn.attribution_window_days, min_trades=self.s.learn.min_trades_for_attribution, now=now)
        rid = self.db.save_attribution(report, digest_md(report))
        self.engine.record_event("research", f"attribution report #{rid}: {report['n_trades']} trades, {len(report.get('filters') or [])} filter candidates", "info", {"report_id": rid})
        return {"report_id": rid, "n": report["n_trades"]}

    def optimizer_job(self) -> dict[str, Any]:
        now = self.engine.now()
        md = snapshot_md(self.db, self.s, now)
        created = run_optimizer(self.db, self.s, md, now=now, log=self.engine.log)
        if created:
            self.engine.population_dirty = True
        return {"created": [c["id"] for c in created]}

    def research_job(self, kind: str, manual: bool = False) -> dict[str, Any]:
        now = self.engine.now()
        md = snapshot_md(self.db, self.s, now)
        status = self.engine.status()
        run_id = run_session(self.db, self.s, md, status, kind, manual=manual, now=now, log=self.engine.log)
        self.engine.population_dirty = True
        run = self.db.run(run_id) if run_id else None
        self.engine.emit_threadsafe("event", {"ts": now_iso(), "level": "info", "kind": "research", "message": f"{kind} session finished: {run['status'] if run else 'n/a'}", "data": {"run_id": run_id}})
        return {"run_id": run_id, "status": run["status"] if run else None}

    def decide_job(self, pid: int, accept: bool) -> dict[str, Any]:
        if not accept:
            self.db.update_proposal(pid, status="rejected", decision_reason="rejected by operator", decided_at=now_iso())
            return {"ok": True}
        md = snapshot_md(self.db, self.s, self.engine.now())
        res = decide_proposal(self.db, self.s, md, pid, now=self.engine.now(), force_accept=True, log=self.engine.log)
        self.engine.population_dirty = True
        return {"ok": True, "proposal": res}

    # ---- entry points used by the API -----------------------------------------------------
    def queue_research(self, kind: str) -> tuple[bool, str]:
        if kind not in ("analyst", "strategist"):
            return False, "kind must be analyst or strategist"
        cfg = self.s.claude.analyst if kind == "analyst" else self.s.claude.strategist
        ok, reason = self.gov.can_start(cfg.max_budget_usd, self.engine.now())
        if not ok:
            return False, reason
        with self._lock:
            if self.current == kind or kind in self.queued:
                return False, f"{kind} session already running or queued"
        self.submit(kind, lambda: self.research_job(kind, manual=True))
        return True, "queued"
