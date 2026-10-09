"""HTTP API for the dashboard (contract in docs/API.md)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentinel.config import ROOT, Settings
from sentinel.engine.runner import Engine
from sentinel.learn.backtest import run_backtest
from sentinel.learn.jobs import JobRunner, snapshot_md
from sentinel.learn.tournament import leaderboard, variant_metrics
from sentinel.store.db import Database, from_iso, to_iso
from sentinel.util.clock import UTC

UI_DIST = ROOT / "ui" / "dist"
RANGES = {"1d": timedelta(days=1), "1w": timedelta(days=7), "1m": timedelta(days=30), "all": None}


class StatusBody(BaseModel):
    status: str


class DecisionBody(BaseModel):
    decision: str


class CalibrateBody(BaseModel):
    observed_weekly_pct: float
    window_hours: float | None = None   # default: the last 7 days


class ResearchBody(BaseModel):
    kind: str


class BacktestBody(BaseModel):
    variant_id: str | None = None
    family: str | None = None
    params: dict[str, Any] | None = None
    markets: list[str] | None = None
    days: int = 45


def _downsample(points: list[dict[str, Any]], max_points: int = 700) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    out = [points[int(i * step)] for i in range(max_points)]
    if out[-1] is not points[-1]:
        out.append(points[-1])
    return out


def create_app(engine: Engine, db: Database, settings: Settings, jobs: JobRunner) -> FastAPI:
    app = FastAPI(title="Sentinel v2", version="2.0.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    subscribers: set[asyncio.Queue] = set()

    def fanout(typ: str, payload: dict) -> None:
        for q in list(subscribers):
            try:
                q.put_nowait((typ, payload))
            except asyncio.QueueFull:
                pass

    engine.listeners.append(fanout)

    def equity_now() -> float:
        try:
            return float(engine.broker.account()["equity"])
        except Exception:  # noqa: BLE001
            return settings.broker.starting_equity

    # ---- status / equity / positions ------------------------------------------------------
    @app.get("/api/status")
    def status() -> dict[str, Any]:
        s = engine.status()
        s["jobs"] = jobs.status()
        return s

    @app.get("/api/equity")
    def equity(range: str = "1w", variant: str | None = None) -> list[dict[str, Any]]:
        now = engine.now()
        delta = RANGES.get(range, RANGES["1w"])
        since = to_iso(now - delta) if delta else None
        if variant:
            v = db.variant(variant)
            if not v:
                raise HTTPException(404, "variant not found")
            trades = db.trades(limit=20000, variant_id=variant, since=since, with_features=False)
            trades.sort(key=lambda t: (t["exit_ts"], t["id"]))
            base = settings.broker.starting_equity * float(v.get("allocation") or 0.02)
            cum = 0.0
            pts = []
            for t in trades:
                cum += float(t["pnl"])
                pts.append({"ts": t["exit_ts"], "equity": round(base + cum, 2), "cash": None, "benchmark": None})
            return _downsample(pts)
        rows = db.equity_curve(since=since)
        return _downsample([{"ts": r["ts"], "equity": r["equity"], "cash": r["cash"], "benchmark": r["benchmark"]} for r in rows])

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        prices = engine.trader.state.last_prices
        out = []
        for lot in engine.ledger.open_lots():
            px = prices.get(lot["symbol"]) or engine.md.last_price(lot["symbol"]) or float(lot["entry_price"])
            sign = 1.0 if lot["side"] == "long" else -1.0
            upnl = (px - float(lot["entry_price"])) * sign * float(lot["qty"])
            rpu = float(lot["risk_per_unit"]) or 1e-9
            out.append({"lot_id": lot["id"], "variant_id": lot["variant_id"], "family": lot["family"], "symbol": lot["symbol"], "side": lot["side"], "qty": float(lot["qty"]),
                        "entry_price": float(lot["entry_price"]), "entry_ts": lot["entry_ts"], "current_price": px, "unrealized_pnl": round(upnl, 2),
                        "unrealized_r": round((px - float(lot["entry_price"])) * sign / rpu, 3), "stop": lot.get("stop"), "target": lot.get("target"),
                        "max_hold_ts": lot.get("max_hold_ts"), "reason": lot.get("reason")})
        return out

    @app.get("/api/trades")
    def trades(limit: int = 200, variant: str | None = None, symbol: str | None = None, since: str | None = None, family: str | None = None) -> list[dict[str, Any]]:
        return db.trades(limit=min(limit, 5000), variant_id=variant, symbol=symbol, since=since, family=family)

    # ---- variants ---------------------------------------------------------------------
    @app.get("/api/variants")
    def variants() -> list[dict[str, Any]]:
        return leaderboard(db, equity_now())

    @app.get("/api/variants/{vid}")
    def variant_detail(vid: str) -> dict[str, Any]:
        v = db.variant(vid)
        if not v:
            raise HTTPException(404, "variant not found")
        row = next((r for r in leaderboard(db, equity_now()) if r["id"] == vid), None) or {**v, "metrics": {}, "sparkline": [], "open_positions": 0}
        trades = db.trades(limit=100, variant_id=vid)
        curve_src = sorted(db.trades(limit=5000, variant_id=vid, with_features=False), key=lambda t: (t["exit_ts"], t["id"]))
        cum = 0.0
        curve = []
        for t in curve_src:
            cum += float(t["pnl"])
            curve.append({"ts": t["exit_ts"], "pnl_cum": round(cum, 2)})
        lineage = []
        cur = v
        seen = set()
        while cur and cur["id"] not in seen:
            seen.add(cur["id"])
            lineage.append({"id": cur["id"], "origin": cur["origin"], "created_at": cur["created_at"], "status": cur["status"]})
            cur = db.variant(cur["parent_id"]) if cur.get("parent_id") else None
        children = [{"id": c["id"], "origin": c["origin"], "created_at": c["created_at"], "status": c["status"]} for c in db.query("SELECT * FROM variants WHERE parent_id=?", (vid,))]
        return {**row, "equity_curve": _downsample(curve), "trades": trades, "lineage": list(reversed(lineage)), "children": children, "notes": v.get("notes")}

    @app.post("/api/variants/{vid}/status")
    def set_variant_status(vid: str, body: StatusBody) -> dict[str, Any]:
        v = db.variant(vid)
        if not v:
            raise HTTPException(404, "variant not found")
        if body.status not in ("active", "paused", "retired"):
            raise HTTPException(400, "status must be active, paused or retired")
        fields: dict[str, Any] = {"status": body.status, "status_reason": "operator"}
        if body.status == "retired":
            fields["retired_at"] = to_iso(engine.now())
            fields["allocation"] = 0.0
        elif body.status == "active" and float(v.get("allocation") or 0) <= 0:
            fields["allocation"] = settings.population.exploration_floor
        db.update_variant(vid, **fields)
        engine.record_event("tournament", f"{vid}: {v['status']} -> {body.status} (operator)", "info", {"variant_id": vid})
        engine.population_dirty = True
        return db.variant(vid)

    # ---- attribution / research / proposals -------------------------------------------------
    @app.get("/api/attribution")
    def attribution() -> dict[str, Any]:
        rep = db.latest_attribution()
        return rep or {"generated_at": None, "n_trades": 0, "window_days": settings.learn.attribution_window_days, "insufficient": True, "features": [], "logistic": [],
                       "filters": [], "heatmap": None, "by_family": [], "by_regime": [], "by_exit_reason": [], "by_side": [], "by_market": [], "by_symbol": [], "overall": None}

    @app.get("/api/research")
    def research(limit: int = 20) -> list[dict[str, Any]]:
        rows = db.runs(limit=min(limit, 200))
        for r in rows:
            r.pop("digest_md", None)
            r.pop("analysis_md", None)
            r.pop("memory_update_md", None)
        return rows

    @app.get("/api/research/memory")
    def research_memory() -> dict[str, Any]:
        p = settings.lab_dir / "memory.md"
        if not p.exists():
            return {"memory_md": "", "updated_at": None}
        return {"memory_md": p.read_text(encoding="utf-8"), "updated_at": datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat().replace("+00:00", "Z")}

    @app.get("/api/research/{run_id}")
    def research_detail(run_id: int) -> dict[str, Any]:
        r = db.run(run_id)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @app.post("/api/research/run")
    def research_run(body: ResearchBody) -> dict[str, Any]:
        ok, reason = jobs.queue_research(body.kind)
        return {"queued": ok, "reason": None if ok else reason}

    @app.get("/api/proposals")
    def proposals(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return db.proposals(status=status, limit=min(limit, 500))

    @app.post("/api/proposals/{pid}/decision")
    def proposal_decision(pid: int, body: DecisionBody) -> dict[str, Any]:
        p = db.proposal(pid)
        if not p:
            raise HTTPException(404, "proposal not found")
        if body.decision not in ("accept", "reject"):
            raise HTTPException(400, "decision must be accept or reject")
        if p["status"] == "testing":
            raise HTTPException(409, "proposal is being tested")
        jobs.submit(f"decide-{pid}", lambda: jobs.decide_job(pid, body.decision == "accept"))
        return {"ok": True, "queued": True}

    # ---- budget / events / engine ---------------------------------------------------------
    @app.get("/api/budget")
    def budget() -> dict[str, Any]:
        return engine.budget_provider() if engine.budget_provider else jobs.gov.status(engine.now())

    @app.post("/api/budget/calibrate")
    def calibrate(body: CalibrateBody) -> dict[str, Any]:
        res = jobs.gov.calibrate(body.observed_weekly_pct, engine.now(), window_hours=body.window_hours or 168.0)
        engine.record_event("research", f"budget calibrated: {body.observed_weekly_pct}% observed over {body.window_hours or 168:g} h -> allowance ${res.get('allowance_usd', 0)}", "info", res)
        return {**res, "budget": budget()}

    @app.get("/api/events")
    def events(limit: int = 100, since: str | None = None) -> list[dict[str, Any]]:
        return db.events(limit=min(limit, 1000), since=since)

    @app.post("/api/engine/{action}")
    def engine_action(action: str) -> dict[str, Any]:
        if action == "pause":
            engine.pause()
        elif action == "resume":
            engine.resume()
        elif action == "halt":
            engine.halt()
        elif action == "flatten":
            engine.flatten()
        else:
            raise HTTPException(400, "action must be pause, resume, halt or flatten")
        return {"ok": True, "engine": engine.status()["engine"]}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        return settings.sanitized()

    @app.get("/api/bars")
    def bars(symbol: str, timeframe: str = "15m", limit: int = 300) -> list[dict[str, Any]]:
        if symbol not in engine.md.symbols:
            raise HTTPException(404, "symbol not in universe")
        df = engine.md.frame(symbol, timeframe, end_ts=int(engine.now().timestamp()), n=min(limit, 2000), include_partial=True)
        return [{"t": ts.isoformat().replace("+00:00", "Z"), "o": round(float(r.o), 6), "h": round(float(r.h), 6), "l": round(float(r.l), 6), "c": round(float(r.c), 6), "v": round(float(r.v), 2)}
                for ts, r in zip(df.index, df.itertuples(index=False))]

    @app.get("/api/backtests")
    def backtests(limit: int = 20) -> list[dict[str, Any]]:
        return db.backtests(limit=min(limit, 100))

    @app.post("/api/backtests")
    async def run_bt(body: BacktestBody) -> dict[str, Any]:
        if body.variant_id:
            v = db.variant(body.variant_id)
            if not v:
                raise HTTPException(404, "variant not found")
            family, params, markets = v["family"], v["params"], v.get("markets")
        elif body.family:
            family, params, markets = body.family, body.params or {}, body.markets
        else:
            raise HTTPException(400, "variant_id or family required")

        def work() -> dict[str, Any]:
            md = snapshot_md(db, settings, engine.now())
            res = run_backtest(settings, md, family, params, days=body.days, end=engine.now(), markets=markets, variant_id=body.variant_id)
            res["trades"] = res["trades"][-200:]
            rid = db.insert_backtest(family, body.variant_id, res["params"], body.days, {k: v for k, v in res.items() if k not in ("trades", "equity_curve")})
            return {"id": rid, **res}

        try:
            return await asyncio.to_thread(work)
        except KeyError as exc:
            raise HTTPException(400, f"unknown family: {exc}") from exc

    # ---- SSE ------------------------------------------------------------------------------
    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        subscribers.add(q)

        async def gen():
            try:
                yield f"event: tick\ndata: {json.dumps(status(), default=str)}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        typ, payload = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: {typ}\ndata: {json.dumps(payload, default=str)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                subscribers.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- static UI ------------------------------------------------------------------------
    if UI_DIST.exists() and (UI_DIST / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=str(UI_DIST / "assets")), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            candidate = UI_DIST / path
            if path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(UI_DIST / "index.html"))
    else:
        @app.get("/")
        def no_ui() -> JSONResponse:
            return JSONResponse({"message": "UI not built. Run: cd ui && npm install && npm run build. API docs at /api/docs"})

    return app
