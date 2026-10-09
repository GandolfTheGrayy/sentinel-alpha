"""Sentinel command line.

  python -m sentinel run [--sim] [--speed 30] [--sim-start 2026-09-01] [--port 8787]
  python -m sentinel backtest (--family F | --variant ID) [--params JSON] [--markets equities,crypto] [--days 45] [--walk-forward 3] [--json]
  python -m sentinel research --kind analyst|strategist [--dry-run]
  python -m sentinel tournament | attribution | optimize | seed | doctor | ui-build
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sentinel.config import ROOT, alpaca_keys, load_settings
from sentinel.util.clock import UTC


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    import uvicorn

    from sentinel.engine.runner import Engine
    from sentinel.learn.jobs import JobRunner
    from sentinel.server.app import create_app
    from sentinel.store.db import Database

    settings = load_settings(args.config, sim=args.sim)
    db = Database(settings.db_path)
    sim_start = datetime.fromisoformat(args.sim_start).replace(tzinfo=UTC) if args.sim_start else None
    engine = Engine(settings, db, sim=settings.sim, sim_speed=args.speed, sim_start=sim_start, log=_log)
    jobs = JobRunner(engine, db, settings)
    jobs.wire_schedule()
    app = create_app(engine, db, settings, jobs)
    port = args.port or settings.server.port
    config = uvicorn.Config(app, host=settings.server.host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)

    async def main() -> None:
        _log(f"Sentinel v2 - mode {settings.mode} - dashboard http://{settings.server.host}:{port}")
        engine_task = asyncio.create_task(engine.run())
        server_task = asyncio.create_task(server.serve())
        done, pending = await asyncio.wait({engine_task, server_task}, return_when=asyncio.FIRST_COMPLETED)
        engine.stop()
        server.should_exit = True
        for t in pending:
            try:
                await asyncio.wait_for(t, timeout=10)
            except Exception:  # noqa: BLE001
                t.cancel()
        for t in done:
            exc = t.exception() if not t.cancelled() else None
            if exc:
                raise exc

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _log("stopped")
    return 0


def _clock_for(settings, db) -> datetime:
    """Real time normally; in --sim the database's own clock (last equity snapshot) so offline jobs line up with the engine."""
    if settings.sim:
        from sentinel.store.db import from_iso

        row = db.one("SELECT ts FROM equity_snapshots ORDER BY ts DESC")
        if row:
            return from_iso(row["ts"])
    return datetime.now(UTC)


def _md_from_db(settings, now):
    from sentinel.learn.jobs import snapshot_md
    from sentinel.store.db import Database

    db = Database(settings.db_path)
    now = _clock_for(settings, db) if now is None else now
    md = snapshot_md(db, settings, now)
    if settings.sim and not any(not md.store.base(s, "1m").empty for s in md.symbols):
        from sentinel.data.synthetic import SyntheticProvider

        md.provider = SyntheticProvider(clock=lambda: now)
        md.backfill(settings.data.backfill_days_minute, settings.data.backfill_days_daily, now=now, log=lambda m: None)
    return db, md, now


def cmd_backtest(args: argparse.Namespace) -> int:
    import sentinel.strategies  # noqa: F401 - registers families
    from sentinel.engine.population import load_evolved_families
    from sentinel.learn.backtest import run_backtest, walk_forward

    settings = load_settings(args.config, sim=args.sim)
    load_evolved_families(settings, log=lambda m: None)
    now = None if not args.end else datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    db, md, now = _md_from_db(settings, now)
    if args.variant:
        v = db.variant(args.variant)
        if not v:
            _log(f"variant {args.variant} not found")
            return 2
        family, params, markets = v["family"], v["params"], v.get("markets")
    else:
        family = args.family
        params = json.loads(args.params) if args.params else {}
        markets = args.markets.split(",") if args.markets else None
    if not family:
        _log("--family or --variant required")
        return 2
    if args.walk_forward and args.walk_forward > 1:
        res = walk_forward(settings, md, family, params, days=args.days, splits=args.walk_forward, end=now, markets=markets)
    else:
        res = run_backtest(settings, md, family, params, days=args.days, end=now, markets=markets, variant_id=args.variant)
    out = {k: v for k, v in res.items() if k not in ("trades", "equity_curve")}
    out["exit_reasons"] = res.get("by_exit_reason")
    out.pop("by_exit_reason", None)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
    else:
        _log(f"{family} {params} over {args.days}d: n={res['n']} expectancy={res['expectancy_r']}R win={res['win_rate']} PF={res['profit_factor']} maxDD={res['max_dd_pct']}% pnl={res['pnl']}")
        if "segments" in res:
            for s in res["segments"]:
                _log(f"  segment to {s['end'][:10]}: n={s['n']} expectancy={s['expectancy_r']} pnl={s['pnl']}")
    return 0


def _offline_engine_status(db, settings):
    from sentinel.engine.population import population_counts
    from sentinel.learn.budget import BudgetGovernor

    row = db.one("SELECT equity, cash FROM equity_snapshots ORDER BY ts DESC")
    eq = float(row["equity"]) if row else settings.broker.starting_equity
    start = float(db.kv_get("starting_equity") or settings.broker.starting_equity)
    return {"mode": settings.mode, "account": {"equity": eq, "cash": float(row["cash"]) if row else eq, "day_pnl": 0.0, "day_pnl_pct": 0.0, "week_pnl": 0.0,
            "total_pnl": eq - start, "total_pnl_pct": (eq / start - 1) * 100 if start else 0.0, "open_positions": len(db.open_lots())},
            "population": population_counts(db), "budget": BudgetGovernor(db, settings).status()}


def cmd_research(args: argparse.Namespace) -> int:
    import sentinel.strategies  # noqa: F401
    from sentinel.engine.population import load_evolved_families, seed_population
    from sentinel.learn.claude_lab import run_session

    settings = load_settings(args.config, sim=args.sim)
    load_evolved_families(settings, log=_log)
    db, md, now = _md_from_db(settings, None)
    seed_population(db, settings, log=_log)
    status = _offline_engine_status(db, settings)
    run_id = run_session(db, settings, md, status, args.kind, manual=True, now=now, log=_log, dry_run=args.dry_run)
    run = db.run(run_id) if run_id else None
    if run:
        _log(f"run #{run_id}: {run['status']} cost=${run['cost_usd']:.2f} proposals={len(run['proposals'])}")
        if run.get("error"):
            _log(f"  {run['error']}")
        for p in run["proposals"]:
            _log(f"  #{p['id']} {p['type']} {p['target'] or ''}: {p['status']} - {p['decision_reason'] or ''}")
        if args.dry_run:
            _log(f"  prompt written under {settings.lab_dir / 'runs'}")
    return 0


def cmd_jobs(args: argparse.Namespace) -> int:
    import sentinel.strategies  # noqa: F401
    from sentinel.engine.population import load_evolved_families, seed_population
    from sentinel.learn.attribution import build_report, digest_md
    from sentinel.learn.optimizer import run_optimizer
    from sentinel.learn.tournament import run_tournament
    from sentinel.store.db import to_iso

    settings = load_settings(args.config, sim=args.sim)
    load_evolved_families(settings, log=_log)
    db, md, now = _md_from_db(settings, None)
    seed_population(db, settings, log=_log)
    status = _offline_engine_status(db, settings)
    if args.command == "tournament":
        res = run_tournament(db, settings, status["account"]["equity"], now=now)
        _log(json.dumps({"changes": res["changes"], "weights": res["weights"]}, indent=2))
    elif args.command == "attribution":
        trades = db.trades(limit=20000, since=to_iso(now - timedelta(days=settings.learn.attribution_window_days)))
        rep = build_report(trades, window_days=settings.learn.attribution_window_days, min_trades=settings.learn.min_trades_for_attribution, now=now)
        db.save_attribution(rep, digest_md(rep))
        _log(digest_md(rep))
    elif args.command == "optimize":
        created = run_optimizer(db, settings, md, now=now, log=_log)
        _log(f"created {[c['id'] for c in created]}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from sentinel.learn.budget import BudgetGovernor
    from sentinel.store.db import Database

    settings = load_settings(args.config, sim=args.sim)
    db = Database(settings.db_path)
    gov = BudgetGovernor(db, settings)
    res = gov.calibrate(args.observed_pct, window_hours=args.hours)
    if not res.get("ok"):
        _log(f"not calibrated: {res.get('reason')}")
        return 1
    st = gov.status()
    _log(f"Sentinel spent ${res['spent_usd_at_calibration']:.2f} over {res['runs']} runs in the last {res['window_hours']:g} h = {args.observed_pct}% of the weekly allowance")
    _log(f"estimated weekly allowance ${res['allowance_usd']:.2f} -> weekly cap ${st['weekly_cap_usd']:.2f} ({settings.claude.weekly_share:.0%}); spent this week ${st['spent_usd']:.2f}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    import sentinel.strategies  # noqa: F401
    from sentinel.engine.population import seed_population
    from sentinel.store.db import Database

    settings = load_settings(args.config, sim=args.sim)
    db = Database(settings.db_path)
    n = seed_population(db, settings, log=_log)
    _log(f"{n} variants created; {len(db.variants())} total")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = load_settings(args.config, sim=args.sim)
    ok = True
    _log(f"python     {sys.version.split()[0]} at {sys.executable}")
    _log(f"config     {'config.yaml' if (ROOT / 'config.yaml').exists() else 'defaults (copy config.example.yaml to config.yaml)'}; mode={settings.mode}")
    key, secret = alpaca_keys()
    if key and secret:
        try:
            from sentinel.engine.broker import AlpacaBroker

            b = AlpacaBroker(key, secret, paper=(settings.broker.mode != "live"))
            a = b.account()
            c = b.clock()
            _log(f"alpaca     OK ({'paper' if b.paper else 'LIVE'}): equity ${a['equity']:,.2f}, cash ${a['cash']:,.2f}, market {'open' if c['is_open'] else 'closed'}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            _log(f"alpaca     FAILED: {exc}")
    else:
        _log("alpaca     no keys in .env (ALPACA_API_KEY / ALPACA_SECRET_KEY) - only --sim will work")
        if not settings.sim:
            ok = False
    from sentinel.learn.claude_lab import claude_available, claude_logged_in

    avail, msg = claude_available(settings)
    _log(f"claude     {'OK ' + msg if avail else 'MISSING: ' + msg}")
    if avail:
        logged, amsg = claude_logged_in(settings)
        _log(f"           auth: {'OK ' + amsg if logged else 'NOT LOGGED IN - ' + amsg}")
        if not logged:
            ok = False
    _log(f"claude     budget cap ${settings.claude.weekly_cap_usd:.2f}/week ({settings.claude.weekly_share:.0%} of est. ${settings.claude.weekly_allowance_usd:.0f} for plan {settings.claude.plan})")
    ui = ROOT / "ui" / "dist" / "index.html"
    _log(f"ui         {'built' if ui.exists() else 'not built (cd ui && npm install && npm run build)'}")
    from sentinel.store.db import Database

    db = Database(settings.db_path)
    nb = db.one("SELECT COUNT(*) AS n FROM bars")["n"]
    nv = len(db.variants())
    nt = db.trade_count()
    _log(f"database   {settings.db_path.name}: {nb} bars, {nv} variants, {nt} trades")
    _log("result     " + ("ready" if ok else "fix the items above"))
    return 0 if ok else 1


def cmd_ui_build(args: argparse.Namespace) -> int:
    ui = ROOT / "ui"
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not (ui / "node_modules").exists():
        subprocess.run([npm, "install"], cwd=str(ui), check=True)
    subprocess.run([npm, "run", "build"], cwd=str(ui), check=True)
    _log("ui built into ui/dist")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sentinel", description="Sentinel v2 - self-improving paper-trading lab")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--sim", action="store_true", help="synthetic market, no broker keys needed")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run the engine, learning jobs and dashboard")
    r.add_argument("--speed", type=float, default=30.0, help="sim: simulated minutes per real second")
    r.add_argument("--sim-start", default=None, help="sim: UTC start, e.g. 2026-09-01T13:30")
    r.add_argument("--port", type=int, default=None)
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("backtest", help="offline backtest on cached bars")
    b.add_argument("--family")
    b.add_argument("--variant")
    b.add_argument("--params", help="JSON object")
    b.add_argument("--markets", help="comma separated: equities,crypto")
    b.add_argument("--days", type=int, default=45)
    b.add_argument("--walk-forward", type=int, default=0)
    b.add_argument("--end", default=None, help="UTC end timestamp (default now)")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_backtest)
    rs = sub.add_parser("research", help="run one Claude research session now")
    rs.add_argument("--kind", choices=["analyst", "strategist"], default="analyst")
    rs.add_argument("--dry-run", action="store_true", help="build the prompt but do not call Claude")
    rs.set_defaults(fn=cmd_research)
    for name, help_ in (("tournament", "evaluate, apply lifecycle, reallocate"), ("attribution", "rebuild the factor report"), ("optimize", "run the parameter search")):
        j = sub.add_parser(name, help=help_)
        j.set_defaults(fn=cmd_jobs)
    s = sub.add_parser("seed", help="create the seed population")
    s.set_defaults(fn=cmd_seed)
    c = sub.add_parser("calibrate", help="rescale the plan-allowance estimate from an observed /usage change")
    c.add_argument("--observed-pct", type=float, required=True, help="percentage points of the weekly limit that Sentinel used in the window")
    c.add_argument("--hours", type=float, default=168.0, help="length of the observation window in hours (default 168 = one week)")
    c.set_defaults(fn=cmd_calibrate)
    d = sub.add_parser("doctor", help="check keys, Claude CLI, UI build and database")
    d.set_defaults(fn=cmd_doctor)
    u = sub.add_parser("ui-build", help="npm install + build the dashboard")
    u.set_defaults(fn=cmd_ui_build)
    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
