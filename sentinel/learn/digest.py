"""Builds the compact markdown digest that a Claude research session reads."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sentinel.config import Settings
from sentinel.data import indicators as ind
from sentinel.data.market import MarketData
from sentinel.learn.attribution import digest_md
from sentinel.learn.tournament import leaderboard
from sentinel.store.db import Database, to_iso
from sentinel.strategies.base import REGISTRY
from sentinel.util.clock import UTC

MEMORY_MAX_CHARS = 6000


def _fmt(x, nd=2, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.0f}%" if pct else f"{x:+.{nd}f}"


def build_digest(db: Database, settings: Settings, md: MarketData, status: dict[str, Any], *, kind: str, now: datetime | None = None, last_run_at: str | None = None) -> str:
    now = now or datetime.now(UTC)
    acct = status.get("account", {})
    pop = status.get("population", {})
    budget = status.get("budget", {})
    lines: list[str] = []
    lines.append(f"# Sentinel lab digest - {now:%Y-%m-%d %H:%M} UTC ({kind} session)")
    lines.append(f"Mode **{status.get('mode')}** | equity ${acct.get('equity', 0):,.0f} | today {acct.get('day_pnl', 0):+,.0f} ({acct.get('day_pnl_pct', 0):+.2f}%) | "
                 f"week {acct.get('week_pnl', 0):+,.0f} | since start {acct.get('total_pnl', 0):+,.0f} ({acct.get('total_pnl_pct', 0):+.2f}%) | open lots {acct.get('open_positions', 0)}")
    lines.append(f"Population: {pop.get('active', 0)} active, {pop.get('incubating', 0)} incubating, {pop.get('probation', 0)} probation, {pop.get('retired', 0)} retired (cap {settings.population.max_variants}). "
                 f"Claude budget this week: ${budget.get('spent_usd', 0):.2f} of ${budget.get('weekly_cap_usd', 0):.2f}.")

    # ---- what changed since the last session -------------------------------------------
    since = last_run_at or to_iso(now - timedelta(days=1))
    n_new = db.trade_count(since=since)
    lines.append(f"\n## Since the last {kind} session ({since[:16]} UTC)")
    lines.append(f"- {n_new} trades closed")
    transitions = [e for e in db.events(limit=200, since=since) if e["kind"] == "tournament" and "->" in e["message"]]
    if transitions:
        lines.append("- Lifecycle: " + "; ".join(e["message"] for e in transitions[:10]))
    props = db.query("SELECT p.id,p.type,p.target,p.status,p.decision_reason,p.created_variant_id,p.rationale FROM proposals p WHERE p.decided_at>=? ORDER BY p.id DESC LIMIT 12", (since,))
    if props:
        lines.append("- Outcomes of earlier proposals:")
        for p in props:
            lines.append(f"  - #{p['id']} {p['type']} on {p['target'] or '-'}: **{p['status']}** - {p['decision_reason'] or ''}" + (f" -> created {p['created_variant_id']}" if p["created_variant_id"] else ""))
    inc = [v for v in db.variants(status=["incubating"])]
    if inc:
        lines.append("- Incubating now: " + ", ".join(f"{v['id']} ({v['origin']})" for v in inc[:12]))

    # ---- tournament table ---------------------------------------------------------------
    lines.append("\n## Tournament (all-time on closed trades; last30 = last 30 trades)")
    lines.append("| variant | status | alloc | n | expR (95% CI) | win | PF | last30 expR | maxDD% | P&L | origin |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = leaderboard(db, float(acct.get("equity") or settings.broker.starting_equity))
    shown = 0
    for r in rows:
        if r["status"] == "retired" and shown >= 20:
            continue
        m = r["metrics"]
        ci = m.get("expectancy_ci") or [None, None]
        l30 = m.get("last_30") or {}
        lines.append(f"| {r['id']} | {r['status']} | {r['allocation']:.2f} | {m.get('n', 0)} | {_fmt(m.get('expectancy_r'))} ({_fmt(ci[0])}..{_fmt(ci[1])}) | "
                     f"{_fmt(m.get('win_rate'), pct=True)} | {m.get('profit_factor') if m.get('profit_factor') is not None else 'n/a'} | {_fmt(l30.get('expectancy_r'))} (n={l30.get('n', 0)}) | "
                     f"{m.get('max_dd_pct', 0):.1f} | {m.get('pnl', 0):+.0f} | {r['origin']} |")
        shown += 1
        if shown >= 28:
            break
    control = next((r for r in rows if r["family"] == "random_entry"), None)
    if control:
        cm = control["metrics"]
        lines.append(f"\nControl (random entries, same exits): n={cm.get('n', 0)}, expectancy {_fmt(cm.get('expectancy_r'))}R, win {_fmt(cm.get('win_rate'), pct=True)}. "
                     "A family is only interesting if it beats this with a non-overlapping confidence interval.")

    # ---- attribution ---------------------------------------------------------------------
    lines.append("\n## Factor attribution")
    lines.append(digest_md(db.latest_attribution() or {}))

    # ---- notable trades ---------------------------------------------------------------
    recent = db.trades(limit=400, since=to_iso(now - timedelta(days=7)), with_features=False)
    if recent:
        best = sorted(recent, key=lambda t: -float(t["pnl_r"]))[:3]
        worst = sorted(recent, key=lambda t: float(t["pnl_r"]))[:3]
        lines.append("\n## Notable trades (last 7 days)")
        for t in best:
            lines.append(f"- WIN {t['variant_id']} {t['side']} {t['symbol']} {float(t['pnl_r']):+.2f}R ({t['exit_reason']}, {float(t['hold_minutes']):.0f} min) - {t.get('reason')}")
        for t in worst:
            lines.append(f"- LOSS {t['variant_id']} {t['side']} {t['symbol']} {float(t['pnl_r']):+.2f}R ({t['exit_reason']}, {float(t['hold_minutes']):.0f} min) - {t.get('reason')}")

    # ---- regime -------------------------------------------------------------------------
    lines.append("\n## Market regime")
    for sym in (settings.universe.benchmark_equity, settings.universe.benchmark_crypto):
        d = md.frame(sym, "1d", end_ts=int(now.timestamp()), n=260)
        if len(d) >= 30:
            c = d["c"]
            r20 = float(c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else 0.0
            r5 = float(c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) > 6 else 0.0
            vol = ind.realized_vol(c, 20).iloc[-1]
            adx = ind.adx(d, 14).iloc[-1]
            s50 = ind.sma(c, 50).iloc[-1]
            lines.append(f"- {sym}: 5d {r5:+.1f}%, 20d {r20:+.1f}%, realised vol {float(vol) * 100:.0f}% ann., ADX14 {float(adx):.0f}, vs SMA50 {float(c.iloc[-1] / s50 - 1) * 100:+.1f}%")

    # ---- families ---------------------------------------------------------------------
    lines.append("\n## Families available")
    for name, cls in sorted(REGISTRY.items()):
        space = ", ".join(f"{k}:{v[1]}-{v[2]}" if v[0] != "choice" else f"{k}:{'/'.join(map(str, v[1]))}" for k, v in cls.PARAM_SPACE.items())
        lines.append(f"- `{name}` ({'/'.join(cls.markets)}, {cls.timeframe}{', CONTROL' if cls.is_control else ''}): {cls.description} Params: {space or 'none'}")

    # ---- memory -------------------------------------------------------------------------
    mem_path = settings.lab_dir / "memory.md"
    memory = mem_path.read_text(encoding="utf-8") if mem_path.exists() else ""
    lines.append("\n## Lab memory (your notes from earlier sessions)")
    lines.append(memory[-MEMORY_MAX_CHARS:] if memory.strip() else "_(empty - this is the first session)_")
    return "\n".join(lines)
