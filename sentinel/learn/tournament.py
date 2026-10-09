"""Variant lifecycle and capital allocation.

Lifecycle: incubating -> active -> probation -> retired (plus paused by the operator or by a
drawdown breach). Transitions are driven by trade counts and expectancy confidence
intervals, never by a single lucky run. Allocation is Thompson sampling over recent
expectancy with an exploration floor and a cap, so every variant keeps producing data.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from sentinel.config import Settings
from sentinel.learn.metrics import compute_metrics, rolling_window
from sentinel.store.db import Database, from_iso, now_iso, to_iso
from sentinel.util.clock import UTC

RECENT_N = 50


def variant_metrics(db: Database, v: dict[str, Any], equity: float) -> dict[str, Any]:
    trades = db.trades(limit=5000, variant_id=v["id"], with_features=False)
    sleeve = max(equity * float(v.get("allocation") or 0.0), equity * 0.02)
    m = compute_metrics(trades, sleeve)
    last30 = compute_metrics(rolling_window(trades, 30), sleeve)
    recent = compute_metrics(rolling_window(trades, RECENT_N), sleeve)
    m["last_30"] = {"n": last30["n"], "win_rate": last30["win_rate"], "expectancy_r": last30["expectancy_r"], "pnl": last30["pnl"]}
    m["recent"] = recent
    m["trades"] = trades
    return m


def evaluate_all(db: Database, settings: Settings, equity: float, persist: bool = True) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ts = now_iso()
    for v in db.variants(include_retired=True):
        m = variant_metrics(db, v, equity)
        out[v["id"]] = m
        if persist and v["status"] != "retired":
            slim = {k: val for k, val in m.items() if k not in ("trades", "recent")}
            db.execute("INSERT OR REPLACE INTO evaluations(ts,variant_id,window,metrics) VALUES(?,?,?,?)", (ts, v["id"], "all", __import__("json").dumps(slim)))
    return out


def run_lifecycle(db: Database, settings: Settings, metrics: dict[str, dict[str, Any]], now: datetime | None = None) -> list[dict[str, Any]]:
    """Apply lifecycle transitions. Returns a list of {variant_id, from, to, reason}."""
    now = now or datetime.now(UTC)
    pop = settings.population
    changes: list[dict[str, Any]] = []

    def transition(v: dict, to: str, reason: str) -> None:
        fields: dict[str, Any] = {"status": to, "status_reason": reason}
        if to == "retired":
            fields["retired_at"] = to_iso(now)
            fields["allocation"] = 0.0
        db.update_variant(v["id"], **fields)
        changes.append({"variant_id": v["id"], "from": v["status"], "to": to, "reason": reason})
        db.add_event("tournament", f"{v['id']}: {v['status']} -> {to} ({reason})", "info", {"variant_id": v["id"], "to": to}, ts=to_iso(now))

    for v in db.variants(include_retired=False):
        if v["is_control"]:
            continue
        m = metrics.get(v["id"]) or {}
        n = int(m.get("n") or 0)
        age_days = (now - from_iso(v["created_at"])).total_seconds() / 86400.0
        exp = m.get("expectancy_r")
        ci = m.get("expectancy_ci") or [0.0, 0.0]
        recent = m.get("recent") or {}
        if v["status"] == "incubating":
            matured = n >= pop.incubation_min_trades or age_days >= pop.incubation_max_days
            if not matured:
                continue
            if n < max(10, pop.incubation_min_trades // 3):
                if age_days >= pop.incubation_max_days * 2:
                    transition(v, "retired", f"only {n} trades after {age_days:.0f} days")
                continue
            if exp is not None and exp > 0 and ci[0] > -0.15:
                transition(v, "active", f"incubation passed: {n} trades, {exp:+.2f}R (CI {ci[0]:+.2f}..{ci[1]:+.2f})")
            elif ci[1] < 0:
                transition(v, "retired", f"incubation failed: {n} trades, {exp:+.2f}R (CI upper {ci[1]:+.2f} < 0)")
            elif age_days >= pop.incubation_max_days * 2:
                transition(v, "retired", f"inconclusive after {age_days:.0f} days: {exp:+.2f}R")
        elif v["status"] == "active":
            rn = int(recent.get("n") or 0)
            rexp = recent.get("expectancy_r")
            rci = recent.get("expectancy_ci") or [0.0, 0.0]
            if rn >= 30 and rexp is not None and rexp < 0 and rci[1] < 0.05:
                transition(v, "probation", f"last {rn} trades {rexp:+.2f}R (CI upper {rci[1]:+.2f})")
        elif v["status"] == "probation":
            since = from_iso(v["updated_at"])
            rn = int(recent.get("n") or 0)
            rexp = recent.get("expectancy_r")
            if rn >= 20 and rexp is not None and rexp > 0:
                transition(v, "active", f"recovered: last {rn} trades {rexp:+.2f}R")
            elif (now - since).total_seconds() / 86400.0 >= 14:
                transition(v, "retired", f"no recovery on probation ({rexp if rexp is not None else 'n/a'}R over {rn} trades)")
        elif v["status"] == "paused" and (v.get("status_reason") or "").startswith("drawdown"):
            since = from_iso(v["updated_at"])
            if (now - since).total_seconds() / 86400.0 >= 7:
                transition(v, "probation", "drawdown pause expired")

    # population cap: retire the weakest non-control variants
    live = [v for v in db.variants(include_retired=False) if not v["is_control"]]
    if len(live) > pop.max_variants:
        def score(v: dict) -> float:
            m = metrics.get(v["id"]) or {}
            return float(m.get("expectancy_r") or 0.0) * min(1.0, (m.get("n") or 0) / 30.0)
        live.sort(key=score)
        for v in live[: len(live) - pop.max_variants]:
            transition(v, "retired", "population cap")
    return changes


def allocate(db: Database, settings: Settings, metrics: dict[str, dict[str, Any]], seed: int | None = None, now: datetime | None = None) -> dict[str, float]:
    """Thompson-sampled allocation weights for every non-retired variant; persisted on the variants table."""
    now = now or datetime.now(UTC)
    pop = settings.population
    rng = np.random.default_rng(seed if seed is not None else int(now.timestamp()) // 86400)
    variants = db.variants(include_retired=False)
    weights: dict[str, float] = {}
    controls = [v for v in variants if v["is_control"]]
    fixed = {v["id"]: pop.control_weight for v in controls}
    incubating = [v for v in variants if not v["is_control"] and v["status"] in ("incubating", "probation")]
    for v in incubating:
        fixed[v["id"]] = pop.exploration_floor
    paused = [v for v in variants if v["status"] == "paused"]
    for v in paused:
        fixed[v["id"]] = 0.0
    pool = [v for v in variants if not v["is_control"] and v["status"] == "active"]
    remaining = max(0.0, 1.0 - sum(fixed.values()))
    if pool and remaining > 0:
        samples = {}
        for v in pool:
            m = metrics.get(v["id"]) or {}
            recent = m.get("recent") or {}
            n = int(recent.get("n") or 0)
            if n >= 5:
                trades = m.get("trades") or []
                r = np.array([float(t["pnl_r"]) for t in rolling_window(trades, RECENT_N)]) if trades else np.array([])
                mu = float(r.mean()) if len(r) else 0.0
                sd = float(r.std(ddof=1)) if len(r) > 1 else 1.0
                sample = rng.normal(mu, max(sd, 0.2) / np.sqrt(max(n, 1)))
            else:
                sample = rng.normal(0.0, 0.25)
            samples[v["id"]] = max(sample, 0.0) + 0.02
        total = sum(samples.values())
        raw = {vid: s / total * remaining for vid, s in samples.items()}
        # enforce floor and cap, then renormalise within the pool
        floor = min(pop.exploration_floor, remaining / len(pool))
        cap = pop.allocation_cap
        for _ in range(5):
            clipped = {vid: min(max(w, floor), cap) for vid, w in raw.items()}
            s = sum(clipped.values())
            if s <= 0:
                break
            raw = {vid: w * remaining / s for vid, w in clipped.items()}
        # final hard clip: a little capital may stay unallocated rather than breach the cap
        weights.update({vid: round(min(w, cap), 4) for vid, w in raw.items()})
    weights.update(fixed)
    ts = to_iso(now)
    for vid, w in weights.items():
        db.update_variant(vid, allocation=w)
        db.execute("INSERT OR REPLACE INTO allocations(ts,variant_id,weight) VALUES(?,?,?)", (ts, vid, w))
    return weights


def run_tournament(db: Database, settings: Settings, equity: float, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    metrics = evaluate_all(db, settings, equity)
    changes = run_lifecycle(db, settings, metrics, now=now)
    weights = allocate(db, settings, metrics, now=now)
    summary = {"changes": changes, "weights": weights, "n_variants": len(weights)}
    db.add_event("tournament", f"tournament: {len(changes)} transitions, {len(weights)} allocations", "info", {"changes": changes}, ts=to_iso(now))
    return summary


def leaderboard(db: Database, equity: float) -> list[dict[str, Any]]:
    out = []
    open_counts = {r["variant_id"]: int(r["n"]) for r in db.query("SELECT variant_id, COUNT(*) AS n FROM lots WHERE status='open' GROUP BY variant_id")}
    for v in db.variants(include_retired=True):
        m = variant_metrics(db, v, equity)
        trades = m.pop("trades")
        m.pop("recent", None)
        spark: list[float] = []
        cum = 0.0
        for t in sorted(trades, key=lambda t: (t["exit_ts"], t.get("id") or 0))[-60:]:
            cum += float(t["pnl"])
            spark.append(round(cum, 2))
        out.append({**{k: v[k] for k in ("id", "family", "name", "status", "status_reason", "origin", "parent_id", "created_at", "allocation", "params", "markets", "timeframe", "is_control", "notes")},
                    "metrics": m, "sparkline": spark, "open_positions": open_counts.get(v["id"], 0)})
    order = {"active": 0, "probation": 1, "incubating": 2, "paused": 3, "retired": 4}
    out.sort(key=lambda r: (order.get(r["status"], 9), -(r["metrics"].get("expectancy_r") or -9)))
    return out
