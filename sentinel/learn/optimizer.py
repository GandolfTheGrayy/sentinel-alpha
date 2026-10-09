"""Parameter search (no LLM): mutate the best variants of each family and keep what survives
a walk-forward backtest against both the incumbent and the random-entry control.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from sentinel.config import Settings
from sentinel.data.market import MarketData
from sentinel.engine.population import create_variant
from sentinel.learn.backtest import walk_forward
from sentinel.store.db import Database, now_iso, to_iso
from sentinel.strategies.base import REGISTRY
from sentinel.util.clock import UTC


def control_baseline(settings: Settings, md: MarketData, markets: list[str], days: int, splits: int, end: datetime, cache: dict[str, dict]) -> dict[str, Any]:
    key = "/".join(sorted(markets))
    if key not in cache:
        cache[key] = walk_forward(settings, md, "random_entry", {}, days=days, splits=splits, end=end, markets=markets)
    return cache[key]


def gate(candidate: dict[str, Any], incumbent: dict[str, Any] | None, control: dict[str, Any] | None, *, min_trades: int = 15) -> tuple[bool, str]:
    """Decide whether a backtested candidate deserves incubation."""
    n = candidate.get("n") or 0
    e = candidate.get("expectancy_r")
    if n < min_trades or e is None:
        return False, f"only {n} trades in the backtest window (need {min_trades})"
    if e <= 0:
        return False, f"expectancy {e:+.2f}R is not positive"
    if candidate.get("segments_positive", 0) < max(1, candidate.get("splits", 1) - 1):
        return False, f"only {candidate.get('segments_positive', 0)}/{candidate.get('splits', 1)} walk-forward segments positive"
    if control and control.get("n", 0) >= 10 and control.get("expectancy_r") is not None and e <= float(control["expectancy_r"]) + 0.05:
        return False, f"expectancy {e:+.2f}R does not beat the random-entry control ({control['expectancy_r']:+.2f}R) by 0.05R"
    if incumbent and incumbent.get("n", 0) >= 10 and incumbent.get("expectancy_r") is not None and e < float(incumbent["expectancy_r"]) + 0.02:
        return False, f"expectancy {e:+.2f}R does not improve on the incumbent ({incumbent['expectancy_r']:+.2f}R) by 0.02R"
    return True, f"OOS expectancy {e:+.2f}R over {n} trades" + (f" vs incumbent {incumbent['expectancy_r']:+.2f}R" if incumbent and incumbent.get("expectancy_r") is not None else "") + (f", control {control['expectancy_r']:+.2f}R" if control and control.get("expectancy_r") is not None else "")


def slim(result: dict[str, Any]) -> dict[str, Any]:
    keep = ("n", "expectancy_r", "expectancy_ci", "win_rate", "profit_factor", "max_dd_pct", "pnl", "sharpe", "segments", "segments_positive", "splits", "days")
    return {k: result.get(k) for k in keep if k in result}


def run_optimizer(db: Database, settings: Settings, md: MarketData, *, now: datetime | None = None, max_new: int = 2, log=print) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    days, splits = settings.learn.backtest_days, settings.learn.walk_forward_splits
    rng = np.random.default_rng(int(now.timestamp()) // 86400)
    live = [v for v in db.variants(include_retired=False) if not v["is_control"]]
    by_family: dict[str, list[dict]] = {}
    for v in live:
        by_family.setdefault(v["family"], []).append(v)
    control_cache: dict[str, dict] = {}
    created: list[dict[str, Any]] = []
    trials_per_family = max(2, settings.learn.optimizer_trials // max(1, len(by_family)))
    for family, variants in by_family.items():
        if family not in REGISTRY or len(created) >= max_new:
            continue
        cls = REGISTRY[family]
        if not cls.PARAM_SPACE:
            continue
        # parent = the variant with the best all-time expectancy (ties -> most trades)
        def score(v: dict) -> tuple:
            row = db.one("SELECT AVG(pnl_r) AS e, COUNT(*) AS n FROM trades WHERE variant_id=? AND is_backtest=0", (v["id"],)) or {}
            return (float(row.get("e") or 0.0), int(row.get("n") or 0))
        parent = max(variants, key=score)
        markets = parent.get("markets") or cls.markets
        incumbent = walk_forward(settings, md, family, parent["params"], days=days, splits=splits, end=now, markets=markets)
        control = control_baseline(settings, md, markets, days, splits, now, control_cache)
        best: tuple[float, dict, dict] | None = None
        seen = set()
        for _ in range(trials_per_family):
            params = cls.mutate(parent["params"], rng, scale=0.3) if rng.random() < 0.7 else cls.random_params(rng)
            key = repr(sorted(params.items()))
            if key in seen or key == repr(sorted(parent["params"].items())):
                continue
            seen.add(key)
            try:
                res = walk_forward(settings, md, family, params, days=days, splits=splits, end=now, markets=markets)
            except Exception as exc:  # noqa: BLE001
                log(f"optimizer trial failed for {family}: {exc}")
                continue
            ok, _ = gate(res, slim(incumbent), slim(control))
            if ok and (best is None or (res["expectancy_r"] or -9) > best[0]):
                best = (float(res["expectancy_r"]), params, res)
        if best is None:
            db.add_event("tournament", f"optimizer: no {family} candidate beat {parent['id']} ({incumbent.get('expectancy_r')}R) and the control", "info", ts=to_iso(now))
            continue
        e, params, res = best
        ok, reason = gate(res, slim(incumbent), slim(control))
        pid = db.insert_proposal(None, "new_variant", parent["id"], {"family": family, "params": params, "markets": markets, "origin": "optimizer"},
                                 f"optimizer mutation of {parent['id']}")
        child = create_variant(db, settings, family, params, origin="optimizer", parent_id=parent["id"], markets=markets, status="incubating",
                               notes=f"optimizer child of {parent['id']}: {reason}")
        db.update_proposal(pid, status="accepted", decision_reason=reason, backtest={**slim(res), "incumbent_expectancy_r": incumbent.get("expectancy_r"), "control_expectancy_r": control.get("expectancy_r")},
                           created_variant_id=child["id"], decided_at=now_iso())
        db.add_event("tournament", f"optimizer created {child['id']} from {parent['id']}: {reason}", "info", {"variant_id": child["id"]}, ts=to_iso(now))
        created.append(child)
    return created
