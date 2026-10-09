import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import sentinel.strategies  # noqa: F401 - register families
from sentinel.config import Settings
from sentinel.engine.population import create_variant, seed_population
from sentinel.learn import claude_lab
from sentinel.learn.attribution import build_report, digest_md
from sentinel.learn.backtest import run_backtest, walk_forward
from sentinel.learn.budget import BudgetGovernor
from sentinel.learn.metrics import compute_metrics, wilson_ci
from sentinel.learn.tournament import allocate, evaluate_all, leaderboard, run_lifecycle
from sentinel.store.db import Database, to_iso
from sentinel.strategies.base import REGISTRY, check_evolved_source
from tests.conftest import NOW


def _fake_trades(db, vid, family, n, mean_r, seed=1, start=None):
    rng = np.random.default_rng(seed)
    start = start or (NOW - timedelta(days=20))
    for i in range(n):
        r = float(rng.normal(mean_r, 1.0))
        t0 = start + timedelta(hours=i * 3)
        lot = {"id": None, "variant_id": vid, "family": family, "symbol": "AAPL" if i % 2 else "BTC/USD", "side": "long", "qty": 10, "entry_ts": to_iso(t0),
               "entry_price": 100.0, "risk_per_unit": 1.0, "hwm": 100 + max(r, 0), "lwm": 100 + min(r, 0), "reason": "x",
               "features": {"hour_et": float(9 + i % 7), "dow": float(i % 5), "rsi_14": float(rng.uniform(10, 90)), "atr_pct": float(rng.uniform(0.2, 2)),
                            "regime_trend": float(i % 2), "regime_vol": float(i % 3), "is_crypto": 0.0 if i % 2 else 1.0, "session": 2.0}}
        db.close_lot(lot, to_iso(t0 + timedelta(minutes=45)), 100.0 + r, "target" if r > 0 else "stop")


def test_metrics_and_ci():
    lo, hi = wilson_ci(5, 10)
    assert lo < 0.5 < hi
    trades = [{"pnl": 10, "pnl_r": 1.0, "hold_minutes": 30, "exit_reason": "target", "entry_ts": "2026-01-01T00:00:00Z", "exit_ts": "2026-01-01T01:00:00Z"}] * 10
    m = compute_metrics(trades, sleeve=1000)
    assert m["n"] == 10 and m["win_rate"] == 1.0 and m["expectancy_r"] == 1.0 and m["max_dd_pct"] == 0.0


def test_backtest_runs_and_is_deterministic(settings, md):
    a = run_backtest(settings, md, "trend_ema", {"tf": "15m"}, days=12, end=NOW, markets=["equities", "crypto"])
    b = run_backtest(settings, md, "trend_ema", {"tf": "15m"}, days=12, end=NOW, markets=["equities", "crypto"])
    assert a["n"] > 0 and a["n"] == b["n"] and a["pnl"] == b["pnl"]
    assert all(abs(t["pnl_r"]) < 25 for t in a["trades"])
    assert set(t["exit_reason"] for t in a["trades"]) <= {"stop", "target", "time", "signal", "session_end", "kill", "rebalance"}
    wf = walk_forward(settings, md, "random_entry", {}, days=12, splits=2, end=NOW, markets=["equities"])
    assert wf["splits"] == 2 and len(wf["segments"]) == 2


@pytest.mark.parametrize("family", sorted(REGISTRY))
def test_every_family_backtests(settings, md, family):
    cls = REGISTRY[family]
    r = run_backtest(settings, md, family, {}, days=10, end=NOW, markets=cls.markets)
    assert r["family"] == family and r["n"] >= 0


def test_population_tournament_and_allocation(settings):
    db = Database(settings.data_dir / "tourney.db")
    seed_population(db, settings)
    variants = db.variants()
    assert len(variants) == 16 and sum(v["allocation"] for v in variants) <= 1.0001
    good = next(v for v in variants if v["family"] == "trend_ema")
    bad = next(v for v in variants if v["family"] == "squeeze")
    _fake_trades(db, good["id"], "trend_ema", 60, 0.4, seed=2)
    _fake_trades(db, bad["id"], "squeeze", 60, -0.6, seed=3)
    metrics = evaluate_all(db, settings, 100_000)
    assert metrics[good["id"]]["n"] == 60 and metrics[good["id"]]["expectancy_r"] > 0
    changes = run_lifecycle(db, settings, metrics, now=NOW)
    assert any(c["variant_id"] == bad["id"] and c["to"] == "probation" for c in changes)
    child = create_variant(db, settings, "trend_ema", {"fast": 7}, origin="claude", parent_id=good["id"], status="incubating")
    _fake_trades(db, child["id"], "trend_ema", 35, 0.5, seed=4)
    metrics = evaluate_all(db, settings, 100_000)
    changes = run_lifecycle(db, settings, metrics, now=NOW)
    assert any(c["variant_id"] == child["id"] and c["to"] == "active" for c in changes)
    w = allocate(db, settings, metrics, seed=1, now=NOW)
    assert abs(sum(w.values()) - 1.0) < 0.02
    assert w[good["id"]] >= w[bad["id"]]
    board = leaderboard(db, 100_000)
    assert board[0]["metrics"]["n"] > 0 and board[0]["status"] == "active"


def test_attribution_report(settings):
    db = Database(settings.data_dir / "attr.db")
    _fake_trades(db, "a#1", "trend_ema", 120, 0.1, seed=5)
    _fake_trades(db, "b#1", "meanrev_bb", 120, -0.1, seed=6)
    trades = db.trades(limit=1000)
    rep = build_report(trades, window_days=45, min_trades=30, now=NOW)
    assert rep["n_trades"] == 240 and not rep["insufficient"]
    assert rep["features"] and rep["heatmap"] and rep["by_family"]
    names = [f["name"] for f in rep["features"]]
    assert "rsi_14" in names and "hour_et" in names
    md_txt = digest_md(rep)
    assert "Attribution window" in md_txt and "By family" in md_txt
    for f in rep["filters"]:
        assert f["verdict"] in ("candidate", "rejected")
    small = build_report(trades[:10], window_days=45, min_trades=30, now=NOW)
    assert small["insufficient"] and small["overall"]["n"] == 10


def test_budget_governor(settings):
    db = Database(settings.data_dir / "budget.db")
    gov = BudgetGovernor(db, settings)
    cap = gov.cap()
    assert cap == pytest.approx(settings.claude.weekly_allowance_usd * settings.claude.weekly_share)
    ok, _ = gov.can_start(1.0, NOW)
    assert ok
    rid = db.insert_run("analyst", "sonnet", "digest")
    db.update_run(rid, status="ok", cost_usd=cap - 0.5, finished_at=to_iso(NOW))
    ok, reason = gov.can_start(1.0, NOW)
    assert not ok and "budget" in reason
    cal = gov.calibrate(10.0, NOW)
    assert cal["ok"] and cal["allowance_usd"] == pytest.approx((cap - 0.5) / 0.10)
    st = gov.status(NOW)
    assert st["spent_usd"] == pytest.approx(cap - 0.5) and len(st["history"]) == 8


def test_claude_result_parsing_and_static_check():
    raw = json.dumps({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.12, "num_turns": 3,
                      "usage": {"input_tokens": 10, "output_tokens": 5}, "structured_output": {"analysis_md": "ok", "proposals": []}})
    data = claude_lab.parse_cli_result("some log line\n" + raw)
    assert claude_lab.extract_structured(data)["analysis_md"] == "ok"
    data2 = {"result": json.dumps({"analysis_md": "x", "proposals": [{"type": "retire", "rationale": "r"}]})}
    assert claude_lab.extract_structured(data2)["proposals"][0]["type"] == "retire"
    good = "from sentinel.strategies.base import Strategy, register\n@register\nclass X(Strategy):\n    family='x'\n"
    assert check_evolved_source(good) == []
    bad = "import os\nfrom sentinel.strategies.base import Strategy, register\n@register\nclass X(Strategy):\n    family='x'\n    def on_bar(self, ctx, s, df):\n        return eval('1')\n"
    probs = check_evolved_source(bad)
    assert any("import not allowed: os" in p for p in probs) and any("eval" in p for p in probs)


def test_proposal_gating_rejects_and_accepts(settings, md, monkeypatch):
    db = Database(settings.data_dir / "gate.db")
    seed_population(db, settings)
    target = next(v for v in db.variants() if v["family"] == "trend_ema")
    pid = db.insert_proposal(None, "param_change", target["id"], {"params": {}}, "no-op")
    res = claude_lab.decide_proposal(db, settings, md, pid, now=NOW)
    assert res["status"] == "rejected" and "no parameter" in res["decision_reason"]
    pid2 = db.insert_proposal(None, "retire", target["id"], {}, "kill it")
    res2 = claude_lab.decide_proposal(db, settings, md, pid2, now=NOW)
    assert res2["status"] == "rejected"
    # force a candidate through the gate with a stubbed backtest
    def fake_wf(s, m, fam, params, **k):
        improved = params.get("fast") == 7 or bool(params.get("filters"))
        e = 0.4 if improved else 0.1
        return {"n": 40, "expectancy_r": e, "expectancy_ci": [e - 0.3, e + 0.3], "win_rate": 0.55, "profit_factor": 1.6, "max_dd_pct": 2.0, "pnl": 500.0, "sharpe": 1.2,
                "segments": [], "segments_positive": 3, "splits": 3, "days": 45, "trades": []}

    monkeypatch.setattr(claude_lab, "walk_forward", fake_wf)
    monkeypatch.setattr(claude_lab, "control_baseline", lambda *a, **k: {"n": 40, "expectancy_r": -0.2})
    pid3 = db.insert_proposal(None, "param_change", target["id"], {"params": {"fast": 7}}, "faster")
    res3 = claude_lab.decide_proposal(db, settings, md, pid3, now=NOW)
    assert res3["status"] == "accepted" and res3["created_variant_id"]
    child = db.variant(res3["created_variant_id"])
    assert child["params"]["fast"] == 7 and child["status"] == "incubating" and child["parent_id"] == target["id"]
    pid4 = db.insert_proposal(None, "filter", target["id"], {"filter": {"feature": "hour_et", "op": "not_in", "values": [9]}}, "skip the open")
    res4 = claude_lab.decide_proposal(db, settings, md, pid4, now=NOW)
    assert res4["status"] == "accepted" and db.variant(res4["created_variant_id"])["params"]["filters"][0]["feature"] == "hour_et"
