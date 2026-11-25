from datetime import datetime, timezone

import pytest

from sentinel.config import CostsCfg, RiskCfg
from sentinel.engine.broker import SimBroker
from sentinel.engine.exits import Bar, check_exit, update_marks
from sentinel.engine.ledger import Ledger
from sentinel.engine.risk import size_order
from sentinel.engine.trader import passes_filters
from sentinel.store.db import to_iso


def test_size_order_risk_based_and_capped():
    cfg = RiskCfg()
    r = size_order(price=100.0, stop=98.0, sleeve_equity=10_000, total_equity=100_000, gross_exposure=0.0, symbol="AAPL", cfg=cfg)
    assert r.qty == 50  # 1% of 10k = $100 risk / $2 per share
    r = size_order(price=100.0, stop=99.9, sleeve_equity=10_000, total_equity=100_000, gross_exposure=0.0, symbol="AAPL", cfg=cfg)
    assert r.qty * 100 <= 10_000 * cfg.max_position_pct_of_sleeve / 100 + 1e-6  # capped by sleeve share
    r = size_order(price=65_000.0, stop=64_000.0, sleeve_equity=10_000, total_equity=100_000, gross_exposure=0.0, symbol="BTC/USD", cfg=cfg)
    assert 0 < r.qty < 1 and r.notional <= 6_000 + 1e-6  # $100 risk wants 0.1 BTC; the 60 % sleeve cap trims it to $6k
    r = size_order(price=100.0, stop=98.0, sleeve_equity=10_000, total_equity=100_000, gross_exposure=100_000, symbol="AAPL", cfg=cfg)
    assert r.qty == 0 and "exposure" in r.reason


def test_exit_rules_intrabar_and_trailing():
    lot = {"side": "long", "entry_price": 100.0, "stop": 98.0, "target": 104.0, "trail_atr": None, "hwm": 100.0, "lwm": 100.0}
    assert check_exit(lot, Bar(0, 100, 101, 99, 100.5), 60, intrabar=True, session_last_minute=False) is None
    reason, px = check_exit(lot, Bar(0, 99, 99.5, 97.5, 98.5), 60, intrabar=True, session_last_minute=False)
    assert reason == "stop" and px == 98.0
    reason, px = check_exit(lot, Bar(0, 97.0, 99.5, 96.5, 98.5), 60, intrabar=True, session_last_minute=False)
    assert reason == "stop" and px == 97.0  # gapped through
    reason, px = check_exit(lot, Bar(0, 103, 105, 102.5, 104.5), 60, intrabar=True, session_last_minute=False)
    assert reason == "target" and px == 104.0
    lot2 = {"side": "long", "entry_price": 100.0, "stop": 98.0, "target": None, "trail_atr": 1.5, "hwm": 100.0, "lwm": 100.0}
    update_marks(lot2, Bar(0, 100, 103, 100, 102))
    assert lot2["stop"] == pytest.approx(101.5)
    update_marks(lot2, Bar(0, 102, 102.5, 101, 101.2))
    assert lot2["stop"] == pytest.approx(101.5)  # never ratchets down
    assert check_exit({"side": "short", "entry_price": 100, "stop": 103, "target": None, "flat_at_session_end": True}, Bar(0, 100, 100, 100, 100), 60, intrabar=False, session_last_minute=True)[0] == "session_end"
    assert check_exit({"side": "long", "entry_price": 100, "stop": None, "target": None, "max_hold_ts_epoch": 50}, Bar(0, 100, 100, 100, 100), 60, intrabar=False, session_last_minute=False)[0] == "time"


def test_sim_broker_costs():
    b = SimBroker(10_000, CostsCfg(equity_slippage_bps=10, crypto_fee_bps=25))
    f = b.market_order("AAPL", "buy", 10, ref_price=100.0)
    assert f.price == pytest.approx(100.1) and b.positions()["AAPL"] == 10
    f2 = b.market_order("AAPL", "sell", 10, ref_price=100.0)
    assert f2.price == pytest.approx(99.9) and not b.positions()
    assert b.cash == pytest.approx(10_000 - 1001 + 999)
    f3 = b.market_order("BTC/USD", "buy", 0.01, ref_price=60_000)
    assert f3.fees == pytest.approx(0.01 * 60_000 * (1 + 0.0008) * 0.0025, rel=1e-6)


def test_ledger_close_math():
    led = Ledger(None, persist=False)
    now = datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)
    lot = led.open({"variant_id": "v#1", "family": "f", "symbol": "AAPL", "side": "short", "qty": 10, "entry_ts": to_iso(now), "entry_price": 100.0,
                    "stop": 102.0, "risk_per_unit": 2.0, "reason": "t", "features": {"x": 1.0}})
    update_marks(lot, Bar(0, 100, 101, 95, 96))
    t = led.close(lot, datetime(2026, 10, 8, 15, 0, tzinfo=timezone.utc), 96.0, "target", fees=1.0)
    assert t["pnl"] == pytest.approx(39.0) and t["pnl_r"] == pytest.approx(2.0) and t["hold_minutes"] == 60
    assert t["mfe_r"] == pytest.approx(2.5) and t["mae_r"] == pytest.approx(-0.5)
    assert led.variant_pnl["v#1"] == pytest.approx(39.0) and not led.lots


def test_filters():
    feats = {"hour_et": 9.0, "atr_pct": 0.8}
    assert passes_filters([{"feature": "hour_et", "op": "not_in", "values": [15]}], feats)
    assert not passes_filters([{"feature": "hour_et", "op": "not_in", "values": [9, 15]}], feats)
    assert passes_filters([{"feature": "atr_pct", "op": "between", "values": [0.5, 1.0]}, {"feature": "missing", "op": "gt", "value": 5}], feats)
    assert not passes_filters([{"feature": "atr_pct", "op": "gt", "value": 1.0}], feats)


class _FakeBroker:
    """Reports whatever equity the test sets; fills instantly at the reference price."""

    name = "fake"

    def __init__(self, equity: float):
        self.equity = equity

    def account(self) -> dict:
        return {"equity": self.equity, "cash": self.equity, "buying_power": self.equity, "last_equity": self.equity}

    def positions(self) -> dict:
        return {}

    def clock(self) -> dict:
        return {}

    def market_order(self, symbol, side, qty, *, ref_price=None, extended_hours=False, client_id=None):
        from sentinel.engine.broker import Fill
        return Fill(symbol, side, qty, float(ref_price), 0.0, datetime(2026, 10, 8, tzinfo=timezone.utc), client_id or "x")


def _trader(settings, equity: float):
    from sentinel.engine.trader import Trader
    events: list[tuple[str, str, str]] = []
    broker = _FakeBroker(equity)
    t = Trader(settings, None, broker, Ledger(None, persist=False), [], {}, on_event=lambda k, m, lvl, d: events.append((k, lvl, m)))
    t._day_rollover(datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc))
    return t, broker, events


def test_daily_halt_ignores_broker_equity_the_ledger_cannot_explain(settings):
    t, broker, events = _trader(settings, 100_000.0)
    # the broker suddenly reports -81% while the ledger has no open lots and no closed trades today
    broker.equity = 19_000.0
    t._risk_checks(1_000)
    assert not t.state.halted and t.state.equity_suspect and t.state.entries_paused_day is None
    assert any(k == "risk" and lvl == "warn" and "disagrees with the ledger" in m for k, lvl, m in events)
    # the reading reverts: one info event, no flapping warnings
    broker.equity = 100_050.0
    t._risk_checks(1_060)
    assert not t.state.equity_suspect and sum("disagrees" in m for _, _, m in events) == 1
    assert any("agrees with the ledger again" in m for _, _, m in events)


def test_daily_halt_fires_when_ledger_agrees_and_clears_at_rollover(settings):
    t, broker, events = _trader(settings, 10_000.0)
    lot = t.ledger.open({"variant_id": "v#1", "family": "f", "symbol": "AAPL", "side": "long", "qty": 100, "entry_ts": to_iso(datetime(2026, 10, 8, 14, 0, tzinfo=timezone.utc)),
                         "entry_price": 100.0, "stop": 98.0, "risk_per_unit": 2.0, "reason": "t", "features": {}})
    # AAPL drops 7%: the ledger's unrealised loss and the broker's equity agree, so the halt is real
    t.state.last_prices = {"AAPL": 93.0}
    broker.equity = 9_300.0
    t._risk_checks(1_000)
    assert t.state.halted and t.state.halt_reason == "risk" and not t.ledger.lots and not t.state.equity_suspect
    assert t.state.day_realized == pytest.approx(-700.0)
    assert any(lvl == "error" and "halted until the next trading day" in m for _, lvl, m in events)
    # same day: stays halted
    t._day_rollover(datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc))
    assert t.state.halted
    # next NY day: the risk halt clears itself and the day baseline restarts from the broker figure
    t._day_rollover(datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc))
    assert not t.state.halted and t.state.halt_reason is None and t.state.day_start_equity == 9_300.0 and t.state.day_realized == 0.0
    assert any("halt cleared at the day rollover" in m for _, _, m in events)
    assert lot["id"] not in t.ledger.lots


def test_operator_halt_survives_rollover_and_bogus_equity_does_not_become_the_day_baseline(settings):
    t, broker, events = _trader(settings, 100_000.0)
    t.state.halted, t.state.halt_reason = True, "operator"
    t._day_rollover(datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc))
    assert t.state.halted and t.state.halt_reason == "operator"
    # the broker is still reporting nonsense at the rollover: carry the ledger figure forward instead
    t._risk_checks(1_000)
    broker.equity = 16_000.0
    t._day_rollover(datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc))
    assert t.state.day_start_equity == pytest.approx(100_000.0)
    assert any("carrying the ledger figure forward" in m for _, _, m in events)
