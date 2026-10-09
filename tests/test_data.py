from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sentinel.data import indicators as ind
from sentinel.data.features import snapshot
from sentinel.data.synthetic import day_bars
from sentinel.util.clock import floor_ts, next_fire, next_open_close, session_at
from tests.conftest import NOW


def test_synthetic_is_deterministic_and_session_bounded():
    a = day_bars("AAPL", NOW.date())
    b = day_bars("AAPL", NOW.date())
    assert len(a) == 390 and a.equals(b)
    assert a.index[0].hour == 13 and a.index[0].minute == 30  # 09:30 ET in October (EDT)
    assert day_bars("AAPL", datetime(2026, 10, 10).date()).empty  # Saturday
    assert len(day_bars("BTC/USD", datetime(2026, 10, 10).date())) == 1440
    assert (a["h"] >= a[["o", "c"]].max(axis=1)).all() and (a["l"] <= a[["o", "c"]].min(axis=1)).all()


def test_resample_alignment_and_no_lookahead(md):
    ts = int(NOW.timestamp())
    f15 = md.frame("SPY", "15m", end_ts=ts)
    assert (f15.index + pd.Timedelta(minutes=15) <= NOW).all()
    ny = f15.index.tz_convert("America/New_York")
    assert set(ny.minute.unique()) <= {0, 15, 30, 45}
    f1h = md.frame("SPY", "1h", end_ts=ts)
    assert set(f1h.index.tz_convert("America/New_York").minute.unique()) == {30}  # aligned to 09:30
    daily = md.frame("SPY", "1d", end_ts=ts)
    assert daily.index[-1].tz_convert("America/New_York").date() < NOW.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date()
    # last closed minute bar at a given time must start one minute earlier
    bar = md.store.minute_bar("SPY", ts)
    assert bar is not None and bar[0] == ts - 60
    assert md.last_price("SPY", end_ts=ts) == bar[4]
    assert md.store.last_closed_bucket("SPY", "15m", ts) == int(f15.index[-1].timestamp())


def test_indicators_basic_shapes():
    idx = pd.date_range("2026-01-01", periods=300, freq="15min", tz="UTC")
    c = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 300)) + 100, index=idx)
    df = pd.DataFrame({"o": c, "h": c + 0.5, "l": c - 0.5, "c": c, "v": 1000.0}, index=idx)
    r = ind.rsi(c, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    a = ind.atr(df, 14).dropna()
    assert (a > 0).all()
    adx = ind.adx(df, 14).dropna()
    assert ((adx >= 0) & (adx <= 100)).all()
    lo, mid, hi = ind.bollinger(c, 20, 2.0)
    assert (hi.dropna() >= mid.dropna()).all()
    pos = ind.bollinger_pos(c, 20, 2.0).dropna()
    assert pos.between(-1, 2).all()


def test_feature_snapshot(md):
    f = snapshot(md, "AAPL", "15m", int(NOW.timestamp()), benchmark="SPY", extra={"signal_strength": 0.4})
    for key in ("rsi_14", "atr_pct", "bb_pos", "hour_et", "dow", "session", "mkt_ret_5d", "signal_strength", "is_crypto"):
        assert key in f, key
    assert f["is_crypto"] == 0.0 and f["session"] == 3.0  # 16:00 ET is post-market
    assert all(isinstance(v, float) for v in f.values())


def test_calendar_helpers():
    assert session_at(datetime(2026, 10, 8, 15, 0, tzinfo=timezone.utc)) == "regular"
    assert session_at(datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc)) == "closed"
    assert session_at(datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)) == "closed"  # Thanksgiving
    o, c = next_open_close(datetime(2026, 10, 8, 21, 0, tzinfo=timezone.utc))
    assert o.date() == datetime(2026, 10, 9).date()
    assert floor_ts(1000, "15m") == 900
    nf = next_fire("Sun 10:00", datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc))
    assert nf.weekday() == 6 and nf.hour == 14
