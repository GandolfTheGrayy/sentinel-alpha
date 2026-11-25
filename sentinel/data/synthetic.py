"""Deterministic synthetic market for --sim mode, demos and tests.

Prices follow a regime-switching GBM with a shared market factor (one for equities, one
for crypto), symbol betas, intraday U-shaped volatility/volume, overnight gaps for
equities and 24/7 trading for crypto. Everything is seeded per (symbol, day) so any
date range can be regenerated identically in any order.
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from sentinel.data.market import COLS, is_crypto
from sentinel.util.clock import NY, UTC, is_trading_day, session_bounds

GENESIS = date(2025, 1, 1)
MINUTES_EQ = 390
MINUTES_CR = 1440


def _seed(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


def _symbol_profile(symbol: str) -> dict:
    rng = np.random.default_rng(_seed("profile", symbol))
    crypto = is_crypto(symbol)
    base_price = float(rng.choice([25, 60, 120, 180, 250, 400, 600]) if not crypto else rng.choice([0.4, 1.8, 25, 90, 180, 3200, 65000]))
    if symbol.upper().startswith("BTC"):
        base_price = 65000.0
    elif symbol.upper().startswith("ETH"):
        base_price = 3200.0
    elif symbol.upper() in ("SPY",):
        base_price = 560.0
    elif symbol.upper() in ("QQQ",):
        base_price = 480.0
    daily_vol = float(rng.uniform(0.012, 0.03)) if not crypto else float(rng.uniform(0.025, 0.05))
    beta = float(rng.uniform(0.6, 1.5))
    drift = float(rng.uniform(-0.0002, 0.0006))
    return {"base_price": base_price, "daily_vol": daily_vol, "beta": beta, "drift": drift, "crypto": crypto}


@lru_cache(maxsize=64)
def _factor_daily(factor: str, n_days: int) -> np.ndarray:
    """Daily factor returns with slowly varying regimes (trend / chop, low / high vol)."""
    rng = np.random.default_rng(_seed("factor", factor))
    out = np.zeros(n_days)
    i = 0
    while i < n_days:
        length = int(rng.integers(8, 40))
        drift = float(rng.choice([-0.0025, -0.001, 0.0, 0.0008, 0.0015, 0.003]))
        vol = float(rng.choice([0.006, 0.009, 0.014, 0.022]))
        if factor == "CRYPTO":
            drift *= 2.0
            vol *= 2.2
        seg = rng.normal(drift, vol, size=min(length, n_days - i))
        out[i:i + len(seg)] = seg
        i += len(seg)
    return out


def _day_index(d: date) -> int:
    return (d - GENESIS).days


@lru_cache(maxsize=256)
def _daily_closes(symbol: str, through_idx: int) -> np.ndarray:
    """Close price for every calendar day index 0..through_idx (equities carry forward on closed days)."""
    prof = _symbol_profile(symbol)
    factor = _factor_daily("CRYPTO" if prof["crypto"] else "MKT", through_idx + 1)
    rng = np.random.default_rng(_seed("daily", symbol))
    idio = rng.normal(0.0, prof["daily_vol"] * 0.7, size=through_idx + 1)
    closes = np.empty(through_idx + 1)
    p = prof["base_price"]
    for i in range(through_idx + 1):
        d = GENESIS + timedelta(days=i)
        if prof["crypto"] or is_trading_day(d):
            r = prof["drift"] + prof["beta"] * factor[i] + idio[i]
            p = p * float(np.exp(r))
        closes[i] = p
    return closes


def _u_shape(n: int) -> np.ndarray:
    x = np.linspace(-1, 1, n)
    return 0.6 + 0.9 * x ** 4


def day_bars(symbol: str, d: date) -> pd.DataFrame:
    """Minute bars for one day (empty for equities on non-trading days)."""
    prof = _symbol_profile(symbol)
    crypto = prof["crypto"]
    if not crypto and not is_trading_day(d):
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
    idx = _day_index(d)
    closes = _daily_closes(symbol, idx)
    prev_close = closes[idx - 1] if idx > 0 else prof["base_price"]
    target_close = closes[idx]
    rng = np.random.default_rng(_seed("intraday", symbol, d.isoformat()))
    if crypto:
        n = MINUTES_CR
        start = datetime.combine(d, time(0, 0), tzinfo=UTC)
        gap = 0.0
    else:
        n = MINUTES_EQ
        bounds = session_bounds(d)
        assert bounds is not None
        start = bounds[0]
        if (bounds[1] - bounds[0]).total_seconds() / 60 < n:
            n = int((bounds[1] - bounds[0]).total_seconds() / 60)
        gap = float(rng.normal(0.0, prof["daily_vol"] * 0.35))
    open_price = prev_close * float(np.exp(gap))
    # Brownian bridge from open to target close with U-shaped intraday vol
    shape = _u_shape(n)
    minute_vol = prof["daily_vol"] / np.sqrt(n) * shape
    noise = rng.normal(0.0, 1.0, size=n) * minute_vol
    noise -= noise.mean()
    total = float(np.log(target_close / open_price))
    log_path = np.cumsum(noise) + np.linspace(total / n, total, n)
    closes_m = open_price * np.exp(log_path)
    opens_m = np.concatenate([[open_price], closes_m[:-1]])
    wiggle = np.abs(rng.normal(0.0, 1.0, size=n)) * minute_vol * opens_m
    highs = np.maximum(opens_m, closes_m) + wiggle
    lows = np.minimum(opens_m, closes_m) - wiggle
    base_vol = 1_500_000 / max(prof["base_price"], 1.0) if not crypto else 40.0 * (65000 / max(prof["base_price"], 0.01)) ** 0.5
    vols = np.maximum(1.0, rng.lognormal(0.0, 0.5, size=n) * base_vol / n * shape * 4.0)
    ts = pd.date_range(start, periods=n, freq="1min", tz=UTC) if crypto else pd.date_range(start, periods=n, freq="1min")
    typical = (highs + lows + closes_m) / 3.0
    df = pd.DataFrame({"o": opens_m, "h": highs, "l": lows, "c": closes_m, "v": vols, "vwap": typical}, index=pd.DatetimeIndex(ts, name="ts"))
    if df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    return df.astype(float)


class SyntheticProvider:
    """BarProvider over the synthetic market. `clock` returns the simulated 'now'."""

    def __init__(self, clock=None):
        self.clock = clock or (lambda: datetime.now(UTC))

    def fetch_bars(self, symbols: list[str], tf: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        d0 = start.astimezone(UTC).date() - timedelta(days=1)
        d1 = end.astimezone(UTC).date() + timedelta(days=1)
        for s in symbols:
            frames = []
            d = d0
            while d <= d1:
                frames.append(day_bars(s, d))
                d += timedelta(days=1)
            df = pd.concat([f for f in frames if not f.empty]) if frames else day_bars(s, d0)
            if df.empty:
                out[s] = df
                continue
            df = df[(df.index >= pd.Timestamp(start).tz_convert(UTC)) & (df.index + pd.Timedelta(minutes=1) <= pd.Timestamp(end).tz_convert(UTC))]
            if tf == "1d":
                df = _to_daily(df, is_crypto(s))
            out[s] = df
        return out

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        now = self.clock()
        out = {}
        for s in symbols:
            df = self.fetch_bars([s], "1m", now - timedelta(days=4), now)[s]
            if not df.empty:
                out[s] = float(df["c"].iloc[-1])
        return out


def _to_daily(df: pd.DataFrame, crypto: bool) -> pd.DataFrame:
    if df.empty:
        return df
    if crypto:
        g = df.resample("1D", label="left", closed="left")
    else:
        g = df.tz_convert(NY).resample("1D", label="left", closed="left")
    out = g.agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}).dropna(subset=["o"])
    out["vwap"] = (out["h"] + out["l"] + out["c"]) / 3.0
    if not crypto:
        out = out.tz_convert(UTC)
    out.index.name = "ts"
    return out[COLS]
