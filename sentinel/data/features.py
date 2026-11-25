"""Feature snapshot taken at every trade entry (and available to strategies).

All values are plain floats so they can be bucketed, regressed and sent to Claude as a
compact table. NaNs are dropped from the snapshot rather than stored.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from sentinel.data import indicators as ind
from sentinel.data.market import MarketData, is_crypto
from sentinel.util.clock import UTC, minutes_since_open, session_at, to_ny

SESSION_CODE = {"closed": 0.0, "pre": 1.0, "regular": 2.0, "post": 3.0}

FEATURE_DOC = {
    "ret_1": "return over the last bar of the strategy timeframe",
    "ret_5": "return over the last 5 bars",
    "ret_20": "return over the last 20 bars",
    "ret_1d": "1-day return",
    "ret_5d": "5-day return",
    "ret_20d": "20-day return",
    "rsi_2": "RSI(2) on the strategy timeframe",
    "rsi_14": "RSI(14) on the strategy timeframe",
    "adx_14": "ADX(14) on the strategy timeframe",
    "atr_pct": "ATR(14) as % of price",
    "bb_pos": "position within Bollinger(20,2): 0 lower band, 1 upper band",
    "vol_ratio": "volume of last bar vs 20-bar average",
    "vwap_dist_pct": "% distance from session VWAP (intraday only)",
    "dist_sma50_pct": "% distance from the 50-day SMA",
    "dist_sma200_pct": "% distance from the 200-day SMA",
    "rvol_20d": "20-day realised volatility (annualised)",
    "gap_pct": "today's open vs previous close (%)",
    "hour_et": "hour of day, America/New_York",
    "dow": "day of week, Monday=0",
    "session": "0 closed, 1 pre, 2 regular, 3 post",
    "min_since_open": "minutes since the 09:30 open (equities)",
    "regime_trend": "1 if ADX(14) on daily bars > 25",
    "regime_vol": "0 low / 1 mid / 2 high tercile of ATR% over the last 100 bars",
    "mkt_ret_1d": "benchmark 1-day return",
    "mkt_ret_5d": "benchmark 5-day return",
    "mkt_rvol_20d": "benchmark 20-day realised vol",
    "mkt_dist_sma50_pct": "benchmark % distance from 50-day SMA",
    "signal_strength": "strategy-specific conviction 0..1",
    "n_open_positions": "open lots across all variants at entry",
    "sleeve_dd_pct": "variant sleeve drawdown from its peak at entry (%)",
    "is_crypto": "1 for crypto symbols",
}


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _last(s: pd.Series) -> float | None:
    if s is None or len(s) == 0:
        return None
    return _f(s.iloc[-1])


def snapshot(md: MarketData, symbol: str, tf: str, ts: int, *, benchmark: str | None = None, extra: dict | None = None) -> dict[str, float]:
    """Compute the feature snapshot for symbol at epoch-second ts using closed bars only."""
    out: dict[str, float | None] = {}
    crypto = is_crypto(symbol)
    df = md.frame(symbol, tf, end_ts=ts, n=260)
    if len(df) >= 25:
        c = df["c"]
        out["ret_1"] = _last(ind.returns(c, 1))
        out["ret_5"] = _last(ind.returns(c, 5))
        out["ret_20"] = _last(ind.returns(c, 20))
        out["rsi_2"] = _last(ind.rsi(c, 2))
        out["rsi_14"] = _last(ind.rsi(c, 14))
        out["adx_14"] = _last(ind.adx(df, 14))
        a = ind.atr(df, 14)
        atr_pct = (a / c * 100.0)
        out["atr_pct"] = _last(atr_pct)
        out["bb_pos"] = _last(ind.bollinger_pos(c, 20, 2.0))
        out["vol_ratio"] = _last(ind.volume_ratio(df["v"], 20))
        tail = atr_pct.dropna().iloc[-100:]
        if len(tail) >= 30 and out["atr_pct"] is not None:
            q1, q2 = np.quantile(tail.values, [1 / 3, 2 / 3])
            out["regime_vol"] = 0.0 if out["atr_pct"] <= q1 else (1.0 if out["atr_pct"] <= q2 else 2.0)
        if not crypto and tf in ("1m", "5m", "15m", "30m"):
            vw = ind.session_vwap(df)
            v = _last(vw)
            lc = _last(c)
            if v and lc:
                out["vwap_dist_pct"] = (lc / v - 1.0) * 100.0
    d = md.frame(symbol, "1d", end_ts=ts, n=260)
    price_now = md.last_price(symbol, end_ts=ts)
    if len(d) >= 21 and price_now:
        dc = d["c"]
        out["ret_1d"] = price_now / float(dc.iloc[-1]) - 1.0
        out["ret_5d"] = price_now / float(dc.iloc[-5]) - 1.0 if len(dc) >= 5 else None
        out["ret_20d"] = price_now / float(dc.iloc[-20]) - 1.0 if len(dc) >= 20 else None
        s50 = _last(ind.sma(dc, 50))
        s200 = _last(ind.sma(dc, 200))
        if s50:
            out["dist_sma50_pct"] = (price_now / s50 - 1.0) * 100.0
        if s200:
            out["dist_sma200_pct"] = (price_now / s200 - 1.0) * 100.0
        out["rvol_20d"] = _last(ind.realized_vol(dc, 20, 365.0 if crypto else 252.0))
        if len(d) >= 30:
            adx_d = _last(ind.adx(d, 14))
            if adx_d is not None:
                out["regime_trend"] = 1.0 if adx_d > 25 else 0.0
        today = md.frame(symbol, "1d", end_ts=ts, n=1, include_partial=True)
        if not today.empty and today.index[-1] > d.index[-1]:
            out["gap_pct"] = (float(today["o"].iloc[-1]) / float(dc.iloc[-1]) - 1.0) * 100.0
    if benchmark:
        b = md.frame(benchmark, "1d", end_ts=ts, n=260)
        bp = md.last_price(benchmark, end_ts=ts)
        if len(b) >= 21 and bp:
            bc = b["c"]
            out["mkt_ret_1d"] = bp / float(bc.iloc[-1]) - 1.0
            out["mkt_ret_5d"] = bp / float(bc.iloc[-5]) - 1.0
            out["mkt_rvol_20d"] = _last(ind.realized_vol(bc, 20, 365.0 if is_crypto(benchmark) else 252.0))
            s50b = _last(ind.sma(bc, 50))
            if s50b:
                out["mkt_dist_sma50_pct"] = (bp / s50b - 1.0) * 100.0
    dt = datetime.fromtimestamp(ts, UTC)
    ny = to_ny(dt)
    out["hour_et"] = float(ny.hour)
    out["dow"] = float(ny.weekday())
    out["session"] = SESSION_CODE[session_at(dt)]
    mso = minutes_since_open(dt)
    if mso is not None and not crypto:
        out["min_since_open"] = float(mso)
    out["is_crypto"] = 1.0 if crypto else 0.0
    for k, v in (extra or {}).items():
        out[k] = _f(v)
    return {k: round(v, 6) for k, v in out.items() if v is not None}
