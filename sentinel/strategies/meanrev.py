"""meanrev_bb: Bollinger band + RSI(2) extremes, exit at the middle band (intraday equities)."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, register, safe_float


@register
class MeanRevBB(Strategy):
    family = "meanrev_bb"
    description = "Fade closes outside the Bollinger bands when RSI(2) is extreme; target the middle band; optional daily trend filter."
    timeframe = "15m"
    markets = ["equities"]
    intraday_only = True
    warmup_bars = 60
    DEFAULTS = {"tf": "15m", "bb_n": 20, "bb_k": 2.0, "rsi_lo": 10.0, "rsi_hi": 90.0, "stop_atr": 1.5, "max_hold": 16, "trend_filter": 1}
    PARAM_SPACE = {
        "tf": ("choice", ["5m", "15m"]),
        "bb_n": ("int", 10, 40),
        "bb_k": ("float", 1.5, 3.0),
        "rsi_lo": ("float", 3.0, 25.0),
        "rsi_hi": ("float", 75.0, 97.0),
        "stop_atr": ("float", 0.8, 3.0),
        "max_hold": ("int", 4, 60),
        "trend_filter": ("choice", [0, 1]),
    }

    def _bands(self, ctx: Context, s: str):
        key = f"bb{self.p['bb_n']}_{self.p['bb_k']:.2f}"
        bands = ctx.series(s, key, lambda d: pd.DataFrame(dict(zip(("lo", "mid", "hi"), ind.bollinger(d["c"], self.p["bb_n"], self.p["bb_k"])))))
        rsi2 = ctx.series(s, "rsi2", lambda d: ind.rsi(d["c"], 2))
        atr = ctx.series(s, "atr14", lambda d: ind.atr(d, 14))
        return bands, rsi2, atr

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s) or ctx.session != "regular":
            return []
        bands, rsi2, atr = self._bands(ctx, s)
        if len(bands) < 2:
            return []
        lo, mid, hi = (safe_float(bands[c].iloc[-1]) for c in ("lo", "mid", "hi"))
        r, av = safe_float(rsi2.iloc[-1]), safe_float(atr.iloc[-1])
        if None in (lo, mid, hi, r, av):
            return []
        price = float(df["c"].iloc[-1])
        above_trend = True
        if self.p["trend_filter"]:
            d = ctx.daily(s, 60)
            if len(d) >= 50:
                sma50 = safe_float(ind.sma(d["c"], 50).iloc[-1])
                above_trend = bool(sma50 and price > sma50)
        if price < lo and r < self.p["rsi_lo"] and above_trend:
            stop = atr_stop(price, av, self.p["stop_atr"], "long")
            if mid > price:
                return [Signal(s, "long", stop=stop, target=mid, max_hold_bars=self.p["max_hold"], flat_at_session_end=True,
                               strength=min(1.0, (self.p["rsi_lo"] - r) / self.p["rsi_lo"] + 0.3), reason=f"close below BB lower, RSI2 {r:.0f}")]
        if price > hi and r > self.p["rsi_hi"] and ctx.allow_short and not (self.p["trend_filter"] and above_trend):
            stop = atr_stop(price, av, self.p["stop_atr"], "short")
            if mid < price:
                return [Signal(s, "short", stop=stop, target=mid, max_hold_bars=self.p["max_hold"], flat_at_session_end=True,
                               strength=min(1.0, (r - self.p["rsi_hi"]) / (100 - self.p["rsi_hi"]) + 0.3), reason=f"close above BB upper, RSI2 {r:.0f}")]
        return []

    def manage(self, ctx: Context, lot: dict, df: pd.DataFrame) -> str | None:
        bands, _, _ = self._bands(ctx, lot["symbol"])
        if len(bands) < 1:
            return None
        mid = safe_float(bands["mid"].iloc[-1])
        price = float(df["c"].iloc[-1])
        if mid is None:
            return None
        if lot["side"] == "long" and price >= mid:
            return "signal"
        if lot["side"] == "short" and price <= mid:
            return "signal"
        return None
