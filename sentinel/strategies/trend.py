"""trend_ema: EMA crossover with an ADX trend filter and an ATR trailing stop."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, register, safe_float


@register
class TrendEMA(Strategy):
    family = "trend_ema"
    description = "Enter on an EMA fast/slow crossover when ADX confirms a trend; ride it with an ATR trailing stop."
    timeframe = "15m"
    markets = ["equities", "crypto"]
    warmup_bars = 120
    DEFAULTS = {"tf": "15m", "fast": 12, "slow": 40, "adx_min": 22.0, "stop_atr": 2.0, "trail_atr": 2.5, "max_hold": 96}
    PARAM_SPACE = {
        "tf": ("choice", ["15m", "1h"]),
        "fast": ("int", 5, 30),
        "slow": ("int", 20, 120),
        "adx_min": ("float", 12.0, 40.0),
        "stop_atr": ("float", 1.0, 4.0),
        "trail_atr": ("float", 1.5, 5.0),
        "max_hold": ("int", 20, 300),
    }

    @classmethod
    def validate_params(cls, p):
        p = super().validate_params(p)
        if p.get("slow", 40) <= p.get("fast", 12):
            p["slow"] = int(p.get("fast", 12)) * 3
        return p

    def _lines(self, ctx: Context, s: str):
        fast = ctx.series(s, f"ema{self.p['fast']}", lambda d: ind.ema(d["c"], self.p["fast"]))
        slow = ctx.series(s, f"ema{self.p['slow']}", lambda d: ind.ema(d["c"], self.p["slow"]))
        adx = ctx.series(s, "adx14", lambda d: ind.adx(d, 14))
        atr = ctx.series(s, "atr14", lambda d: ind.atr(d, 14))
        return fast, slow, adx, atr

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s):
            return []
        fast, slow, adx, atr = self._lines(ctx, s)
        if len(fast) < 3 or len(adx) < 2:
            return []
        f1, f0, s1, s0 = safe_float(fast.iloc[-2]), safe_float(fast.iloc[-1]), safe_float(slow.iloc[-2]), safe_float(slow.iloc[-1])
        a, av = safe_float(adx.iloc[-1]), safe_float(atr.iloc[-1])
        if None in (f1, f0, s1, s0, a, av) or a < self.p["adx_min"]:
            return []
        price = float(df["c"].iloc[-1])
        strength = min(1.0, (a - self.p["adx_min"]) / 25.0 + 0.3)
        if f1 <= s1 and f0 > s0:
            return [Signal(s, "long", stop=atr_stop(price, av, self.p["stop_atr"], "long"), trail_atr=av * self.p["trail_atr"],
                           max_hold_bars=self.p["max_hold"], strength=strength, reason=f"EMA{self.p['fast']}/{self.p['slow']} cross up, ADX {a:.0f}")]
        if f1 >= s1 and f0 < s0 and ctx.allow_short:
            return [Signal(s, "short", stop=atr_stop(price, av, self.p["stop_atr"], "short"), trail_atr=av * self.p["trail_atr"],
                           max_hold_bars=self.p["max_hold"], strength=strength, reason=f"EMA{self.p['fast']}/{self.p['slow']} cross down, ADX {a:.0f}")]
        return []

    def manage(self, ctx: Context, lot: dict, df: pd.DataFrame) -> str | None:
        fast, slow, _, _ = self._lines(ctx, lot["symbol"])
        if len(fast) < 2:
            return None
        f0, s0 = safe_float(fast.iloc[-1]), safe_float(slow.iloc[-1])
        if f0 is None or s0 is None:
            return None
        if lot["side"] == "long" and f0 < s0:
            return "signal"
        if lot["side"] == "short" and f0 > s0:
            return "signal"
        return None
