"""crypto_mtf: multi-timeframe crypto trend following - 4h trend, 15m pullback entry, 24/7."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, r_target, register, safe_float


@register
class CryptoMTF(Strategy):
    family = "crypto_mtf"
    description = "Long-only crypto: 4h EMA trend up, wait for a 15m RSI pullback, enter on the first up-close; ATR stop, R target and trailing."
    timeframe = "15m"
    markets = ["crypto"]
    shortable = False
    regular_session_only = False
    warmup_bars = 100
    DEFAULTS = {"trend_ema": 50, "fast_ema": 20, "rsi_pullback": 35.0, "stop_atr": 2.0, "r_mult": 3.0, "trail_atr": 3.0, "max_hold": 192}
    PARAM_SPACE = {
        "trend_ema": ("int", 20, 120),
        "fast_ema": ("int", 8, 40),
        "rsi_pullback": ("float", 20.0, 50.0),
        "stop_atr": ("float", 1.0, 4.0),
        "r_mult": ("float", 1.5, 6.0),
        "trail_atr": ("float", 1.5, 6.0),
        "max_hold": ("int", 24, 600),
    }

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s):
            return []
        h4 = ctx.frame(s, "4h", n=200)
        if len(h4) < self.p["trend_ema"] + 5:
            return []
        trend = ctx.series(s, f"ema{self.p['trend_ema']}_4h", lambda d: ind.ema(d["c"], self.p["trend_ema"]), tf="4h", n=200)
        fast4 = ctx.series(s, f"ema{self.p['fast_ema']}_4h", lambda d: ind.ema(d["c"], self.p["fast_ema"]), tf="4h", n=200)
        t, f4 = safe_float(trend.iloc[-1]), safe_float(fast4.iloc[-1])
        c4 = float(h4["c"].iloc[-1])
        if None in (t, f4) or not (c4 > t and f4 > t):
            return []
        rsi = ctx.series(s, "rsi14", lambda d: ind.rsi(d["c"], 14))
        atr = ctx.series(s, "atr14", lambda d: ind.atr(d, 14))
        if len(rsi) < 4:
            return []
        recent = rsi.iloc[-4:-1]
        r0, av = safe_float(rsi.iloc[-1]), safe_float(atr.iloc[-1])
        if r0 is None or av is None:
            return []
        pulled_back = bool((recent < self.p["rsi_pullback"]).any())
        up_close = float(df["c"].iloc[-1]) > float(df["c"].iloc[-2])
        if pulled_back and up_close and r0 >= self.p["rsi_pullback"]:
            price = float(df["c"].iloc[-1])
            stop = atr_stop(price, av, self.p["stop_atr"], "long")
            return [Signal(s, "long", stop=stop, target=r_target(price, stop, self.p["r_mult"], "long"), trail_atr=av * self.p["trail_atr"],
                           max_hold_bars=self.p["max_hold"], strength=min(1.0, (c4 / t - 1.0) * 20 + 0.3),
                           reason=f"4h uptrend (EMA{self.p['trend_ema']}), 15m RSI pullback to {float(recent.min()):.0f}")]
        return []
