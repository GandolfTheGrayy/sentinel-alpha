"""squeeze: Bollinger-inside-Keltner volatility squeeze, trade the expansion in the momentum direction."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, r_target, register, safe_float


@register
class Squeeze(Strategy):
    family = "squeeze"
    description = "After at least k bars of Bollinger bands inside the Keltner channel, enter when the bands release, in the direction of momentum."
    timeframe = "15m"
    markets = ["equities", "crypto"]
    warmup_bars = 80
    DEFAULTS = {"tf": "15m", "min_squeeze": 6, "kc_mult": 1.5, "bb_k": 2.0, "stop_atr": 1.5, "r_mult": 2.0, "max_hold": 48}
    PARAM_SPACE = {
        "tf": ("choice", ["15m", "1h"]),
        "min_squeeze": ("int", 3, 20),
        "kc_mult": ("float", 1.0, 2.5),
        "bb_k": ("float", 1.5, 2.5),
        "stop_atr": ("float", 0.8, 3.0),
        "r_mult": ("float", 1.0, 4.0),
        "max_hold": ("int", 10, 200),
    }

    def _state(self, ctx: Context, s: str) -> pd.DataFrame:
        def fn(d: pd.DataFrame) -> pd.DataFrame:
            lo, mid, hi = ind.bollinger(d["c"], 20, self.p["bb_k"])
            klo, kmid, khi = ind.keltner(d, 20, self.p["kc_mult"])
            sq = (lo > klo) & (hi < khi)
            mom = d["c"] - ind.sma(d["c"], 20)
            return pd.DataFrame({"sq": sq.astype(float), "mom": mom, "atr": ind.atr(d, 14)})
        return ctx.series(s, f"squeeze{self.p['kc_mult']:.2f}_{self.p['bb_k']:.2f}", fn)

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s):
            return []
        st = self._state(ctx, s)
        k = self.p["min_squeeze"]
        if len(st) < k + 2:
            return []
        prev = st["sq"].iloc[-(k + 1):-1]
        if prev.sum() < k or st["sq"].iloc[-1] != 0.0:
            return []
        mom, av = safe_float(st["mom"].iloc[-1]), safe_float(st["atr"].iloc[-1])
        if mom is None or av is None or av <= 0:
            return []
        price = float(df["c"].iloc[-1])
        side = "long" if mom > 0 else "short"
        if side == "short" and not ctx.allow_short:
            return []
        stop = atr_stop(price, av, self.p["stop_atr"], side)
        return [Signal(s, side, stop=stop, target=r_target(price, stop, self.p["r_mult"], side), max_hold_bars=self.p["max_hold"],
                       strength=min(1.0, abs(mom) / av), reason=f"squeeze release after {int(prev.sum())} bars, momentum {side}")]
