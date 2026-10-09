"""Control families. They exist so the learning loop can separate signal from exit rules, costs and luck."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, r_target, register, safe_float


@register
class RandomEntry(Strategy):
    family = "random_entry"
    description = "CONTROL: random entries with the standard ATR stop / R target / time exits. Any real family must beat this."
    timeframe = "15m"
    markets = ["equities", "crypto"]
    is_control = True
    warmup_bars = 30
    DEFAULTS = {"entry_prob": 0.01, "stop_atr": 2.0, "r_mult": 2.0, "max_hold": 32, "long_only": 0}
    PARAM_SPACE = {}

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s) or ctx.rng.random() > self.p["entry_prob"]:
            return []
        av = safe_float(ctx.series(s, "atr14", lambda d: ind.atr(d, 14)).iloc[-1])
        if not av:
            return []
        price = float(df["c"].iloc[-1])
        side = "long" if (self.p["long_only"] or not ctx.allow_short or ctx.rng.random() < 0.5) else "short"
        stop = atr_stop(price, av, self.p["stop_atr"], side)
        return [Signal(s, side, stop=stop, target=r_target(price, stop, self.p["r_mult"], side), max_hold_bars=self.p["max_hold"],
                       flat_at_session_end=False, strength=0.5, reason="random control entry")]


@register
class BuyHold(Strategy):
    family = "buy_hold"
    description = "CONTROL: buy-and-hold the benchmark of each market with a very wide stop. The passive baseline."
    timeframe = "1d"
    markets = ["equities", "crypto"]
    trigger = ["09:35", "21:00"]
    is_control = True
    shortable = False
    regular_session_only = False
    warmup_bars = 5
    DEFAULTS = {"stop_pct": 50.0}
    PARAM_SPACE = {}

    def on_cycle(self, ctx: Context) -> list[Signal]:
        out = []
        for s in ctx.symbols:
            if ctx.has_lot(s):
                continue
            price = ctx.price(s)
            if not price:
                continue
            out.append(Signal(s, "long", stop=price * (1 - self.p["stop_pct"] / 100.0), strength=0.5, reason="benchmark buy & hold"))
        return out
