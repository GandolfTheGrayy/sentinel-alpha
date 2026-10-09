"""Breakout families: opening-range breakout (equities, intraday) and Donchian channel breakout."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, r_target, register, safe_float
from sentinel.util.clock import NY


@register
class BreakoutORB(Strategy):
    family = "breakout_orb"
    description = "Opening-range breakout: trade the first break of the opening N-minute range with volume confirmation; flat by the close."
    timeframe = "5m"
    markets = ["equities"]
    intraday_only = True
    warmup_bars = 30
    DEFAULTS = {"or_minutes": 30, "latest_entry_min": 150, "vol_mult": 1.2, "r_mult": 2.0, "stop_atr": 1.5, "max_hold": 60}
    PARAM_SPACE = {
        "or_minutes": ("choice", [15, 30, 60]),
        "latest_entry_min": ("int", 60, 300),
        "vol_mult": ("float", 0.8, 2.5),
        "r_mult": ("float", 1.0, 4.0),
        "stop_atr": ("float", 0.8, 3.0),
        "max_hold": ("int", 12, 78),
    }

    def __init__(self, params=None):
        super().__init__(params)
        self._traded: set[tuple[str, str]] = set()

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.session != "regular" or ctx.has_lot(s):
            return []
        ny = df.index.tz_convert(NY)
        today = ny[-1].date()
        if (s, str(today)) in self._traded:
            return []
        day = df[ny.date == today]
        n_or = max(1, int(self.p["or_minutes"] // 5))
        if len(day) <= n_or:
            return []
        mins_since_open = (len(day) - 1) * 5 + 5
        if mins_since_open > self.p["latest_entry_min"]:
            return []
        opening = day.iloc[:n_or]
        or_hi, or_lo = float(opening["h"].max()), float(opening["l"].min())
        last = day.iloc[-1]
        price = float(last["c"])
        vr = safe_float(ind.volume_ratio(df["v"], 20).iloc[-1])
        atr = safe_float(ctx.series(s, "atr14", lambda d: ind.atr(d, 14)).iloc[-1])
        if vr is None or atr is None or vr < self.p["vol_mult"]:
            return []
        side = None
        if price > or_hi and float(day.iloc[-2]["c"]) <= or_hi:
            side = "long"
        elif price < or_lo and float(day.iloc[-2]["c"]) >= or_lo and ctx.allow_short:
            side = "short"
        if not side:
            return []
        stop = atr_stop(price, atr, self.p["stop_atr"], side)
        # never risk more than the opening range itself
        if side == "long":
            stop = max(stop, or_lo) if or_lo < price else stop
        else:
            stop = min(stop, or_hi) if or_hi > price else stop
        self._traded.add((s, str(today)))
        rng_pct = (or_hi - or_lo) / price * 100
        return [Signal(s, side, stop=stop, target=r_target(price, stop, self.p["r_mult"], side), max_hold_bars=self.p["max_hold"], flat_at_session_end=True,
                       strength=min(1.0, vr / 3.0), reason=f"ORB {self.p['or_minutes']}m {side} break, range {rng_pct:.2f}%, vol x{vr:.1f}", extra={"or_range_pct": rng_pct})]


@register
class BreakoutDonchian(Strategy):
    family = "breakout_donchian"
    description = "Donchian channel breakout with a chandelier-style ATR trailing stop; works on equities and crypto."
    timeframe = "1h"
    markets = ["equities", "crypto"]
    warmup_bars = 80
    DEFAULTS = {"tf": "1h", "n": 20, "stop_atr": 2.0, "trail_atr": 3.0, "max_hold": 120, "min_atr_pct": 0.3}
    PARAM_SPACE = {
        "tf": ("choice", ["1h", "4h", "15m"]),
        "n": ("int", 10, 60),
        "stop_atr": ("float", 1.0, 4.0),
        "trail_atr": ("float", 1.5, 6.0),
        "max_hold": ("int", 24, 400),
        "min_atr_pct": ("float", 0.0, 1.5),
    }

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.has_lot(s):
            return []
        ch = ctx.series(s, f"donch{self.p['n']}", lambda d: pd.DataFrame(dict(zip(("up", "dn"), ind.donchian(d, self.p["n"])))))
        atr = ctx.series(s, "atr14", lambda d: ind.atr(d, 14))
        if len(ch) < 2:
            return []
        up, dn, av = safe_float(ch["up"].iloc[-1]), safe_float(ch["dn"].iloc[-1]), safe_float(atr.iloc[-1])
        if None in (up, dn, av):
            return []
        price = float(df["c"].iloc[-1])
        if av / price * 100 < self.p["min_atr_pct"]:
            return []
        if price > up:
            stop = atr_stop(price, av, self.p["stop_atr"], "long")
            return [Signal(s, "long", stop=stop, trail_atr=av * self.p["trail_atr"], max_hold_bars=self.p["max_hold"],
                           strength=min(1.0, (price - up) / av + 0.4), reason=f"Donchian({self.p['n']}) upside break")]
        if price < dn and ctx.allow_short:
            stop = atr_stop(price, av, self.p["stop_atr"], "short")
            return [Signal(s, "short", stop=stop, trail_atr=av * self.p["trail_atr"], max_hold_bars=self.p["max_hold"],
                           strength=min(1.0, (dn - price) / av + 0.4), reason=f"Donchian({self.p['n']}) downside break")]
        return []
