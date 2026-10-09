"""vwap_revert: fade stretched intraday moves away from session VWAP back toward it (equities)."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, register, safe_float
from sentinel.util.clock import NY


@register
class VWAPRevert(Strategy):
    family = "vwap_revert"
    description = "When price is stretched more than z ATRs from the session VWAP on a volume climax, fade it with VWAP as the target."
    timeframe = "5m"
    markets = ["equities"]
    intraday_only = True
    warmup_bars = 40
    DEFAULTS = {"tf": "5m", "z": 2.0, "vol_mult": 1.3, "min_after_open": 30, "stop_atr": 1.5, "max_hold": 24, "latest_entry_min": 330}
    PARAM_SPACE = {
        "tf": ("choice", ["5m", "15m"]),
        "z": ("float", 1.0, 4.0),
        "vol_mult": ("float", 0.8, 3.0),
        "min_after_open": ("int", 5, 120),
        "stop_atr": ("float", 0.8, 3.0),
        "max_hold": ("int", 6, 60),
        "latest_entry_min": ("int", 120, 360),
    }

    def on_bar(self, ctx: Context, s: str, df: pd.DataFrame) -> list[Signal]:
        if ctx.session != "regular" or ctx.has_lot(s):
            return []
        vwap = ctx.series(s, "svwap", ind.session_vwap)
        atr = ctx.series(s, "atr14", lambda d: ind.atr(d, 14))
        if len(vwap) < 2:
            return []
        v, av = safe_float(vwap.iloc[-1]), safe_float(atr.iloc[-1])
        if v is None or av is None or av <= 0:
            return []
        ny = df.index.tz_convert(NY)
        today = ny[-1].date()
        bars_today = int((ny.date == today).sum())
        tf_min = 5 if self.tf() == "5m" else 15
        mins = bars_today * tf_min
        if mins < self.p["min_after_open"] or mins > self.p["latest_entry_min"]:
            return []
        price = float(df["c"].iloc[-1])
        z = (price - v) / av
        vr = safe_float(ind.volume_ratio(df["v"], 20).iloc[-1])
        if vr is None or vr < self.p["vol_mult"]:
            return []
        if z < -self.p["z"]:
            return [Signal(s, "long", stop=atr_stop(price, av, self.p["stop_atr"], "long"), target=v, max_hold_bars=self.p["max_hold"], flat_at_session_end=True,
                           strength=min(1.0, abs(z) / 4.0), reason=f"{z:.1f} ATR below VWAP on vol x{vr:.1f}", extra={"vwap_z": z})]
        if z > self.p["z"] and ctx.allow_short:
            return [Signal(s, "short", stop=atr_stop(price, av, self.p["stop_atr"], "short"), target=v, max_hold_bars=self.p["max_hold"], flat_at_session_end=True,
                           strength=min(1.0, abs(z) / 4.0), reason=f"{z:.1f} ATR above VWAP on vol x{vr:.1f}", extra={"vwap_z": z})]
        return []
