"""Daily families: cross-sectional momentum rotation and the overnight-premium hold (equities)."""
from __future__ import annotations

import pandas as pd

from sentinel.data import indicators as ind
from sentinel.strategies.base import Context, Signal, Strategy, atr_stop, register, safe_float


@register
class XSMomentum(Strategy):
    family = "xs_momentum"
    description = "Each morning rank the equity universe by medium-term return (skipping the last few days) and hold the top names; rotate out when they drop from the top set."
    timeframe = "1d"
    markets = ["equities"]
    trigger = ["09:35"]
    shortable = False
    warmup_bars = 70
    DEFAULTS = {"lookback": 60, "skip": 5, "n_long": 4, "stop_atr": 3.0, "abs_filter": 1}
    PARAM_SPACE = {
        "lookback": ("int", 20, 160),
        "skip": ("int", 0, 10),
        "n_long": ("int", 2, 8),
        "stop_atr": ("float", 1.5, 6.0),
        "abs_filter": ("choice", [0, 1]),
    }

    def _ranking(self, ctx: Context) -> list[tuple[str, float]]:
        scores = []
        lb, skip = self.p["lookback"], self.p["skip"]
        for s in ctx.symbols:
            d = ctx.daily(s, lb + skip + 5)
            if len(d) < lb + skip + 1:
                continue
            c = d["c"]
            ref = float(c.iloc[-1 - skip]) if skip else float(c.iloc[-1])
            base = float(c.iloc[-1 - skip - lb])
            if base <= 0:
                continue
            scores.append((s, ref / base - 1.0))
        scores.sort(key=lambda x: -x[1])
        return scores

    def _top_set(self, ctx: Context) -> tuple[list[tuple[str, float]], list[str]]:
        if getattr(self, "_top_ts", None) != ctx.ts:
            ranking = self._ranking(ctx)
            self._top_ranking = ranking
            self._top = [s for s, sc in ranking[: self.p["n_long"]] if (sc > 0 or not self.p["abs_filter"])]
            self._top_ts = ctx.ts
        return self._top_ranking, self._top

    def on_cycle(self, ctx: Context) -> list[Signal]:
        if ctx.session != "regular":
            return []
        ranking, top = self._top_set(ctx)
        out: list[Signal] = []
        for s in top:
            if ctx.has_lot(s):
                continue
            d = ctx.daily(s, 30)
            price = ctx.price(s)
            av = safe_float(ind.atr(d, 14).iloc[-1]) if len(d) >= 15 else None
            if not price or not av:
                continue
            sc = dict(ranking)[s]
            out.append(Signal(s, "long", stop=atr_stop(price, av, self.p["stop_atr"], "long"), strength=min(1.0, sc * 2 + 0.3),
                              reason=f"top-{self.p['n_long']} {self.p['lookback']}d momentum ({sc * 100:+.1f}%)", extra={"xs_rank_score": sc}))
        return out

    def manage(self, ctx: Context, lot: dict, df: pd.DataFrame) -> str | None:
        if ctx.session != "regular":
            return None
        _, top = self._top_set(ctx)
        return None if lot["symbol"] in top else "rebalance"


@register
class Overnight(Strategy):
    family = "overnight"
    description = "Buy strong names shortly before the close and sell shortly after the next open to harvest the overnight premium."
    timeframe = "1d"
    markets = ["equities"]
    trigger = ["15:50"]
    shortable = False
    regular_session_only = True
    warmup_bars = 60
    DEFAULTS = {"n_symbols": 4, "trend_sma": 50, "max_day_move_pct": 3.0, "stop_atr": 2.5, "exit_after_open_min": 5, "rank_lookback": 20}
    PARAM_SPACE = {
        "n_symbols": ("int", 1, 8),
        "trend_sma": ("int", 20, 200),
        "max_day_move_pct": ("float", 1.0, 6.0),
        "stop_atr": ("float", 1.0, 5.0),
        "exit_after_open_min": ("int", 1, 60),
        "rank_lookback": ("int", 5, 60),
    }

    def on_cycle(self, ctx: Context) -> list[Signal]:
        if ctx.session != "regular":
            return []
        cands = []
        for s in ctx.symbols:
            if ctx.has_lot(s):
                continue
            d = ctx.daily(s, max(self.p["trend_sma"], self.p["rank_lookback"]) + 5)
            price = ctx.price(s)
            if price is None or len(d) < max(self.p["trend_sma"], self.p["rank_lookback"]) + 1:
                continue
            sma = safe_float(ind.sma(d["c"], self.p["trend_sma"]).iloc[-1])
            if not sma or price < sma:
                continue
            prev_close = float(d["c"].iloc[-1])
            day_move = (price / prev_close - 1.0) * 100
            if abs(day_move) > self.p["max_day_move_pct"]:
                continue
            mom = price / float(d["c"].iloc[-self.p["rank_lookback"]]) - 1.0
            av = safe_float(ind.atr(d, 14).iloc[-1])
            if av:
                cands.append((mom, s, price, av, day_move))
        cands.sort(reverse=True)
        out = []
        for mom, s, price, av, day_move in cands[: self.p["n_symbols"]]:
            out.append(Signal(s, "long", stop=atr_stop(price, av, self.p["stop_atr"], "long"), max_hold_bars=None, strength=min(1.0, mom * 3 + 0.3),
                              reason=f"overnight hold, {self.p['rank_lookback']}d mom {mom * 100:+.1f}%, day {day_move:+.1f}%",
                              extra={"overnight_exit_min": float(self.p["exit_after_open_min"]), "day_move_pct": day_move}))
        return out
