"""Generic exit rules shared by the live engine and the backtester (pure functions)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bar:
    ts: int
    o: float
    h: float
    l: float
    c: float


def update_marks(lot: dict, bar: Bar) -> None:
    """Update high/low water marks and ratchet a trailing stop. Mutates lot in place."""
    lot["hwm"] = max(float(lot.get("hwm") or lot["entry_price"]), bar.h)
    lot["lwm"] = min(float(lot.get("lwm") or lot["entry_price"]), bar.l)
    trail = lot.get("trail_atr")
    if trail:
        if lot["side"] == "long":
            new_stop = lot["hwm"] - float(trail)
            if lot.get("stop") is None or new_stop > float(lot["stop"]):
                lot["stop"] = new_stop
        else:
            new_stop = lot["lwm"] + float(trail)
            if lot.get("stop") is None or new_stop < float(lot["stop"]):
                lot["stop"] = new_stop


def check_exit(lot: dict, bar: Bar, now_ts: int, *, intrabar: bool, session_last_minute: bool, max_hold_ts: int | None = None) -> tuple[str, float] | None:
    """Return (reason, exit_price) if the lot should be closed on this bar.

    intrabar=True (backtests) uses the bar's high/low to detect stop/target touches and
    assumes the stop is hit first when both are touched. intrabar=False (live) only uses
    the close/last price and exits at that price.
    """
    stop = lot.get("stop")
    target = lot.get("target")
    long = lot["side"] == "long"
    if intrabar:
        if stop is not None:
            stop = float(stop)
            if (long and bar.l <= stop) or (not long and bar.h >= stop):
                # gapped through the stop: fill at the open
                px = min(bar.o, stop) if long else max(bar.o, stop)
                return "stop", px
        if target is not None:
            target = float(target)
            if (long and bar.h >= target) or (not long and bar.l <= target):
                px = max(bar.o, target) if long else min(bar.o, target)
                return "target", px
    else:
        px = bar.c
        if stop is not None and ((long and px <= float(stop)) or (not long and px >= float(stop))):
            return "stop", px
        if target is not None and ((long and px >= float(target)) or (not long and px <= float(target))):
            return "target", px
    mh = max_hold_ts if max_hold_ts is not None else lot.get("max_hold_ts_epoch")
    if mh is not None and now_ts >= int(mh):
        return "time", bar.c
    if session_last_minute and lot.get("flat_at_session_end"):
        return "session_end", bar.c
    return None
