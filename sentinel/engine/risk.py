"""Position sizing and exposure limits."""
from __future__ import annotations

import math
from dataclasses import dataclass

from sentinel.config import RiskCfg
from sentinel.data.market import is_crypto


@dataclass
class SizeResult:
    qty: float
    notional: float
    reason: str | None = None   # populated when qty == 0


def size_order(price: float, stop: float, sleeve_equity: float, total_equity: float, gross_exposure: float, symbol: str, cfg: RiskCfg) -> SizeResult:
    """Risk-based sizing: risk `risk_per_trade_pct` of the sleeve between entry and stop, then cap."""
    if price <= 0 or sleeve_equity <= 0:
        return SizeResult(0.0, 0.0, "no sleeve equity")
    risk_per_unit = abs(price - stop)
    if risk_per_unit <= 0:
        return SizeResult(0.0, 0.0, "zero risk per unit")
    risk_dollars = sleeve_equity * cfg.risk_per_trade_pct / 100.0
    qty = risk_dollars / risk_per_unit
    cap_notional = min(sleeve_equity * cfg.max_position_pct_of_sleeve / 100.0, total_equity * cfg.max_position_pct_of_equity / 100.0)
    room = total_equity * cfg.max_gross_exposure_pct / 100.0 - gross_exposure
    if room <= 0:
        return SizeResult(0.0, 0.0, "gross exposure limit")
    cap_notional = min(cap_notional, room)
    qty = min(qty, cap_notional / price)
    if is_crypto(symbol):
        qty = math.floor(qty * 1e6) / 1e6
        if qty * price < 10.0:
            return SizeResult(0.0, 0.0, "below $10 crypto minimum")
    else:
        qty = float(math.floor(qty))
        if qty < 1:
            return SizeResult(0.0, 0.0, "less than one share")
    return SizeResult(qty, qty * price)
