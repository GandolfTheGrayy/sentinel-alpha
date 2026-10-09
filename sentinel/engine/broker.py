"""Broker adapters: a simulated broker (backtests, --sim) and Alpaca (paper/live)."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sentinel.config import CostsCfg
from sentinel.data.market import is_crypto
from sentinel.util.clock import UTC, is_regular_session, next_open_close


@dataclass
class Fill:
    symbol: str
    side: str          # buy | sell
    qty: float
    price: float
    fees: float
    ts: datetime
    order_id: str


class Broker(Protocol):
    name: str

    def account(self) -> dict: ...
    def positions(self) -> dict[str, float]: ...
    def clock(self) -> dict: ...
    def market_order(self, symbol: str, side: str, qty: float, *, ref_price: float | None = None, extended_hours: bool = False, client_id: str | None = None) -> Fill | None: ...


class SimBroker:
    """Fills instantly at ref_price with adverse slippage and per-side fees."""

    name = "sim"

    def __init__(self, starting_cash: float, costs: CostsCfg, clock=None):
        self.cash = float(starting_cash)
        self.starting_cash = float(starting_cash)
        self.costs = costs
        self.pos: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self._clock = clock or (lambda: datetime.now(UTC))

    def set_marks(self, prices: dict[str, float]) -> None:
        self.marks.update({k: v for k, v in prices.items() if v})

    def account(self) -> dict:
        pos_value = sum(q * self.marks.get(s, 0.0) for s, q in self.pos.items())
        equity = self.cash + pos_value
        return {"equity": equity, "cash": self.cash, "buying_power": max(0.0, self.cash) * 2, "last_equity": equity}

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self.pos.items() if abs(q) > 1e-12}

    def clock(self) -> dict:
        now = self._clock()
        o, c = next_open_close(now)
        return {"is_open": is_regular_session(now), "next_open": o, "next_close": c}

    def market_order(self, symbol: str, side: str, qty: float, *, ref_price: float | None = None, extended_hours: bool = False, client_id: str | None = None) -> Fill | None:
        px = ref_price if ref_price is not None else self.marks.get(symbol)
        if not px or qty <= 0:
            return None
        crypto = is_crypto(symbol)
        slip = (self.costs.crypto_slippage_bps if crypto else self.costs.equity_slippage_bps) / 1e4
        fee_bps = (self.costs.crypto_fee_bps if crypto else self.costs.equity_fee_bps) / 1e4
        fill_px = px * (1 + slip) if side == "buy" else px * (1 - slip)
        notional = fill_px * qty
        fees = notional * fee_bps
        if side == "buy":
            self.cash -= notional + fees
            self.pos[symbol] = self.pos.get(symbol, 0.0) + qty
        else:
            self.cash += notional - fees
            self.pos[symbol] = self.pos.get(symbol, 0.0) - qty
        if abs(self.pos[symbol]) < 1e-12:
            self.pos.pop(symbol, None)
        self.marks[symbol] = px
        return Fill(symbol, side, qty, fill_px, fees, self._clock(), client_id or uuid.uuid4().hex[:12])


class AlpacaBroker:
    name = "alpaca"

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        from alpaca.trading.client import TradingClient

        self.client = TradingClient(api_key, secret_key, paper=paper)
        self.paper = paper
        self._crypto_map: dict[str, str] = {}
        self._clock_cache: tuple[float, dict] | None = None

    def register_crypto(self, symbols: list[str]) -> None:
        for s in symbols:
            self._crypto_map[s.replace("/", "")] = s

    def account(self) -> dict:
        a = self.client.get_account()
        return {"equity": float(a.equity or 0), "cash": float(a.cash or 0), "buying_power": float(a.buying_power or 0), "last_equity": float(a.last_equity or 0)}

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.client.get_all_positions():
            sym = self._crypto_map.get(p.symbol, p.symbol)
            qty = float(p.qty)
            out[sym] = qty if str(p.side).lower().endswith("long") else -abs(qty)
        return out

    def clock(self) -> dict:
        if self._clock_cache and time.time() - self._clock_cache[0] < 30:
            return self._clock_cache[1]
        c = self.client.get_clock()
        out = {"is_open": bool(c.is_open), "next_open": c.next_open, "next_close": c.next_close}
        self._clock_cache = (time.time(), out)
        return out

    def market_order(self, symbol: str, side: str, qty: float, *, ref_price: float | None = None, extended_hours: bool = False, client_id: str | None = None) -> Fill | None:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        crypto = is_crypto(symbol)
        if not crypto and not extended_hours:
            # the broker's calendar is authoritative: never queue an equity market order into the next session
            try:
                if not self.clock()["is_open"]:
                    return None
            except Exception:  # noqa: BLE001
                pass
        req = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6) if crypto else int(qty),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC if crypto else TimeInForce.DAY,
            extended_hours=bool(extended_hours and not crypto),
            client_order_id=client_id or f"sentinel-{uuid.uuid4().hex[:16]}",
        )
        order = self.client.submit_order(req)
        deadline = time.time() + 30
        while time.time() < deadline:
            o = self.client.get_order_by_id(order.id)
            status = str(o.status).lower()
            if "filled" in status and o.filled_avg_price and float(o.filled_qty or 0) > 0 and "partially" not in status:
                return Fill(symbol, side, float(o.filled_qty), float(o.filled_avg_price), 0.0, datetime.now(UTC), str(o.id))
            if any(k in status for k in ("canceled", "cancelled", "rejected", "expired")):
                return None
            time.sleep(0.5)
        try:
            self.client.cancel_order_by_id(order.id)
        except Exception:  # noqa: BLE001
            pass
        o = self.client.get_order_by_id(order.id)
        if o.filled_avg_price and float(o.filled_qty or 0) > 0:
            return Fill(symbol, side, float(o.filled_qty), float(o.filled_avg_price), 0.0, datetime.now(UTC), str(o.id))
        return None
