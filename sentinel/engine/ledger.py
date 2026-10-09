"""Per-variant lot ledger. In-memory, optionally persisted to SQLite (live mode)."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from sentinel.store.db import Database, from_iso, to_iso
from sentinel.util.clock import UTC


class Ledger:
    def __init__(self, db: Database | None, persist: bool):
        self.db = db
        self.persist = persist and db is not None
        self.lots: dict[int, dict[str, Any]] = {}
        self.trades: list[dict[str, Any]] = []
        self._next_id = 1
        self.variant_pnl: dict[str, float] = defaultdict(float)
        self.variant_peak: dict[str, float] = defaultdict(float)
        self.variant_trades: dict[str, int] = defaultdict(int)
        if self.persist:
            for lot in db.open_lots():
                self._attach_epoch(lot)
                self.lots[int(lot["id"])] = lot
            for row in db.query("SELECT variant_id, SUM(pnl) AS pnl, COUNT(*) AS n FROM trades WHERE is_backtest=0 GROUP BY variant_id"):
                self.variant_pnl[row["variant_id"]] = float(row["pnl"] or 0.0)
                self.variant_trades[row["variant_id"]] = int(row["n"])
            # rebuild peaks from the trade history (cheap enough at startup)
            run: dict[str, float] = defaultdict(float)
            for t in db.query("SELECT variant_id, pnl FROM trades WHERE is_backtest=0 ORDER BY exit_ts, id"):
                run[t["variant_id"]] += float(t["pnl"])
                self.variant_peak[t["variant_id"]] = max(self.variant_peak[t["variant_id"]], run[t["variant_id"]])

    @staticmethod
    def _attach_epoch(lot: dict[str, Any]) -> None:
        lot["entry_ts_epoch"] = int(from_iso(lot["entry_ts"]).timestamp())
        lot["max_hold_ts_epoch"] = int(from_iso(lot["max_hold_ts"]).timestamp()) if lot.get("max_hold_ts") else None

    # ---- queries ------------------------------------------------------------------------
    def open_lots(self, variant_id: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]:
        out = list(self.lots.values())
        if variant_id:
            out = [l for l in out if l["variant_id"] == variant_id]
        if symbol:
            out = [l for l in out if l["symbol"] == symbol]
        return out

    def side_held(self, symbol: str) -> str | None:
        for l in self.lots.values():
            if l["symbol"] == symbol:
                return l["side"]
        return None

    def gross_exposure(self, prices: dict[str, float]) -> float:
        total = 0.0
        for l in self.lots.values():
            px = prices.get(l["symbol"]) or float(l["entry_price"])
            total += abs(float(l["qty"]) * px)
        return total

    def unrealized(self, prices: dict[str, float]) -> float:
        total = 0.0
        for l in self.lots.values():
            px = prices.get(l["symbol"])
            if px is None:
                continue
            sign = 1.0 if l["side"] == "long" else -1.0
            total += (px - float(l["entry_price"])) * sign * float(l["qty"])
        return total

    def variant_drawdown_pct(self, variant_id: str, sleeve_equity: float) -> float:
        if sleeve_equity <= 0:
            return 0.0
        peak = self.variant_peak.get(variant_id, 0.0)
        cur = self.variant_pnl.get(variant_id, 0.0)
        return max(0.0, (peak - cur) / sleeve_equity * 100.0)

    # ---- mutations ----------------------------------------------------------------------
    def open(self, lot: dict[str, Any]) -> dict[str, Any]:
        lot = dict(lot)
        lot.setdefault("hwm", lot["entry_price"])
        lot.setdefault("lwm", lot["entry_price"])
        lot.setdefault("fees", 0.0)
        lot["status"] = "open"
        if self.persist:
            lot["id"] = self.db.insert_lot(lot)
        else:
            lot["id"] = self._next_id
            self._next_id += 1
        self._attach_epoch(lot)
        self.lots[lot["id"]] = lot
        return lot

    def sync_marks(self, lot: dict[str, Any]) -> None:
        if self.persist:
            self.db.update_lot(int(lot["id"]), hwm=lot.get("hwm"), lwm=lot.get("lwm"), stop=lot.get("stop"))

    def close(self, lot: dict[str, Any], exit_dt: datetime | int, exit_price: float, reason: str, fees: float = 0.0, is_backtest: bool = False) -> dict[str, Any]:
        exit_ts = to_iso(exit_dt) if isinstance(exit_dt, datetime) else to_iso(datetime.fromtimestamp(exit_dt, UTC))
        if self.persist:
            trade = self.db.close_lot(lot, exit_ts, exit_price, reason, fees=fees, is_backtest=is_backtest)
        else:
            trade = _close_lot_memory(lot, exit_ts, exit_price, reason, fees)
        self.lots.pop(int(lot["id"]), None)
        self.trades.append(trade)
        vid = lot["variant_id"]
        self.variant_pnl[vid] += float(trade["pnl"])
        self.variant_trades[vid] += 1
        self.variant_peak[vid] = max(self.variant_peak[vid], self.variant_pnl[vid])
        return trade


def _close_lot_memory(lot: dict[str, Any], exit_ts: str, exit_price: float, reason: str, fees: float) -> dict[str, Any]:
    side = 1.0 if lot["side"] == "long" else -1.0
    qty = float(lot["qty"])
    entry = float(lot["entry_price"])
    total_fees = float(lot.get("fees", 0.0)) + fees
    pnl = (exit_price - entry) * side * qty - total_fees
    rpu = float(lot["risk_per_unit"]) or 1e-9
    pnl_r = (exit_price - entry) * side / rpu
    hwm = float(lot.get("hwm") or entry)
    lwm = float(lot.get("lwm") or entry)
    if lot["side"] == "long":
        mfe_r, mae_r = (hwm - entry) / rpu, (lwm - entry) / rpu
    else:
        mfe_r, mae_r = (entry - lwm) / rpu, (entry - hwm) / rpu
    hold_minutes = max(0.0, (from_iso(exit_ts) - from_iso(lot["entry_ts"])).total_seconds() / 60.0)
    return {
        "id": None, "lot_id": lot.get("id"), "variant_id": lot["variant_id"], "family": lot["family"], "symbol": lot["symbol"], "side": lot["side"], "qty": qty,
        "entry_ts": lot["entry_ts"], "entry_price": entry, "exit_ts": exit_ts, "exit_price": float(exit_price),
        "pnl": round(pnl, 4), "pnl_r": round(pnl_r, 4), "fees": round(total_fees, 4), "hold_minutes": round(hold_minutes, 1), "exit_reason": reason,
        "mae_r": round(min(mae_r, 0.0), 4), "mfe_r": round(max(mfe_r, 0.0), 4), "reason": lot.get("reason"), "features": lot.get("features") or {},
    }
