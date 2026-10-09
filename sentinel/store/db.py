"""Thin SQLite layer: one shared connection guarded by a lock, JSON helpers, DAOs.

The engine is single-threaded asyncio; FastAPI handlers run in a thread pool, so the
connection is created with check_same_thread=False and every call takes the lock.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return to_iso(o)
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def loads(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    # ---- primitives -------------------------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, [tuple(r) for r in rows])

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql + " LIMIT 1", params)
        return rows[0] if rows else None

    def transaction(self):
        return _Tx(self)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- kv ----------------------------------------------------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return loads(row["value"], default) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))

    # ---- events ------------------------------------------------------------------------
    def add_event(self, kind: str, message: str, level: str = "info", data: dict | None = None, ts: str | None = None) -> dict[str, Any]:
        ts = ts or now_iso()
        cur = self.execute("INSERT INTO events(ts,level,kind,message,data) VALUES(?,?,?,?,?)", (ts, level, kind, message, dumps(data or {})))
        return {"id": cur.lastrowid, "ts": ts, "level": level, "kind": kind, "message": message, "data": data or {}}

    def events(self, limit: int = 100, since: str | None = None) -> list[dict[str, Any]]:
        if since:
            rows = self.query("SELECT * FROM events WHERE ts>? ORDER BY id DESC LIMIT ?", (since, limit))
        else:
            rows = self.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        for r in rows:
            r["data"] = loads(r["data"], {})
        return rows

    # ---- bars --------------------------------------------------------------------------
    def upsert_bars(self, symbol: str, tf: str, rows: Iterable[tuple[int, float, float, float, float, float, float | None]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        self.executemany(
            "INSERT INTO bars(symbol,tf,ts,o,h,l,c,v,vwap) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(symbol,tf,ts) DO UPDATE SET o=excluded.o,h=excluded.h,l=excluded.l,c=excluded.c,v=excluded.v,vwap=excluded.vwap",
            [(symbol, tf, *r) for r in rows],
        )
        return len(rows)

    def bars(self, symbol: str, tf: str, start_ts: int | None = None, end_ts: int | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        sql = "SELECT ts,o,h,l,c,v,vwap FROM bars WHERE symbol=? AND tf=?"
        params: list[Any] = [symbol, tf]
        if start_ts is not None:
            sql += " AND ts>=?"
            params.append(int(start_ts))
        if end_ts is not None:
            sql += " AND ts<?"
            params.append(int(end_ts))
        if limit:
            sql += " ORDER BY ts DESC LIMIT ?"
            params.append(int(limit))
            rows = self.query(sql, params)
            rows.reverse()
            return rows
        sql += " ORDER BY ts"
        return self.query(sql, params)

    def last_bar_ts(self, symbol: str, tf: str) -> int | None:
        row = self.one("SELECT MAX(ts) AS ts FROM bars WHERE symbol=? AND tf=?", (symbol, tf))
        return int(row["ts"]) if row and row["ts"] is not None else None

    # ---- variants ----------------------------------------------------------------------
    def variants(self, status: str | list[str] | None = None, include_retired: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM variants"
        params: list[Any] = []
        clauses = []
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            clauses.append("status IN (%s)" % ",".join("?" * len(statuses)))
            params += statuses
        elif not include_retired:
            clauses.append("status != 'retired'")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at"
        return [self._variant_row(r) for r in self.query(sql, params)]

    def variant(self, vid: str) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM variants WHERE id=?", (vid,))
        return self._variant_row(row) if row else None

    @staticmethod
    def _variant_row(r: dict[str, Any]) -> dict[str, Any]:
        r = dict(r)
        r["params"] = loads(r["params"], {})
        r["markets"] = loads(r["markets"], [])
        r["is_control"] = bool(r["is_control"])
        return r

    def insert_variant(self, v: dict[str, Any]) -> None:
        ts = now_iso()
        self.execute(
            "INSERT INTO variants(id,family,name,params,status,status_reason,origin,parent_id,markets,timeframe,is_control,allocation,notes,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (v["id"], v["family"], v["name"], dumps(v["params"]), v.get("status", "incubating"), v.get("status_reason"), v.get("origin", "seed"),
             v.get("parent_id"), dumps(v.get("markets", [])), v["timeframe"], 1 if v.get("is_control") else 0, float(v.get("allocation", 0.0)),
             v.get("notes"), v.get("created_at", ts), ts),
        )

    def update_variant(self, vid: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now_iso()
        cols = []
        vals: list[Any] = []
        for k, val in fields.items():
            if k in ("params", "markets") and not isinstance(val, str):
                val = dumps(val)
            cols.append(f"{k}=?")
            vals.append(val)
        vals.append(vid)
        self.execute(f"UPDATE variants SET {', '.join(cols)} WHERE id=?", vals)

    def next_variant_id(self, family: str) -> str:
        rows = self.query("SELECT id FROM variants WHERE family=?", (family,))
        n = 0
        for r in rows:
            try:
                n = max(n, int(str(r["id"]).rsplit("#", 1)[1]))
            except (IndexError, ValueError):
                continue
        return f"{family}#{n + 1}"

    # ---- lots / trades -----------------------------------------------------------------
    def open_lots(self, variant_id: str | None = None, symbol: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM lots WHERE status='open'"
        params: list[Any] = []
        if variant_id:
            sql += " AND variant_id=?"
            params.append(variant_id)
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        sql += " ORDER BY entry_ts"
        rows = self.query(sql, params)
        for r in rows:
            r["features"] = loads(r["features"], {})
            r["signal"] = loads(r["signal"], {})
        return rows

    def insert_lot(self, lot: dict[str, Any]) -> int:
        cur = self.execute(
            "INSERT INTO lots(variant_id,family,symbol,side,qty,entry_ts,entry_price,stop,target,max_hold_ts,flat_at_session_end,trail_atr,risk_per_unit,reason,features,signal,broker_order_id,hwm,lwm,fees,status) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open')",
            (lot["variant_id"], lot["family"], lot["symbol"], lot["side"], lot["qty"], lot["entry_ts"], lot["entry_price"], lot.get("stop"), lot.get("target"),
             lot.get("max_hold_ts"), 1 if lot.get("flat_at_session_end") else 0, lot.get("trail_atr"), lot["risk_per_unit"], lot.get("reason"),
             dumps(lot.get("features", {})), dumps(lot.get("signal", {})), lot.get("broker_order_id"), lot["entry_price"], lot["entry_price"], lot.get("fees", 0.0)),
        )
        return int(cur.lastrowid)

    def update_lot(self, lot_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE lots SET {cols} WHERE id=?", [*fields.values(), lot_id])

    def close_lot(self, lot: dict[str, Any], exit_ts: str, exit_price: float, exit_reason: str, fees: float = 0.0, is_backtest: bool = False) -> dict[str, Any]:
        side = 1.0 if lot["side"] == "long" else -1.0
        qty = float(lot["qty"])
        gross = (exit_price - float(lot["entry_price"])) * side * qty
        total_fees = float(lot.get("fees", 0.0)) + fees
        pnl = gross - total_fees
        rpu = float(lot["risk_per_unit"]) or 1e-9
        pnl_r = (exit_price - float(lot["entry_price"])) * side / rpu
        hwm = float(lot.get("hwm") or lot["entry_price"])
        lwm = float(lot.get("lwm") or lot["entry_price"])
        if lot["side"] == "long":
            mfe_r = (hwm - float(lot["entry_price"])) / rpu
            mae_r = (lwm - float(lot["entry_price"])) / rpu
        else:
            mfe_r = (float(lot["entry_price"]) - lwm) / rpu
            mae_r = (float(lot["entry_price"]) - hwm) / rpu
        hold_minutes = max(0.0, (from_iso(exit_ts) - from_iso(lot["entry_ts"])).total_seconds() / 60.0)
        trade = {
            "lot_id": lot.get("id"), "variant_id": lot["variant_id"], "family": lot["family"], "symbol": lot["symbol"], "side": lot["side"], "qty": qty,
            "entry_ts": lot["entry_ts"], "entry_price": float(lot["entry_price"]), "exit_ts": exit_ts, "exit_price": float(exit_price),
            "pnl": round(pnl, 4), "pnl_r": round(pnl_r, 4), "fees": round(total_fees, 4), "hold_minutes": round(hold_minutes, 1), "exit_reason": exit_reason,
            "mae_r": round(min(mae_r, 0.0), 4), "mfe_r": round(max(mfe_r, 0.0), 4), "reason": lot.get("reason"),
            "features": lot.get("features") if isinstance(lot.get("features"), dict) else loads(lot.get("features"), {}),
        }
        if lot.get("id") is not None:
            self.execute("UPDATE lots SET status='closed' WHERE id=?", (lot["id"],))
        cur = self.execute(
            "INSERT INTO trades(lot_id,variant_id,family,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,pnl_r,fees,hold_minutes,exit_reason,mae_r,mfe_r,reason,features,is_backtest) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade["lot_id"], trade["variant_id"], trade["family"], trade["symbol"], trade["side"], trade["qty"], trade["entry_ts"], trade["entry_price"], trade["exit_ts"],
             trade["exit_price"], trade["pnl"], trade["pnl_r"], trade["fees"], trade["hold_minutes"], trade["exit_reason"], trade["mae_r"], trade["mfe_r"], trade["reason"],
             dumps(trade["features"]), 1 if is_backtest else 0),
        )
        trade["id"] = int(cur.lastrowid)
        return trade

    def trades(self, limit: int = 200, variant_id: str | None = None, symbol: str | None = None, since: str | None = None, family: str | None = None, with_features: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM trades WHERE is_backtest=0"
        params: list[Any] = []
        if variant_id:
            sql += " AND variant_id=?"
            params.append(variant_id)
        if family:
            sql += " AND family=?"
            params.append(family)
        if symbol:
            sql += " AND symbol=?"
            params.append(symbol)
        if since:
            sql += " AND exit_ts>=?"
            params.append(since)
        sql += " ORDER BY exit_ts DESC, id DESC LIMIT ?"
        params.append(int(limit))
        rows = self.query(sql, params)
        for r in rows:
            r["features"] = loads(r["features"], {}) if with_features else {}
            r.pop("is_backtest", None)
        return rows

    def trade_count(self, variant_id: str | None = None, since: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM trades WHERE is_backtest=0"
        params: list[Any] = []
        if variant_id:
            sql += " AND variant_id=?"
            params.append(variant_id)
        if since:
            sql += " AND exit_ts>=?"
            params.append(since)
        row = self.one(sql, params)
        return int(row["n"]) if row else 0

    # ---- equity snapshots --------------------------------------------------------------
    def add_equity_snapshot(self, ts: str, equity: float, cash: float, benchmark: float | None, per_variant: dict[str, float]) -> None:
        self.execute(
            "INSERT INTO equity_snapshots(ts,equity,cash,benchmark,per_variant) VALUES(?,?,?,?,?) ON CONFLICT(ts) DO UPDATE SET equity=excluded.equity,cash=excluded.cash,benchmark=excluded.benchmark,per_variant=excluded.per_variant",
            (ts, equity, cash, benchmark, dumps(per_variant)),
        )

    def equity_curve(self, since: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
        if since:
            rows = self.query("SELECT ts,equity,cash,benchmark,per_variant FROM equity_snapshots WHERE ts>=? ORDER BY ts DESC LIMIT ?", (since, limit))
        else:
            rows = self.query("SELECT ts,equity,cash,benchmark,per_variant FROM equity_snapshots ORDER BY ts DESC LIMIT ?", (limit,))
        rows.reverse()
        return rows

    # ---- claude runs / proposals -------------------------------------------------------
    def insert_run(self, kind: str, model: str, digest_md: str) -> int:
        cur = self.execute("INSERT INTO claude_runs(kind,model,started_at,status,digest_md) VALUES(?,?,?,'running',?)", (kind, model, now_iso(), digest_md))
        return int(cur.lastrowid)

    def update_run(self, run_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE claude_runs SET {cols} WHERE id=?", [*fields.values(), run_id])

    def runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT r.*, (SELECT COUNT(*) FROM proposals p WHERE p.run_id=r.id) AS n_proposals, "
            "(SELECT COUNT(*) FROM proposals p WHERE p.run_id=r.id AND p.status='accepted') AS n_accepted "
            "FROM claude_runs r ORDER BY id DESC LIMIT ?", (limit,))
        return rows

    def run(self, run_id: int) -> dict[str, Any] | None:
        row = self.one("SELECT * FROM claude_runs WHERE id=?", (run_id,))
        if not row:
            return None
        row["proposals"] = self.proposals(run_id=run_id)
        return row

    def weekly_claude_spend(self, week_start_iso: str) -> tuple[float, int]:
        row = self.one("SELECT COALESCE(SUM(cost_usd),0) AS c, COUNT(*) AS n FROM claude_runs WHERE started_at>=? AND status IN ('ok','error','running')", (week_start_iso,))
        return float(row["c"]), int(row["n"])

    def insert_proposal(self, run_id: int | None, ptype: str, target: str | None, payload: dict, rationale: str) -> int:
        cur = self.execute("INSERT INTO proposals(run_id,type,target,payload,rationale,status,created_at) VALUES(?,?,?,?,?,'pending',?)",
                           (run_id, ptype, target, dumps(payload), rationale, now_iso()))
        return int(cur.lastrowid)

    def update_proposal(self, pid: int, **fields: Any) -> None:
        if "backtest" in fields and not isinstance(fields["backtest"], str):
            fields["backtest"] = dumps(fields["backtest"])
        if "payload" in fields and not isinstance(fields["payload"], str):
            fields["payload"] = dumps(fields["payload"])
        cols = ", ".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE proposals SET {cols} WHERE id=?", [*fields.values(), pid])

    def proposals(self, status: str | None = None, run_id: int | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT * FROM proposals"
        params: list[Any] = []
        clauses = []
        if status:
            clauses.append("status=?")
            params.append(status)
        if run_id is not None:
            clauses.append("run_id=?")
            params.append(run_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.query(sql, params)
        for r in rows:
            r["payload"] = loads(r["payload"], {})
            r["backtest"] = loads(r["backtest"], None)
        return rows

    def proposal(self, pid: int) -> dict[str, Any] | None:
        rows = self.query("SELECT * FROM proposals WHERE id=?", (pid,))
        if not rows:
            return None
        r = rows[0]
        r["payload"] = loads(r["payload"], {})
        r["backtest"] = loads(r["backtest"], None)
        return r

    # ---- misc --------------------------------------------------------------------------
    def insert_backtest(self, family: str, variant_id: str | None, params: dict, days: int, result: dict) -> int:
        cur = self.execute("INSERT INTO backtests(ts,family,variant_id,params,days,result) VALUES(?,?,?,?,?,?)", (now_iso(), family, variant_id, dumps(params), days, dumps(result)))
        return int(cur.lastrowid)

    def backtests(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.query("SELECT id,ts,family,variant_id,params,days,result FROM backtests ORDER BY id DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            res = loads(r["result"], {})
            res.pop("trades", None)
            res.pop("equity_curve", None)
            out.append({"id": r["id"], "ts": r["ts"], "family": r["family"], "variant_id": r["variant_id"], "params": loads(r["params"], {}), "days": r["days"], **res})
        return out

    def save_attribution(self, report: dict, digest_md: str) -> int:
        cur = self.execute("INSERT INTO attribution_reports(ts,report,digest_md) VALUES(?,?,?)", (now_iso(), dumps(report), digest_md))
        return int(cur.lastrowid)

    def latest_attribution(self) -> dict | None:
        row = self.one("SELECT report FROM attribution_reports ORDER BY id DESC")
        return loads(row["report"], None) if row else None

    def latest_attribution_digest(self) -> str:
        row = self.one("SELECT digest_md FROM attribution_reports ORDER BY id DESC")
        return (row["digest_md"] or "") if row else ""


class _Tx:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db._lock.acquire()
        self.db._conn.execute("BEGIN")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.db._conn.execute("COMMIT")
            else:
                self.db._conn.execute("ROLLBACK")
        finally:
            self.db._lock.release()
        return False
