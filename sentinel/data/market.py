"""In-memory bar store with resampling, plus the MarketData facade used by the engine,
the backtester and the feature builder.

Base timeframes are '1m' and '1d' (persisted in SQLite). Any other timeframe is
resampled on demand and cached until new base bars arrive. Equity intraday buckets are
aligned to the 09:30 America/New_York session open; crypto buckets are aligned to UTC.
"""
from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone
from typing import Protocol

import numpy as np
import pandas as pd

from sentinel.store.db import Database
from sentinel.util.clock import NY, TF_SECONDS, UTC, floor_ts

COLS = ["o", "h", "l", "c", "v", "vwap"]
RESAMPLE_RULE = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def is_crypto(symbol: str) -> bool:
    return "/" in symbol


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)


_EMPTY = empty_frame()


def frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return empty_frame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="s", utc=True)
    df = df.set_index("ts").sort_index()
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS].astype(float)


def rows_from_frame(df: pd.DataFrame) -> list[tuple]:
    out = []
    for ts, r in zip(df.index, df.itertuples(index=False)):
        vwap = float(r.vwap) if r.vwap == r.vwap else None  # NaN check
        out.append((int(ts.timestamp()), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v), vwap))
    return out


def resample(df: pd.DataFrame, tf: str, crypto: bool) -> pd.DataFrame:
    """Resample 1-minute bars into tf. Drops empty buckets."""
    if df.empty or tf == "1m":
        return df
    rule = RESAMPLE_RULE[tf]
    agg = {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    kw: dict = {"label": "left", "closed": "left"}
    if tf != "1d":
        kw["origin"] = "epoch" if crypto else "start_day"
        if not crypto:
            kw["offset"] = "9h30min"
    src = df if crypto else df.tz_convert(NY)
    out = src.resample(rule, **kw).agg(agg)
    pv = (((df["h"] + df["l"] + df["c"]) / 3.0) * df["v"])
    pvs = (pv if crypto else pv.tz_convert(NY)).resample(rule, **kw).sum()
    out["vwap"] = pvs / out["v"].replace(0.0, np.nan)
    out = out.dropna(subset=["o", "c"])
    if not crypto:
        out = out.tz_convert(UTC)
    out.index.name = "ts"
    return out[COLS]


class BarProvider(Protocol):
    def fetch_bars(self, symbols: list[str], tf: str, start: datetime, end: datetime) -> dict[str, pd.DataFrame]: ...
    def latest_prices(self, symbols: list[str]) -> dict[str, float]: ...


class BarStore:
    def __init__(self) -> None:
        self._base: dict[tuple[str, str], pd.DataFrame] = {}
        self._cache: dict[tuple[str, str], tuple[int, pd.DataFrame]] = {}
        self._arr: dict[tuple[str, str], tuple[int, dict[str, np.ndarray]]] = {}
        self._last: dict[str, tuple[float, int]] = {}

    # ---- fast numpy paths (used every minute by the trader) -----------------------------
    def _arrays(self, symbol: str, tf: str) -> dict[str, np.ndarray] | None:
        base_len = len(self.base(symbol, "1m"))
        key = (symbol, tf)
        cached = self._arr.get(key)
        if cached is not None and cached[0] == base_len:
            return cached[1]
        df = self.frame(symbol, tf)
        if df.empty:
            return None
        arr = {"ts": df.index.values.astype("datetime64[s]").astype("int64"), "o": df["o"].values.astype(float), "h": df["h"].values.astype(float),
               "l": df["l"].values.astype(float), "c": df["c"].values.astype(float)}
        self._arr[key] = (base_len, arr)
        return arr

    def minute_bar(self, symbol: str, end_ts: int) -> tuple[int, float, float, float, float] | None:
        """Last fully closed 1-minute bar at end_ts: (ts, o, h, l, c)."""
        arr = self._arrays(symbol, "1m")
        if arr is None:
            return None
        i = int(np.searchsorted(arr["ts"], end_ts - 60, side="right")) - 1
        if i < 0:
            return None
        return int(arr["ts"][i]), float(arr["o"][i]), float(arr["h"][i]), float(arr["l"][i]), float(arr["c"][i])

    def last_closed_price(self, symbol: str, end_ts: int) -> float | None:
        bar = self.minute_bar(symbol, end_ts)
        if bar is not None:
            return bar[4]
        d = self.frame(symbol, "1d", end_ts=end_ts, n=1, include_partial=True)
        return float(d["c"].iloc[-1]) if not d.empty else None

    def last_closed_bucket(self, symbol: str, tf: str, end_ts: int) -> int | None:
        """Start timestamp of the last bucket of tf that has fully closed by end_ts."""
        if tf == "1d":
            df = self.frame(symbol, "1d", end_ts=end_ts, n=1)
            return int(df.index[-1].timestamp()) if not df.empty else None
        arr = self._arrays(symbol, tf)
        if arr is None:
            return None
        i = int(np.searchsorted(arr["ts"], end_ts - TF_SECONDS[tf], side="right")) - 1
        return int(arr["ts"][i]) if i >= 0 else None

    def base(self, symbol: str, tf: str) -> pd.DataFrame:
        df = self._base.get((symbol, tf))
        return df if df is not None else _EMPTY

    def set_base(self, symbol: str, tf: str, df: pd.DataFrame) -> None:
        self._base[(symbol, tf)] = df.sort_index()
        self._invalidate(symbol)

    def append(self, symbol: str, tf: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        cur = self._base.get((symbol, tf))
        if cur is None or cur.empty:
            merged = df.sort_index()
        else:
            merged = pd.concat([cur, df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        n_new = len(merged) - (0 if cur is None else len(cur))
        self._base[(symbol, tf)] = merged
        self._invalidate(symbol)
        if tf == "1m":
            last_ts = merged.index[-1]
            self._last[symbol] = (float(merged["c"].iloc[-1]), int(last_ts.timestamp()) + 60)
        return max(n_new, 0)

    def _invalidate(self, symbol: str) -> None:
        for key in [k for k in self._cache if k[0] == symbol]:
            self._cache.pop(key, None)
        for key in [k for k in self._arr if k[0] == symbol]:
            self._arr.pop(key, None)

    def set_last_price(self, symbol: str, price: float, ts: int) -> None:
        self._last[symbol] = (float(price), int(ts))

    def last_price(self, symbol: str) -> float | None:
        v = self._last.get(symbol)
        if v is None:
            b = self._base.get((symbol, "1m"))
            if b is not None and not b.empty:
                return float(b["c"].iloc[-1])
            d = self._base.get((symbol, "1d"))
            if d is not None and not d.empty:
                return float(d["c"].iloc[-1])
            return None
        return v[0]

    def last_price_ts(self, symbol: str) -> int | None:
        v = self._last.get(symbol)
        return v[1] if v else None

    def frame(self, symbol: str, tf: str, end_ts: int | None = None, n: int | None = None, include_partial: bool = False) -> pd.DataFrame:
        """Bars of `tf` for symbol, only buckets fully closed by end_ts (unless include_partial)."""
        crypto = is_crypto(symbol)
        if tf == "1d" and (symbol, "1d") in self._base and not self._base[(symbol, "1d")].empty:
            df = self._base[(symbol, "1d")]
        elif tf == "1m":
            df = self.base(symbol, "1m")
        else:
            base = self.base(symbol, "1m")
            key = (symbol, tf)
            cached = self._cache.get(key)
            if cached is not None and cached[0] == len(base):
                df = cached[1]
            else:
                df = resample(base, tf, crypto)
                self._cache[key] = (len(base), df)
        if df.empty:
            return df
        if end_ts is not None:
            end = pd.Timestamp(end_ts, unit="s", tz="UTC")
            if include_partial:
                cut = df.index.searchsorted(end, side="right")
            elif tf == "1d":
                # a daily bar is closed once its session date is before the end date (NY for equities, UTC for crypto)
                if crypto:
                    day_start = end.normalize()
                else:
                    day_start = end.tz_convert(NY).normalize().tz_convert(UTC)
                cut = df.index.searchsorted(day_start, side="left")
            else:
                cut = df.index.searchsorted(end - pd.Timedelta(seconds=TF_SECONDS[tf]), side="right")
            df = df.iloc[:cut]
        if n is not None and len(df) > n:
            df = df.iloc[-n:]
        return df

    def symbols(self) -> list[str]:
        return sorted({k[0] for k in self._base})


class MarketData:
    """Facade: provider -> DB -> BarStore. Knows the universe and asset classes."""

    def __init__(self, db: Database, provider: BarProvider, equities: list[str], crypto: list[str], feed: str = "iex"):
        self.db = db
        self.provider = provider
        self.equities = list(equities)
        self.crypto = list(crypto)
        self.feed = feed
        self.store = BarStore()
        self.last_poll_ts: int = 0
        self.poll_errors: int = 0

    @property
    def symbols(self) -> list[str]:
        return self.equities + self.crypto

    def asset_class(self, symbol: str) -> str:
        return "crypto" if is_crypto(symbol) else "equities"

    # ---- loading ----------------------------------------------------------------------
    def load_from_db(self, minute_days: int, daily_days: int, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        m_since = int((now - timedelta(days=minute_days)).timestamp())
        d_since = int((now - timedelta(days=daily_days)).timestamp())
        for s in self.symbols:
            self.store.set_base(s, "1m", frame_from_rows(self.db.bars(s, "1m", start_ts=m_since)))
            self.store.set_base(s, "1d", frame_from_rows(self.db.bars(s, "1d", start_ts=d_since)))

    def backfill(self, minute_days: int, daily_days: int, now: datetime | None = None, log=print) -> None:
        """Fetch any missing history from the provider into DB + store."""
        now = now or datetime.now(UTC)
        for tf, days in (("1d", daily_days), ("1m", minute_days)):
            for group in (self.equities, self.crypto):
                if not group:
                    continue
                # fetch from the oldest gap among the group's symbols
                starts = []
                for s in group:
                    last = self.db.last_bar_ts(s, tf)
                    default_start = now - timedelta(days=days)
                    if last is None:
                        starts.append(default_start)
                    else:
                        starts.append(max(default_start, datetime.fromtimestamp(last, UTC) - timedelta(seconds=TF_SECONDS[tf])))
                start = min(starts)
                if (now - start).total_seconds() < TF_SECONDS[tf]:
                    continue
                try:
                    data = self.provider.fetch_bars(group, tf, start, now)
                except Exception as exc:  # noqa: BLE001
                    self.poll_errors += 1
                    log(f"backfill {tf} failed for {group[:3]}...: {exc}")
                    continue
                for s, df in data.items():
                    if df is None or df.empty:
                        continue
                    self.db.upsert_bars(s, tf, rows_from_frame(df))
                    self.store.append(s, tf, df)
                log(f"backfill {tf}: {sum(len(d) for d in data.values())} bars for {len(data)} symbols from {start:%Y-%m-%d %H:%M}")

    def poll(self, now: datetime | None = None) -> int:
        """Fetch new minute bars since the last stored bar for every symbol. Returns bar count."""
        now = now or datetime.now(UTC)
        total = 0
        for group in (self.equities, self.crypto):
            if not group:
                continue
            last_ts = [self.store.base(s, "1m").index[-1].timestamp() if not self.store.base(s, "1m").empty else (now - timedelta(days=2)).timestamp() for s in group]
            start = datetime.fromtimestamp(min(last_ts), UTC) - timedelta(minutes=2)
            try:
                data = self.provider.fetch_bars(group, "1m", start, now)
            except Exception as exc:  # noqa: BLE001
                self.poll_errors += 1
                raise RuntimeError(f"poll failed: {exc}") from exc
            for s, df in data.items():
                if df is None or df.empty:
                    continue
                self.db.upsert_bars(s, "1m", rows_from_frame(df))
                total += self.store.append(s, "1m", df)
        # daily bars: refresh once a day (cheap)
        if now.timestamp() - self.last_poll_ts > 6 * 3600:
            for group in (self.equities, self.crypto):
                if not group:
                    continue
                try:
                    data = self.provider.fetch_bars(group, "1d", now - timedelta(days=10), now)
                    for s, df in data.items():
                        if df is not None and not df.empty:
                            self.db.upsert_bars(s, "1d", rows_from_frame(df))
                            self.store.append(s, "1d", df)
                except Exception:  # noqa: BLE001
                    self.poll_errors += 1
        self.last_poll_ts = int(now.timestamp())
        return total

    def refresh_prices(self, symbols: list[str] | None = None) -> dict[str, float]:
        symbols = symbols or self.symbols
        try:
            prices = self.provider.latest_prices(symbols)
        except Exception:  # noqa: BLE001
            self.poll_errors += 1
            return {}
        ts = int(_time.time())
        for s, p in prices.items():
            if p and p > 0:
                self.store.set_last_price(s, p, ts)
        return prices

    # ---- access -----------------------------------------------------------------------
    def frame(self, symbol: str, tf: str, end_ts: int | None = None, n: int | None = None, include_partial: bool = False) -> pd.DataFrame:
        return self.store.frame(symbol, tf, end_ts=end_ts, n=n, include_partial=include_partial)

    def last_price(self, symbol: str, end_ts: int | None = None) -> float | None:
        """Latest price. With end_ts: the close of the last minute bar fully closed by end_ts (no look-ahead)."""
        if end_ts is not None:
            return self.store.last_closed_price(symbol, end_ts)
        return self.store.last_price(symbol)

    def symbols_for(self, markets: list[str]) -> list[str]:
        out: list[str] = []
        if "equities" in markets:
            out += self.equities
        if "crypto" in markets:
            out += self.crypto
        return out
