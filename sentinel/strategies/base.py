"""Strategy base classes, the signal contract and the family registry.

A *family* is a Strategy subclass. A *variant* is a family plus a concrete parameter
dict. Strategies are pure: they read bars from the context and return signals; the
engine owns sizing, execution and the generic exits (stop, target, time, session end,
trailing). A strategy may add its own exit logic in `manage`.

Indicators should be requested through `ctx.series(...)` so that backtests compute each
indicator once over the whole history instead of once per bar.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np
import pandas as pd

from sentinel.data.market import MarketData

ParamSpec = tuple  # ("int", lo, hi) | ("float", lo, hi) | ("choice", [..])


@dataclass
class Signal:
    symbol: str
    side: str                     # "long" | "short"
    stop: float                   # absolute stop price
    target: float | None = None   # absolute target price
    max_hold_bars: int | None = None
    flat_at_session_end: bool = False
    trail_atr: float | None = None  # trailing stop distance in price units (None = fixed stop)
    strength: float = 0.5
    reason: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    def validate(self, price: float) -> str | None:
        if self.side not in ("long", "short"):
            return "bad side"
        if not (price > 0 and self.stop > 0):
            return "bad price/stop"
        if self.side == "long" and self.stop >= price:
            return "long stop must be below price"
        if self.side == "short" and self.stop <= price:
            return "short stop must be above price"
        if self.target is not None:
            if self.side == "long" and self.target <= price:
                return "long target must be above price"
            if self.side == "short" and self.target >= price:
                return "short target must be below price"
        return None


class IndicatorCache:
    """Caches full-history indicator series keyed by (symbol, tf, key, base_len)."""

    def __init__(self) -> None:
        self._data: dict[tuple, pd.Series | pd.DataFrame] = {}

    def get(self, md: MarketData, symbol: str, tf: str, key: str, fn: Callable[[pd.DataFrame], Any]):
        full = md.frame(symbol, tf)  # whole loaded history, closed bars included partial last bucket
        k = (symbol, tf, key, len(full))
        hit = self._data.get(k)
        if hit is None:
            hit = fn(full)
            # keep the cache bounded: drop older lengths for the same key
            for old in [x for x in self._data if x[:3] == k[:3] and x != k]:
                self._data.pop(old, None)
            self._data[k] = hit
        return hit

    def clear(self) -> None:
        self._data.clear()


@dataclass
class Context:
    """What a strategy sees on a bar close."""
    md: MarketData
    ts: int                              # epoch seconds of 'now' (bar close)
    tf: str
    symbols: list[str]                   # symbols this variant may trade on this cycle
    open_lots: list[dict]                # this variant's open lots
    session: str                         # regular | pre | post | closed (equities)
    rng: np.random.Generator             # seeded per variant (for control strategies)
    cache: IndicatorCache
    variant_id: str = ""
    allow_short: bool = True

    def frame(self, symbol: str, tf: str | None = None, n: int = 300) -> pd.DataFrame:
        return self.md.frame(symbol, tf or self.tf, end_ts=self.ts, n=n)

    def daily(self, symbol: str, n: int = 260) -> pd.DataFrame:
        return self.md.frame(symbol, "1d", end_ts=self.ts, n=n)

    def series(self, symbol: str, key: str, fn: Callable[[pd.DataFrame], Any], tf: str | None = None, n: int = 300):
        """Indicator over the symbol's history, sliced to closed bars up to ts. fn(df) -> Series/DataFrame."""
        tf = tf or self.tf
        full = self.cache.get(self.md, symbol, tf, key, fn)
        df = self.frame(symbol, tf, n=n)
        if df.empty:
            return full.iloc[0:0]
        out = full.loc[:df.index[-1]]
        return out.iloc[-n:] if len(out) > n else out

    def price(self, symbol: str) -> float | None:
        return self.md.last_price(symbol, end_ts=self.ts)

    def lots_for(self, symbol: str) -> list[dict]:
        return [l for l in self.open_lots if l["symbol"] == symbol]

    def has_lot(self, symbol: str) -> bool:
        return any(l["symbol"] == symbol for l in self.open_lots)


class Strategy:
    family: ClassVar[str] = ""
    description: ClassVar[str] = ""
    timeframe: ClassVar[str] = "15m"
    markets: ClassVar[list[str]] = ["equities", "crypto"]
    trigger: ClassVar[str | list[str]] = "bar"   # "bar" or ["HH:MM", ...] America/New_York
    is_control: ClassVar[bool] = False
    intraday_only: ClassVar[bool] = False        # equities: flatten at session end
    regular_session_only: ClassVar[bool] = True  # equities: only act during the regular session
    shortable: ClassVar[bool] = True
    warmup_bars: ClassVar[int] = 60
    DEFAULTS: ClassVar[dict[str, Any]] = {}
    PARAM_SPACE: ClassVar[dict[str, ParamSpec]] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        self.p: dict[str, Any] = {**self.DEFAULTS, **(params or {})}
        self.p = self.validate_params(self.p)

    # ---- resolved attributes (params may override class defaults) -----------------------
    def tf(self) -> str:
        return str(self.p.get("tf", self.timeframe))

    def market_list(self) -> list[str]:
        m = self.p.get("markets")
        if isinstance(m, str):
            return [m]
        return list(m) if m else list(self.markets)

    # ---- hooks --------------------------------------------------------------------------
    def on_cycle(self, ctx: Context) -> list[Signal]:
        """Called once per closed bar of tf() (or per time trigger). Default: per-symbol on_bar."""
        out: list[Signal] = []
        for s in ctx.symbols:
            df = ctx.frame(s)
            if len(df) < self.warmup_bars:
                continue
            try:
                out.extend(self.on_bar(ctx, s, df))
            except Exception as exc:  # noqa: BLE001 - one symbol must not break the cycle
                ctx.md.db.add_event("system", f"{self.family} on_bar failed for {s}: {exc}", level="warn")
        return out

    def on_bar(self, ctx: Context, symbol: str, df: pd.DataFrame) -> list[Signal]:
        return []

    def manage(self, ctx: Context, lot: dict, df: pd.DataFrame) -> str | None:
        """Optional strategy-specific exit. Return an exit reason or None."""
        return None

    # ---- params -------------------------------------------------------------------------
    @classmethod
    def validate_params(cls, p: dict[str, Any]) -> dict[str, Any]:
        out = dict(p)
        for k, spec in cls.PARAM_SPACE.items():
            if k not in out:
                continue
            kind = spec[0]
            try:
                if kind == "int":
                    out[k] = int(min(max(round(float(out[k])), spec[1]), spec[2]))
                elif kind == "float":
                    out[k] = float(min(max(float(out[k]), spec[1]), spec[2]))
                elif kind == "choice":
                    if out[k] not in spec[1]:
                        out[k] = spec[1][0]
            except (TypeError, ValueError):
                out[k] = cls.DEFAULTS.get(k)
        return out

    @classmethod
    def mutate(cls, params: dict[str, Any], rng: np.random.Generator, scale: float = 0.25) -> dict[str, Any]:
        out = dict(params)
        keys = [k for k in cls.PARAM_SPACE]
        if not keys:
            return out
        n_mut = max(1, int(round(len(keys) * 0.4)))
        for k in rng.choice(keys, size=min(n_mut, len(keys)), replace=False):
            spec = cls.PARAM_SPACE[k]
            kind = spec[0]
            if kind in ("int", "float"):
                lo, hi = float(spec[1]), float(spec[2])
                cur = float(out.get(k, cls.DEFAULTS.get(k, (lo + hi) / 2)))
                new = cur + rng.normal(0.0, scale * (hi - lo))
                out[k] = int(round(new)) if kind == "int" else float(new)
            else:
                out[k] = spec[1][int(rng.integers(len(spec[1])))]
        return cls.validate_params(out)

    @classmethod
    def random_params(cls, rng: np.random.Generator) -> dict[str, Any]:
        out = dict(cls.DEFAULTS)
        for k, spec in cls.PARAM_SPACE.items():
            if spec[0] == "int":
                out[k] = int(rng.integers(spec[1], spec[2] + 1))
            elif spec[0] == "float":
                out[k] = float(rng.uniform(spec[1], spec[2]))
            else:
                out[k] = spec[1][int(rng.integers(len(spec[1])))]
        return cls.validate_params(out)

    @classmethod
    def display_name(cls, params: dict[str, Any]) -> str:
        shown = []
        for k in list(cls.PARAM_SPACE)[:4]:
            if k in params:
                v = params[k]
                shown.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
        return f"{cls.family} " + " ".join(shown) if shown else cls.family

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {"family": cls.family, "description": cls.description, "timeframe": cls.timeframe, "markets": cls.markets,
                "trigger": cls.trigger, "is_control": cls.is_control, "intraday_only": cls.intraday_only, "shortable": cls.shortable,
                "defaults": cls.DEFAULTS, "param_space": {k: list(v) for k, v in cls.PARAM_SPACE.items()}}


# ---- helpers shared by families ---------------------------------------------------------
def atr_stop(price: float, atr_v: float, mult: float, side: str) -> float:
    dist = max(atr_v * mult, price * 0.0005)
    return price - dist if side == "long" else price + dist


def r_target(price: float, stop: float, r_mult: float, side: str) -> float:
    risk = abs(price - stop)
    return price + risk * r_mult if side == "long" else price - risk * r_mult


def safe_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


# ---- registry ---------------------------------------------------------------------------
REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    if not cls.family:
        raise ValueError("strategy needs a family name")
    REGISTRY[cls.family] = cls
    return cls


def get_family(name: str) -> type[Strategy]:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy family: {name}")
    return REGISTRY[name]


ALLOWED_IMPORTS = {"numpy", "pandas", "math", "dataclasses", "typing", "sentinel.strategies.base", "sentinel.data.indicators", "sentinel.data", "sentinel.strategies", "__future__"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "getattr", "setattr", "delattr", "input", "breakpoint", "vars", "dir"}


def check_evolved_source(src: str) -> list[str]:
    """Static allow-list check for Claude-written strategy modules. Returns a list of problems."""
    problems: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                root = n.split(".")[0]
                if n not in ALLOWED_IMPORTS and root not in ALLOWED_IMPORTS:
                    problems.append(f"import not allowed: {n}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"forbidden name: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ("__init__",):
            problems.append(f"dunder attribute access: {node.attr}")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)):
            problems.append(f"construct not allowed: {type(node).__name__}")
    if "@register" not in src and "register(" not in src:
        problems.append("module must register a Strategy subclass with @register")
    return problems


def load_evolved(dir_path: Path, log=print) -> list[str]:
    """Import every *.py in dir_path that passes the static check. Returns loaded family names."""
    loaded: list[str] = []
    if not dir_path.exists():
        return loaded
    for py in sorted(dir_path.glob("*.py")):
        if py.name.startswith("_"):
            continue
        src = py.read_text(encoding="utf-8")
        problems = check_evolved_source(src)
        if problems:
            log(f"evolved strategy {py.name} rejected: {problems[:3]}")
            continue
        before = set(REGISTRY)
        name = f"sentinel.strategies.evolved.{py.stem}"
        try:
            spec = importlib.util.spec_from_file_location(name, py)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001
            log(f"evolved strategy {py.name} failed to import: {exc}")
            continue
        loaded += sorted(set(REGISTRY) - before)
    return loaded
