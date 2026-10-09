"""Per-variant performance statistics with confidence intervals."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 400, seed: int = 7) -> tuple[float, float]:
    if len(x) == 0:
        return (0.0, 0.0)
    if len(x) < 5:
        m = float(np.mean(x))
        return (m, m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def max_drawdown(pnls: np.ndarray, sleeve: float) -> float:
    if len(pnls) == 0 or sleeve <= 0:
        return 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    dd = peak - cum
    return float(dd.max() / sleeve * 100.0)


def compute_metrics(trades: list[dict[str, Any]], sleeve: float) -> dict[str, Any]:
    """Summary metrics for a list of closed trades (dicts with pnl, pnl_r, hold_minutes, exit_reason)."""
    n = len(trades)
    if n == 0:
        return {"n": 0, "win_rate": None, "win_rate_ci": [0.0, 1.0], "expectancy_r": None, "expectancy_ci": [0.0, 0.0], "profit_factor": None,
                "sharpe": None, "max_dd_pct": 0.0, "pnl": 0.0, "avg_hold_minutes": None, "avg_win_r": None, "avg_loss_r": None, "by_exit_reason": []}
    pnl = np.array([float(t["pnl"]) for t in trades])
    r = np.array([float(t["pnl_r"]) for t in trades])
    wins = int((pnl > 0).sum())
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    hold = np.array([float(t.get("hold_minutes") or 0.0) for t in trades])
    # per-trade Sharpe annualised by trade frequency (trades per year estimated from the span)
    sharpe = None
    if n >= 5 and r.std(ddof=1) > 0:
        try:
            first = min(t["entry_ts"] for t in trades)
            last = max(t["exit_ts"] for t in trades)
            from sentinel.store.db import from_iso
            span_days = max(1.0, (from_iso(last) - from_iso(first)).total_seconds() / 86400.0)
            per_year = n / span_days * 365.0
            sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(min(per_year, 2000.0)))
        except Exception:  # noqa: BLE001
            sharpe = None
    by_reason: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_reason[t.get("exit_reason") or "?"].append(float(t["pnl_r"]))
    return {
        "n": n,
        "win_rate": round(wins / n, 4),
        "win_rate_ci": [round(x, 4) for x in wilson_ci(wins, n)],
        "expectancy_r": round(float(r.mean()), 4),
        "expectancy_ci": [round(x, 4) for x in bootstrap_mean_ci(r)],
        "profit_factor": round(pf, 3) if pf != float("inf") else 99.0,
        "sharpe": round(sharpe, 3) if sharpe is not None else None,
        "max_dd_pct": round(max_drawdown(pnl, sleeve), 3),
        "pnl": round(float(pnl.sum()), 2),
        "avg_hold_minutes": round(float(hold.mean()), 1),
        "avg_win_r": round(float(r[r > 0].mean()), 3) if (r > 0).any() else None,
        "avg_loss_r": round(float(r[r <= 0].mean()), 3) if (r <= 0).any() else None,
        "by_exit_reason": [{"exit_reason": k, "n": len(v), "expectancy_r": round(float(np.mean(v)), 3)} for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))],
    }


def rolling_window(trades: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Last n trades by exit time."""
    return sorted(trades, key=lambda t: (t["exit_ts"], t.get("id") or 0))[-n:]
