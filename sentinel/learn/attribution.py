"""Factor attribution: which conditions at entry led to good or bad outcomes.

Deterministic statistics only (no LLM). Produces the report consumed by the Factors page
and a compact markdown digest for Claude. Filter candidates are validated out of sample
(chosen on the first 70 % of trades by time, scored on the last 30 %) to limit the
multiple-comparisons trap.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from sentinel.store.db import from_iso
from sentinel.util.clock import UTC

CATEGORICAL = {"hour_et", "dow", "session", "regime_trend", "regime_vol", "is_crypto"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _ci(x: np.ndarray) -> list[float]:
    if len(x) < 2:
        m = float(x.mean()) if len(x) else 0.0
        return [round(m, 3), round(m, 3)]
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    m = float(x.mean())
    return [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)]


def _frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in trades:
        f = t.get("features") or {}
        rows.append({"pnl_r": float(t["pnl_r"]), "pnl": float(t["pnl"]), "win": 1.0 if float(t["pnl"]) > 0 else 0.0, "family": t["family"], "variant_id": t["variant_id"],
                     "symbol": t["symbol"], "side": t["side"], "exit_reason": t.get("exit_reason"), "exit_ts": t["exit_ts"], "hold_minutes": float(t.get("hold_minutes") or 0),
                     **{k: float(v) for k, v in f.items() if isinstance(v, (int, float))}})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("exit_ts").reset_index(drop=True)
    return df


def _bucketize(s: pd.Series, name: str) -> tuple[pd.Series, list[str]]:
    s = s.astype(float)
    if name in CATEGORICAL or s.nunique() <= 6:
        vals = sorted(s.dropna().unique().tolist())
        labels = [_cat_label(name, v) for v in vals]
        mapping = {v: lab for v, lab in zip(vals, labels)}
        return s.map(mapping), labels
    q = 5 if s.notna().sum() >= 150 else 3
    try:
        cats = pd.qcut(s, q=q, duplicates="drop")
    except ValueError:
        return pd.Series([None] * len(s), index=s.index), []
    labels = []
    for iv in cats.cat.categories:
        labels.append(f"{iv.left:.3g}..{iv.right:.3g}")
    mapping = {iv: lab for iv, lab in zip(cats.cat.categories, labels)}
    return cats.map(mapping).astype(object), labels


def _cat_label(name: str, v: float) -> str:
    if name == "dow":
        return DOW[int(v)] if 0 <= int(v) < 7 else str(int(v))
    if name == "session":
        return {0: "closed", 1: "pre", 2: "regular", 3: "post"}.get(int(v), str(int(v)))
    if name == "regime_vol":
        return {0: "low vol", 1: "mid vol", 2: "high vol"}.get(int(v), str(int(v)))
    if name == "regime_trend":
        return "trending" if int(v) == 1 else "ranging"
    if name == "is_crypto":
        return "crypto" if int(v) == 1 else "equities"
    if name == "hour_et":
        return f"{int(v):02d}h"
    return f"{v:g}"


def build_report(trades: list[dict[str, Any]], *, window_days: int, min_trades: int, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=window_days)
    trades = [t for t in trades if from_iso(t["exit_ts"]) >= since]
    df = _frame(trades)
    n = len(df)
    report: dict[str, Any] = {"generated_at": now.isoformat().replace("+00:00", "Z"), "n_trades": n, "window_days": window_days,
                              "overall": None, "features": [], "logistic": [], "importance_method": None, "filters": [], "heatmap": None,
                              "by_family": [], "by_regime": [], "by_exit_reason": [], "by_side": [], "by_market": [], "by_symbol": [], "insufficient": n < min_trades}
    if n == 0:
        return report
    y = df["pnl_r"].values
    report["overall"] = {"n": n, "expectancy_r": round(float(y.mean()), 3), "ci": _ci(y), "win_rate": round(float(df["win"].mean()), 3), "pnl": round(float(df["pnl"].sum()), 2)}
    report["by_family"] = _group(df, "family")
    report["by_exit_reason"] = _group(df, "exit_reason")
    report["by_side"] = _group(df, "side")
    report["by_symbol"] = _group(df, "symbol")[:15]
    if "is_crypto" in df:
        df["market"] = np.where(df["is_crypto"] == 1.0, "crypto", "equities")
        report["by_market"] = _group(df, "market")
    if "regime_trend" in df and "regime_vol" in df:
        df["regime"] = df["regime_trend"].map({1.0: "trend", 0.0: "range"}).fillna("?") + "/" + df["regime_vol"].map({0.0: "low_vol", 1.0: "mid_vol", 2.0: "high_vol"}).fillna("?")
        report["by_regime"] = _group(df, "regime")
    if n < min_trades:
        return report

    meta_cols = {"pnl_r", "pnl", "win", "family", "variant_id", "symbol", "side", "exit_reason", "exit_ts", "hold_minutes", "market", "regime"}
    feats = [c for c in df.columns if c not in meta_cols and df[c].notna().mean() >= 0.7 and df[c].nunique() > 1]

    # ---- importances ----------------------------------------------------------------------
    importance: dict[str, float] = {}
    if n >= 80 and feats:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            from sklearn.inspection import permutation_importance

            X = df[feats].astype(float)
            model = HistGradientBoostingRegressor(max_depth=3, max_iter=120, learning_rate=0.05, min_samples_leaf=max(10, n // 25), random_state=7)
            model.fit(X, y)
            pi = permutation_importance(model, X, y, n_repeats=5, random_state=7)
            for f, imp in zip(feats, pi.importances_mean):
                importance[f] = round(float(max(imp, 0.0)), 5)
            report["importance_method"] = "gradient boosting permutation importance (R-multiple)"
        except Exception:  # noqa: BLE001
            importance = {}
    if not importance:
        # cheap fallback: absolute spearman correlation with R
        for f in feats:
            try:
                importance[f] = round(abs(float(df[f].corr(df["pnl_r"], method="spearman"))), 5)
            except Exception:  # noqa: BLE001
                importance[f] = 0.0
        report["importance_method"] = "abs. Spearman correlation with R-multiple"

    # ---- logistic regression on win ----------------------------------------------------
    if n >= 50 and feats:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            X = df[feats].astype(float).fillna(df[feats].median())
            Xs = StandardScaler().fit_transform(X)
            lr = LogisticRegression(C=0.5, max_iter=500).fit(Xs, df["win"].values)
            coefs = sorted(zip(feats, lr.coef_[0]), key=lambda kv: -abs(kv[1]))
            report["logistic"] = [{"name": f, "coef": round(float(c), 4)} for f, c in coefs[:15]]
        except Exception:  # noqa: BLE001
            report["logistic"] = []

    # ---- bucketed expectancy -----------------------------------------------------------
    features_out = []
    split = int(n * 0.7)
    overall_mean = float(y.mean())
    filters = []
    for f in feats:
        b, labels = _bucketize(df[f], f)
        if not labels:
            continue
        buckets = []
        for lab in labels:
            mask = (b == lab).values
            k = int(mask.sum())
            if k == 0:
                continue
            r = y[mask]
            buckets.append({"label": lab, "n": k, "expectancy_r": round(float(r.mean()), 3), "ci": _ci(r), "win_rate": round(float(df["win"].values[mask].mean()), 3)})
            # filter candidate: a bucket that is clearly negative and worse than the overall mean, chosen in-sample
            if k >= 15:
                mask_in = mask[:split]
                r_in = y[:split][mask_in]
                if len(r_in) >= 10 and r_in.mean() < 0 and _ci(r_in)[1] < overall_mean:
                    keep_in = y[:split][~mask_in]
                    keep_oos = y[split:][~mask[split:]]
                    oos_all = y[split:]
                    if len(keep_oos) >= 10 and len(oos_all) >= 15:
                        oos_delta = float(keep_oos.mean() - oos_all.mean())
                        filters.append({"feature": f, "rule": f"exclude {f} in {lab}", "n_removed": k, "expectancy_before": round(overall_mean, 3),
                                        "expectancy_after": round(float(np.concatenate([keep_in, keep_oos]).mean()), 3), "oos_delta": round(oos_delta, 3),
                                        "oos_n_removed": int(mask[split:].sum()), "verdict": "candidate" if (oos_delta > 0.03 and mask[split:].sum() >= 5) else "rejected"})
        features_out.append({"name": f, "importance": importance.get(f, 0.0), "buckets": buckets})
    features_out.sort(key=lambda x: -x["importance"])
    report["features"] = features_out
    filters.sort(key=lambda x: (x["verdict"] != "candidate", -x["oos_delta"]))
    report["filters"] = filters[:20]

    # ---- hour x weekday heatmap -----------------------------------------------------------
    if "hour_et" in df and "dow" in df:
        hours = sorted(int(h) for h in df["hour_et"].dropna().unique())
        dows = sorted(int(d) for d in df["dow"].dropna().unique())
        values, counts = [], []
        for d in dows:
            row_v, row_c = [], []
            for h in hours:
                m = (df["dow"] == d) & (df["hour_et"] == h)
                k = int(m.sum())
                row_c.append(k)
                row_v.append(round(float(y[m.values].mean()), 3) if k else None)
            values.append(row_v)
            counts.append(row_c)
        report["heatmap"] = {"rows": [DOW[d] for d in dows], "cols": [f"{h:02d}" for h in hours], "values": values, "counts": counts}
    return report


def _group(df: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    out = []
    if col not in df:
        return out
    for key, g in df.groupby(col):
        r = g["pnl_r"].values
        out.append({col: key, "n": int(len(g)), "expectancy_r": round(float(r.mean()), 3), "ci": _ci(r), "win_rate": round(float(g["win"].mean()), 3), "pnl": round(float(g["pnl"].sum()), 2)})
    out.sort(key=lambda x: -x["n"])
    return out


def digest_md(report: dict[str, Any], max_features: int = 10) -> str:
    """Compact markdown for Claude (a few hundred tokens)."""
    if not report or report.get("n_trades", 0) == 0:
        return "_No closed trades in the attribution window yet._"
    o = report.get("overall") or {}
    lines = [f"**Attribution window:** last {report['window_days']} days, {report['n_trades']} trades, expectancy {o.get('expectancy_r', 0):+.2f}R "
             f"(95% CI {o.get('ci', [0, 0])[0]:+.2f}..{o.get('ci', [0, 0])[1]:+.2f}), win rate {o.get('win_rate', 0):.0%}, P&L {o.get('pnl', 0):+.0f}."]
    if report.get("insufficient"):
        lines.append("_Too few trades for factor analysis (need more closed trades)._")
    for key, title in (("by_family", "By family"), ("by_market", "By market"), ("by_regime", "By regime"), ("by_exit_reason", "By exit reason"), ("by_side", "By side")):
        rows = report.get(key) or []
        if rows:
            lines.append(f"\n**{title}:** " + "; ".join(f"{r[key.replace('by_', '')]} n={r['n']} {r['expectancy_r']:+.2f}R wr={r['win_rate']:.0%}" for r in rows[:8]))
    feats = report.get("features") or []
    if feats:
        lines.append(f"\n**Top features** ({report.get('importance_method')}):")
        for f in feats[:max_features]:
            b = ", ".join(f"{x['label']}: {x['expectancy_r']:+.2f}R (n={x['n']})" for x in f["buckets"])
            lines.append(f"- `{f['name']}` imp={f['importance']:.3f} -> {b}")
    lg = report.get("logistic") or []
    if lg:
        lines.append("\n**Logistic (P(win), standardised coefs):** " + ", ".join(f"{x['name']} {x['coef']:+.2f}" for x in lg[:8]))
    fl = [f for f in (report.get("filters") or []) if f["verdict"] == "candidate"]
    if fl:
        lines.append("\n**Filter candidates (validated out of sample):**")
        for f in fl[:6]:
            lines.append(f"- {f['rule']}: removes {f['n_removed']} trades, expectancy {f['expectancy_before']:+.2f}R -> {f['expectancy_after']:+.2f}R, OOS delta {f['oos_delta']:+.2f}R")
    hm = report.get("heatmap")
    if hm:
        best, worst = [], []
        for i, row in enumerate(hm["values"]):
            for j, v in enumerate(row):
                if v is not None and hm["counts"][i][j] >= 8:
                    (best if v > 0 else worst).append((v, f"{hm['rows'][i]} {hm['cols'][j]}h n={hm['counts'][i][j]}"))
        best.sort(reverse=True)
        worst.sort()
        if best or worst:
            lines.append("\n**Time-of-week:** best " + "; ".join(f"{lab} {v:+.2f}R" for v, lab in best[:3]) + " | worst " + "; ".join(f"{lab} {v:+.2f}R" for v, lab in worst[:3]))
    return "\n".join(lines)
