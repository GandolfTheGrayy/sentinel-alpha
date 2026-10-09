"""Claude usage governor: a rolling 7-day cap expressed in API-equivalent USD.

Claude Code reports `total_cost_usd` for every headless run. We sum those over the last
7 days and refuse to start a session whose own budget would push the total over the cap.
The cap is `weekly_share` x an estimated plan allowance; the estimate can be calibrated
from the percentage Claude Code's `/usage` attributes to Sentinel.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sentinel.config import Settings
from sentinel.store.db import Database, to_iso
from sentinel.util.clock import UTC


class BudgetGovernor:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.s = settings

    def allowance(self) -> float:
        cal = self.db.kv_get("budget_calibration")
        if cal and cal.get("allowance_usd"):
            return float(cal["allowance_usd"])
        return self.s.claude.weekly_allowance_usd

    def cap(self) -> float:
        return round(self.allowance() * self.s.claude.weekly_share, 2)

    def spent(self, now: datetime | None = None) -> tuple[float, int]:
        now = now or datetime.now(UTC)
        return self.db.weekly_claude_spend(to_iso(now - timedelta(days=7)))

    def can_start(self, session_budget: float, now: datetime | None = None) -> tuple[bool, str]:
        if not self.s.claude.enabled:
            return False, "claude.enabled is false"
        spent, _ = self.spent(now)
        cap = self.cap()
        if spent + session_budget > cap:
            return False, f"budget: ${spent:.2f} spent + ${session_budget:.2f} session > ${cap:.2f} weekly cap"
        return True, "ok"

    def calibrate(self, observed_weekly_pct: float, now: datetime | None = None) -> dict[str, Any]:
        """The user reports that Sentinel's runs over the last 7 days used X % of the weekly allowance."""
        spent, _ = self.spent(now)
        if observed_weekly_pct <= 0 or spent <= 0:
            self.db.kv_set("budget_calibration", None)
            return {"ok": False, "reason": "need a positive observed percentage and some recorded spend"}
        allowance = spent / (observed_weekly_pct / 100.0)
        cal = {"observed_pct": observed_weekly_pct, "spent_usd_at_calibration": round(spent, 2), "allowance_usd": round(allowance, 2), "at": to_iso(now or datetime.now(UTC))}
        self.db.kv_set("budget_calibration", cal)
        return {"ok": True, **cal}

    def status(self, now: datetime | None = None, schedule: dict[str, str] | None = None, next_runs: dict[str, str | None] | None = None) -> dict[str, Any]:
        now = now or datetime.now(UTC)
        spent, runs = self.spent(now)
        cap = self.cap()
        cal = self.db.kv_get("budget_calibration") or {}
        history = []
        for i in range(8):
            w_end = now - timedelta(days=7 * i)
            w_start = w_end - timedelta(days=7)
            row = self.db.one("SELECT COALESCE(SUM(cost_usd),0) AS c, COUNT(*) AS n FROM claude_runs WHERE started_at>=? AND started_at<? AND status IN ('ok','error')", (to_iso(w_start), to_iso(w_end)))
            history.append({"week_start": to_iso(w_start), "spent_usd": round(float(row["c"]), 2) if row else 0.0, "runs": int(row["n"]) if row else 0})
        history.reverse()
        return {
            "plan": self.s.claude.plan, "weekly_share": self.s.claude.weekly_share, "weekly_allowance_usd_est": round(self.allowance(), 2),
            "weekly_cap_usd": cap, "spent_usd": round(spent, 2), "remaining_usd": round(max(0.0, cap - spent), 2), "week_start": to_iso(now - timedelta(days=7)),
            "runs_this_week": runs, "share_of_plan": self.s.claude.weekly_share, "enabled": self.s.claude.enabled,
            "calibration": {"observed_pct": cal.get("observed_pct"), "allowance_usd": cal.get("allowance_usd"), "at": cal.get("at"),
                            "note": "Enter the weekly % that Claude Code's /usage attributes to Sentinel (after a week of runs) to rescale the allowance estimate."},
            "models": {"analyst": self.s.claude.analyst.model, "strategist": self.s.claude.strategist.model},
            "session_budgets": {"analyst": self.s.claude.analyst.max_budget_usd, "strategist": self.s.claude.strategist.max_budget_usd},
            "schedule": schedule or {"analyst": f"{self.s.schedules.analyst} America/New_York daily", "strategist": f"{self.s.schedules.strategist} America/New_York"},
            "next_analyst_run": (next_runs or {}).get("analyst"), "next_strategist_run": (next_runs or {}).get("strategist"),
            "history": history,
        }
