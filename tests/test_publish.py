import subprocess
from datetime import datetime, timedelta, timezone

from sentinel.learn.publish import build_journal, commit_message, publish
from sentinel.store.db import Database, to_iso

NOW = datetime(2026, 10, 8, 21, 0, tzinfo=timezone.utc)


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _seed(db: Database) -> None:
    exit_ts = to_iso(NOW - timedelta(hours=2))
    db.execute("INSERT INTO trades(variant_id,family,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,pnl_r,fees,hold_minutes,exit_reason) "
               "VALUES('orb#1','breakout_orb','AAPL','long',10,?,100,?,103,30.0,1.5,0,120,'target')", (to_iso(NOW - timedelta(hours=4)), exit_ts))
    db.execute("INSERT INTO trades(variant_id,family,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,pnl_r,fees,hold_minutes,exit_reason) "
               "VALUES('squeeze#2','squeeze','MSFT','long',5,?,200,?,196,-20.5,-1.0,0,60,'stop')", (to_iso(NOW - timedelta(hours=3)), exit_ts))
    # an old trade outside the window must not count
    db.execute("INSERT INTO trades(variant_id,family,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,pnl_r,fees,hold_minutes,exit_reason) "
               "VALUES('orb#1','breakout_orb','SPY','long',1,?,400,?,401,999,1,0,10,'target')", (to_iso(NOW - timedelta(days=3)), to_iso(NOW - timedelta(days=2))))
    db.insert_variant({"id": "orb#1", "family": "breakout_orb", "name": "orb", "params": {}, "status": "active", "status_reason": None, "origin": "seed", "parent_id": None,
                       "markets": ["equities"], "timeframe": "5m", "is_control": False, "allocation": 0.4, "notes": None})
    db.add_event("tournament", "orb#1: incubating -> active (n=30, CI above zero)", "info", ts=to_iso(NOW - timedelta(hours=1)))
    db.add_event("risk", "daily loss 3.10% >= pause limit; no new entries today", "warn", ts=to_iso(NOW - timedelta(hours=1)))
    for i in range(3):
        db.add_event("risk", 'entry order failed orb#1 IWM: {"code":40310000,"message":"insufficient qty"}', "error", ts=to_iso(NOW - timedelta(minutes=50 - i)))
    rid = db.insert_run("analyst", "sonnet", "digest")
    db.update_run(rid, status="ok", cost_usd=0.44, num_turns=7, analysis_md="# Findings\n\nsqueeze has no edge.")
    pid = db.insert_proposal(rid, "retire", "squeeze#2", {"type": "retire"}, "no edge")
    db.update_proposal(pid, status="rejected", decision_reason="fails the 15-trade backtest gate")


STATUS = {"mode": "paper", "engine": {"running": True, "paused": False, "halted": False},
          "account": {"equity": 95_000.0, "day_pnl": -120.5, "day_pnl_pct": -0.127, "week_pnl": 300.0, "week_pnl_pct": 0.3, "total_pnl": -5000.0, "total_pnl_pct": -5.0,
                      "starting_equity": 100_000.0, "open_positions": 2, "gross_exposure_pct": 12.5},
          "population": {"active": 3, "incubating": 1}, "budget": {"week_spent_usd": 1.25, "weekly_cap_usd": 12.5}}


def test_build_journal_covers_the_last_day(settings, tmp_path):
    db = Database(tmp_path / "j.db")
    _seed(db)
    md, facts = build_journal(db, settings, STATUS, NOW)
    assert facts["day"] == "2026-10-08" and facts["n_trades"] == 2 and facts["pnl"] == 9.5 and facts["equity"] == 95_000.0
    assert "| orb#1 | 1 | 1 | +30.00 | +1.50 |" in md and "999" not in md
    assert "orb#1: incubating -> active" in md and "**warn** risk" in md
    assert "×3 **error** risk: entry order failed orb#1 IWM" in md and md.count("insufficient qty") == 0
    assert "### analyst session #1 — ok, $0.44, 7 turns" in md and "#### Findings" in md
    assert "| #1 retire | squeeze#2 | rejected: fails the 15-trade backtest gate |" in md
    assert "| Equity | 95,000.00 |" in md and "| Day P&L | -120.50 (-0.13%) |" in md
    assert commit_message(facts) == "journal 2026-10-08: 2 trades, +9.50 P&L, equity 95,000; analyst #1: 1 proposals, 0 accepted"


def test_publish_commits_journal_and_memory_only(settings, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "lab@example.invalid")
    _git(repo, "config", "user.name", "Sentinel lab")
    lab = repo / "lab"
    lab.mkdir()
    (lab / "memory.md").write_text("## Lab memory\n- squeeze is the worst family\n", encoding="utf-8")
    (repo / "unrelated.py").write_text("x = 1\n", encoding="utf-8")  # work in progress that must stay out of the commit
    db = Database(tmp_path / "p.db")
    _seed(db)

    res = publish(db, settings, STATUS, NOW, log=lambda m: None, push=False, lab_dir=lab, repo=repo)
    assert res["committed"] and not res["pushed"] and res["sha"]
    assert _git(repo, "log", "--format=%s", "-1").strip() == res["message"]
    committed = set(_git(repo, "ls-tree", "-r", "--name-only", "HEAD").split())
    assert committed == {"lab/journal/2026-10-08.md", "lab/journal/README.md", "lab/memory.md"}
    index = (lab / "journal" / "README.md").read_text(encoding="utf-8")
    assert "- [2026-10-08](2026-10-08.md) — 2 trades, +9.50 P&L, equity 95,000" in index

    # same day, nothing new: no empty commit
    again = publish(db, settings, STATUS, NOW, log=lambda m: None, push=False, lab_dir=lab, repo=repo)
    assert not again["committed"] and again["reason"] == "no changes"

    # the next day gets its own page and the index grows
    nxt = publish(db, settings, STATUS, NOW + timedelta(days=1), log=lambda m: None, push=False, lab_dir=lab, repo=repo)
    assert nxt["committed"] and nxt["day"] == "2026-10-09"
    assert _git(repo, "rev-list", "--count", "HEAD").strip() == "2"
    assert (lab / "journal" / "README.md").read_text(encoding="utf-8").index("2026-10-09") < (lab / "journal" / "README.md").read_text(encoding="utf-8").index("2026-10-08")


def test_publish_skips_an_ignored_lab(settings, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text("lab-sim/\n", encoding="utf-8")
    lab = repo / "lab-sim"
    db = Database(tmp_path / "s.db")
    res = publish(db, settings, STATUS, NOW, log=lambda m: None, push=False, lab_dir=lab, repo=repo)
    assert not res["committed"] and res["reason"] == "lab is gitignored"
    assert (lab / "journal" / "2026-10-08.md").exists()
