"""Daily journal: one markdown page per day under lab/journal/, committed and pushed with lab/memory.md.

Everything else the lab produces lives in the local database and the gitignored lab/runs, so without
this job the repository never shows that the engine and the research sessions are running. The page is
built from the database only; git is driven through subprocess with explicit pathspecs so the commit
can never sweep in unrelated work sitting in the working tree.
"""
from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from sentinel.config import Settings
from sentinel.store.db import Database, to_iso
from sentinel.util.clock import to_ny

WINDOW_HOURS = 24
SUMMARY_TAG = "<!-- summary: "


def _money(x: float | None) -> str:
    return f"{float(x or 0):+,.2f}"


def _pct(x: float | None) -> str:
    return f"{float(x or 0):+.2f}%"


def _demote(md: str, levels: int = 3) -> str:
    """Push an analysis' own headings below the journal's so the page keeps one outline."""
    return re.sub(r"^(#{1,3})(\s)", lambda m: "#" * min(6, len(m.group(1)) + levels) + m.group(2), md, flags=re.M)


def _collapse(events: list[dict[str, Any]], cap: int = 25) -> list[str]:
    """One line per distinct message (broker payloads stripped), with a count and the first and last time it fired."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in events:
        text = re.sub(r"\s*\{.*\}\s*$", "", e["message"]).strip().rstrip(" :")
        key = (e["level"], e["kind"], text)
        g = groups.setdefault(key, {"n": 0, "first": e["ts"], "last": e["ts"]})
        g["n"] += 1
        g["last"] = e["ts"]
    lines = []
    for (level, kind, text), g in groups.items():
        when = g["first"] if g["n"] == 1 else f"{g['first']} .. {g['last']} ×{g['n']}"
        lines.append(f"{when} **{level}** {kind}: {text}")
    if len(lines) > cap:
        lines = lines[:cap] + [f"… {len(lines) - cap} more distinct messages"]
    return lines


def build_journal(db: Database, settings: Settings, status: dict[str, Any], now: datetime) -> tuple[str, dict[str, Any]]:
    """Render the day's page; also return the figures the commit message and the index use."""
    day = to_ny(now).date().isoformat()
    since = to_iso(now - timedelta(hours=WINDOW_HOURS))
    acct = status.get("account") or {}
    engine = status.get("engine") or {}
    pop = status.get("population") or {}
    budget = status.get("budget") or {}

    trades = db.trades(limit=5000, since=since, with_features=False)
    pnl = sum(float(t["pnl"]) for t in trades)
    r_sum = sum(float(t["pnl_r"]) for t in trades)
    wins = sum(1 for t in trades if float(t["pnl"]) > 0)
    stops = sum(1 for t in trades if t["exit_reason"] == "stop")
    per_variant: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "r": 0.0, "wins": 0})
    for t in trades:
        v = per_variant[t["variant_id"]]
        v["n"] += 1
        v["pnl"] += float(t["pnl"])
        v["r"] += float(t["pnl_r"])
        v["wins"] += 1 if float(t["pnl"]) > 0 else 0

    events = db.events(limit=5000, since=since)
    lifecycle = [e for e in events if e["kind"] == "tournament"]
    risk = [e for e in events if e["kind"] == "risk" and e["level"] in ("warn", "error")]
    errors = [e for e in events if e["level"] == "error" and e["kind"] != "risk"]
    runs = [r for r in db.runs(limit=10) if (r.get("started_at") or "") >= since]
    variants = db.variants(include_retired=False)
    equity = float(acct.get("equity") or 0.0)

    state = "halted" if engine.get("halted") else "paused" if engine.get("paused") else "running" if engine.get("running") else "stopped"
    if engine.get("halted") and engine.get("halt_reason"):
        state += f" ({engine['halt_reason']})"
    if engine.get("equity_suspect"):
        state += ", broker equity disagrees with the ledger"

    L: list[str] = []
    L.append(f"{SUMMARY_TAG}{len(trades)} trades, {_money(pnl)} P&L, equity {equity:,.0f} -->")
    L.append(f"# Sentinel journal — {day}")
    L.append("")
    L.append(f"Mode **{status.get('mode') or settings.mode}** · engine {state} · written {to_iso(now)} · covers the {WINDOW_HOURS} hours before that.")
    L.append("")
    L.append("## Account")
    L.append("")
    L.append("| | |")
    L.append("|---|---|")
    L.append(f"| Equity | {equity:,.2f} |")
    L.append(f"| Day P&L | {_money(acct.get('day_pnl'))} ({_pct(acct.get('day_pnl_pct'))}) |")
    L.append(f"| Week P&L | {_money(acct.get('week_pnl'))} ({_pct(acct.get('week_pnl_pct'))}) |")
    start_eq = float(acct.get("starting_equity") or 0)
    L.append(f"| Since start | {_money(acct.get('total_pnl'))} ({_pct(acct.get('total_pnl_pct'))})" + (f" from {start_eq:,.0f} |" if start_eq > 0 else " |"))
    L.append(f"| Open positions | {acct.get('open_positions', 0)} · gross exposure {float(acct.get('gross_exposure_pct') or 0):.1f}% |")
    L.append("")
    L.append(f"## Trades ({len(trades)})")
    L.append("")
    if trades:
        L.append(f"Net {_money(pnl)} · {r_sum:+.2f}R · {wins} winners / {len(trades) - wins} losers · {stops} stopped out.")
        L.append("")
        L.append("| Variant | Trades | Wins | P&L | R |")
        L.append("|---|---:|---:|---:|---:|")
        for vid, v in sorted(per_variant.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
            L.append(f"| {vid} | {int(v['n'])} | {int(v['wins'])} | {_money(v['pnl'])} | {v['r']:+.2f} |")
    else:
        L.append("No trades closed in the window.")
    L.append("")
    L.append("## Population")
    L.append("")
    L.append(" · ".join(f"{k} {v}" for k, v in pop.items()) if pop else f"{len(variants)} variants in play.")
    L.append("")
    if variants:
        L.append("| Variant | Status | Allocation | Origin | Markets |")
        L.append("|---|---|---:|---|---|")
        for v in sorted(variants, key=lambda x: (-float(x.get("allocation") or 0), x["id"])):
            flag = " (control)" if v.get("is_control") else ""
            L.append(f"| {v['id']}{flag} | {v['status']} | {float(v.get('allocation') or 0) * 100:.1f}% | {v.get('origin', '')} | {', '.join(v.get('markets') or [])} |")
        L.append("")
    if lifecycle:
        L.append("## Lifecycle changes")
        L.append("")
        for e in reversed(lifecycle):
            L.append(f"- {e['ts']} {e['message']}")
        L.append("")
    if risk or errors:
        L.append("## Risk and errors")
        L.append("")
        for line in _collapse(sorted(risk + errors, key=lambda e: e["ts"])):
            L.append(f"- {line}")
        L.append("")
    L.append("## Research")
    L.append("")
    run_facts: list[dict[str, Any]] = []
    if not runs:
        L.append("No Claude session ran in the window.")
        L.append("")
    for r in reversed(runs):
        props = db.proposals(run_id=r["id"], limit=20)
        accepted = sum(1 for p in props if p["status"] == "accepted")
        run_facts.append({"kind": r["kind"], "id": r["id"], "status": r["status"], "proposals": len(props), "accepted": accepted})
        L.append(f"### {r['kind']} session #{r['id']} — {r['status']}, ${float(r.get('cost_usd') or 0):.2f}, {int(r.get('num_turns') or 0)} turns, model {r.get('model', '')}")
        L.append("")
        if r.get("error"):
            L.append(f"Error: {r['error']}")
            L.append("")
        if props:
            L.append("| Proposal | Target | Decision |")
            L.append("|---|---|---|")
            for p in props:
                L.append(f"| #{p['id']} {p['type']} | {p.get('target') or ''} | {p['status']}{': ' + p['decision_reason'] if p.get('decision_reason') else ''} |")
            L.append("")
        analysis = (r.get("analysis_md") or "").strip()
        if analysis:
            if len(analysis) > 6000:
                analysis = analysis[:6000].rstrip() + "\n\n_(truncated)_"
            L.append(_demote(analysis))
            L.append("")
    scalars = {k: v for k, v in budget.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)}
    if scalars:
        L.append("## Claude budget")
        L.append("")
        L.append("| | |")
        L.append("|---|---|")
        for k, v in scalars.items():
            L.append(f"| {k} | {v:,.2f} |" if isinstance(v, float) else f"| {k} | {v} |")
        L.append("")
    L.append("---")
    L.append("_Written by `python -m sentinel publish` from the lab database. The analyst's running notebook is [memory.md](../memory.md)._")
    L.append("")
    facts = {"day": day, "n_trades": len(trades), "pnl": pnl, "equity": equity, "runs": run_facts}
    return "\n".join(L), facts


def commit_message(facts: dict[str, Any]) -> str:
    msg = f"journal {facts['day']}: {facts['n_trades']} trades, {_money(facts['pnl'])} P&L, equity {facts['equity']:,.0f}"
    for r in facts.get("runs") or []:
        msg += f"; {r['kind']} #{r['id']}: {r['proposals']} proposals, {r['accepted']} accepted" if r["status"] == "ok" else f"; {r['kind']} #{r['id']}: {r['status']}"
    return msg


def write_index(journal_dir: Path) -> Path:
    """lab/journal/README.md: every day, newest first, with the one-line summary each page carries."""
    rows = []
    for page in sorted(journal_dir.glob("????-??-??.md"), reverse=True):
        first = page.read_text(encoding="utf-8").split("\n", 1)[0]
        summary = first[len(SUMMARY_TAG):-4].strip() if first.startswith(SUMMARY_TAG) and first.endswith("-->") else ""
        rows.append(f"- [{page.stem}]({page.name})" + (f" — {summary}" if summary else ""))
    text = "# Sentinel journal\n\nOne page per day, written by the engine after the analyst session: account, trades, population, lifecycle changes, and what Claude found.\n\n" + "\n".join(rows) + "\n"
    idx = journal_dir / "README.md"
    idx.write_text(text, encoding="utf-8", newline="\n")
    return idx


# ---- git ----------------------------------------------------------------------------------
def _git(git: str, repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run([git, "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, check=check)


def repo_root(git: str, start: Path) -> Path | None:
    try:
        r = _git(git, start, "rev-parse", "--show-toplevel", check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return Path(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None


def publish(db: Database, settings: Settings, status: dict[str, Any], now: datetime, *, log: Callable[[str], Any] = print,
            push: bool | None = None, dry_run: bool = False, lab_dir: Path | None = None, repo: Path | None = None) -> dict[str, Any]:
    """Write today's page and the index, then commit them with lab/memory.md and push.

    Returns what happened; a failed push is reported, not raised, because the commit stands and
    goes out with the next one.
    """
    cfg = settings.publish
    lab = lab_dir or settings.lab_dir
    journal_dir = lab / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    md, facts = build_journal(db, settings, status, now)
    page = journal_dir / f"{facts['day']}.md"
    page.write_text(md, encoding="utf-8", newline="\n")
    write_index(journal_dir)
    result: dict[str, Any] = {**facts, "page": str(page), "committed": False, "pushed": False}
    if dry_run:
        return result
    repo = repo or repo_root(cfg.git, lab)
    if repo is None:
        log("publish: lab is not inside a git repository; page written only")
        return {**result, "reason": "no repository"}
    paths = [journal_dir] + ([lab / "memory.md"] if (lab / "memory.md").exists() else [])
    try:
        rel = [p.resolve().relative_to(repo.resolve()).as_posix() for p in paths]
    except ValueError:
        log("publish: lab is outside the repository; page written only")
        return {**result, "reason": "lab outside repository"}
    if _git(cfg.git, repo, "check-ignore", "-q", "--", *rel, check=False).returncode == 0:
        log(f"publish: {lab.name}/ is gitignored; page written only")
        return {**result, "reason": "lab is gitignored"}
    _git(cfg.git, repo, "add", "--", *rel)
    if _git(cfg.git, repo, "diff", "--cached", "--quiet", "--", *rel, check=False).returncode == 0:
        return {**result, "reason": "no changes"}
    msg = commit_message(facts)
    _git(cfg.git, repo, "commit", "-q", "-m", msg, "--", *rel)
    sha = _git(cfg.git, repo, "rev-parse", "--short", "HEAD").stdout.strip()
    result.update(committed=True, sha=sha, message=msg)
    log(f"publish: committed {sha}: {msg}")
    if cfg.push if push is None else push:
        r = _git(cfg.git, repo, "push", "-q", cfg.remote, "HEAD", check=False)
        if r.returncode == 0:
            result["pushed"] = True
            log(f"publish: pushed to {cfg.remote}")
        else:
            result["push_error"] = (r.stderr or r.stdout).strip()[-400:]
            log(f"publish: push to {cfg.remote} failed: {result['push_error']}")
    return result
