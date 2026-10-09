"""Claude research sessions on the owner's Claude Code subscription.

A session is one headless `claude -p` call with a hard dollar cap, a turn cap, a
read-only tool surface (plus `python -m sentinel backtest ...`) and a JSON schema for
the answer. The engine validates and backtests every proposal before anything changes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinel.config import ROOT, Settings
from sentinel.data.market import MarketData
from sentinel.engine.population import create_variant
from sentinel.learn.backtest import walk_forward
from sentinel.learn.budget import BudgetGovernor
from sentinel.learn.digest import build_digest
from sentinel.learn.optimizer import control_baseline, gate, slim
from sentinel.store.db import Database, dumps, now_iso, to_iso
from sentinel.strategies.base import REGISTRY, check_evolved_source, load_evolved
from sentinel.util.clock import UTC

EVOLVED_DIR = ROOT / "sentinel" / "strategies" / "evolved"

RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analysis_md": {"type": "string", "description": "Your analysis in markdown: what is working, what is not, why, with numbers."},
        "memory_update_md": {"type": "string", "description": "Replacement text for the lab memory: durable lessons and open hypotheses (<= 4000 chars)."},
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["param_change", "new_variant", "filter", "retire", "new_strategy_code"]},
                    "target": {"type": ["string", "null"], "description": "variant id (param_change, filter, retire)"},
                    "family": {"type": ["string", "null"], "description": "family name (new_variant, new_strategy_code)"},
                    "params": {"type": ["object", "null"], "description": "parameter overrides"},
                    "markets": {"type": ["array", "null"], "items": {"type": "string"}},
                    "filter": {"type": ["object", "null"], "description": "{feature, op: in|not_in|lt|le|gt|ge|between, values|value}"},
                    "module_name": {"type": ["string", "null"], "description": "snake_case module name for new_strategy_code"},
                    "code": {"type": ["string", "null"], "description": "full Python source for new_strategy_code"},
                    "rationale": {"type": "string"},
                    "expected_effect": {"type": ["string", "null"]},
                },
                "required": ["type", "rationale"],
            },
        },
    },
    "required": ["analysis_md", "proposals"],
}

SYSTEM_PROMPT = """You are the research analyst of Sentinel, an autonomous paper-trading lab. The lab trades a population of
strategy *variants* (family + parameters) around the clock, records a feature snapshot for every trade, allocates
capital by recent risk-adjusted performance, and retires what does not work. You never place trades. Your job is to
read the evidence, explain it, and propose the next experiments. Everything you propose is backtested walk-forward
against the incumbent and against a random-entry control before it gets any capital; proposals that fail are rejected
with a reason you will see next session.

Rules
- Be quantitative and sceptical: small samples prove nothing; a result is only interesting if its confidence interval
  excludes zero and it beats the random-entry control. Prefer fewer, better-reasoned proposals (0-4) over many.
- Respect costs: crypto pays ~25 bps per side. Short-hold crypto ideas rarely survive fees.
- Use the tools to test before proposing. The backtester is deterministic and offline:
    python -m sentinel backtest --family trend_ema --params '{"fast": 9, "slow": 30, "tf": "15m"}' --days 45 --json
    python -m sentinel backtest --variant trend_ema#2 --days 45 --json
    python -m sentinel backtest --family random_entry --markets crypto --days 45 --json     (the control)
  Add --walk-forward 3 for a 3-segment walk-forward. Each run takes a few seconds. You may read strategy source under
  sentinel/strategies/ to understand a family before changing it. Do not try to run anything else.
- Proposal types: param_change (target variant + params), new_variant (family + params + markets), filter (target +
  filter on an entry feature, e.g. {"feature": "hour_et", "op": "not_in", "values": [9, 15]} or {"feature": "atr_pct",
  "op": "gt", "value": 0.4}), retire (target; only with evidence), new_strategy_code (strategist sessions only: a
  complete module that subclasses Strategy from sentinel.strategies.base, uses @register, numpy/pandas/indicators only).
- Keep analysis_md under 600 words. memory_update_md replaces the lab memory: carry forward what is still true,
  drop what is stale, record hypotheses with their status. Never include secrets or file paths in memory.
- Finish within the turn and dollar budget you were given; if you run low, stop testing and answer with what you have.
"""

ANALYST_TASK = """Today's task (daily analyst session): review what happened since the last session, update the running
picture of which families/conditions are working, and propose at most 3 small, testable changes (param tweaks or
entry filters) with the strongest evidence. Test the most promising one or two with the backtester before proposing."""

STRATEGIST_TASK = """Today's task (weekly strategist session): take the long view. Which families deserve more variants,
which should be pruned, which market conditions are consistently costly? You may propose new variants of existing
families, entry filters, retirements, and - if you have a well-founded idea that no existing family captures - one new
family as new_strategy_code (backtest it first with --family after writing it only if the tool allows; otherwise rely
on reasoning and keep it simple). At most 4 proposals."""


def _claude_cmd(settings: Settings) -> str:
    cli = settings.claude.cli
    return shutil.which(cli) or cli


def claude_available(settings: Settings) -> tuple[bool, str]:
    cmd = _claude_cmd(settings)
    if not shutil.which(cmd) and not Path(cmd).exists():
        return False, f"claude CLI not found ({settings.claude.cli}); install Claude Code or set claude.cli"
    try:
        out = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=30)
        return True, out.stdout.strip() or out.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return False, f"claude --version failed: {exc}"


def parse_cli_result(raw: str) -> dict[str, Any]:
    """The last JSON object printed by `claude -p --output-format json`."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # stream of JSON lines or trailing noise: find the last {...} that parses
    for m in reversed(list(re.finditer(r"\{[\s\S]*\}", raw))):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON in claude output: " + raw[:300])


def extract_structured(data: dict[str, Any]) -> dict[str, Any]:
    for key in ("structured_output", "structuredOutput"):
        if isinstance(data.get(key), dict):
            return data[key]
    res = data.get("result")
    if isinstance(res, dict):
        return res
    if isinstance(res, str):
        try:
            return json.loads(res)
        except json.JSONDecodeError:
            m = re.search(r"\{[\s\S]*\}", res)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
    raise ValueError("claude result did not contain the structured answer")


def run_session(db: Database, settings: Settings, md: MarketData, status: dict[str, Any], kind: str, *, manual: bool = False, now: datetime | None = None,
                log=print, dry_run: bool = False) -> int | None:
    """Run one analyst/strategist session end to end. Returns the claude_runs id (or None if skipped without a row)."""
    now = now or datetime.now(UTC)
    cfg = settings.claude.analyst if kind == "analyst" else settings.claude.strategist
    gov = BudgetGovernor(db, settings)
    ok, reason = gov.can_start(cfg.max_budget_usd, now)
    last = db.one("SELECT started_at FROM claude_runs WHERE kind=? AND status='ok' ORDER BY id DESC", (kind,))
    last_at = last["started_at"] if last else None
    if ok and not manual and cfg.min_new_trades and last_at:
        n_new = db.trade_count(since=last_at)
        if n_new < cfg.min_new_trades:
            ok, reason = False, f"only {n_new} new trades since the last {kind} session (need {cfg.min_new_trades})"
    if ok:
        avail, msg = claude_available(settings)
        if not avail:
            ok, reason = False, msg
    if ok and not manual:
        # back off after an authentication failure instead of retrying every schedule slot
        last_err = db.one("SELECT finished_at, error FROM claude_runs WHERE status='error' ORDER BY id DESC")
        if last_err and last_err.get("error") and "authenticate" in str(last_err["error"]).lower():
            try:
                from sentinel.store.db import from_iso
                age_h = (now - from_iso(last_err["finished_at"])).total_seconds() / 3600.0
            except Exception:  # noqa: BLE001
                age_h = 999.0
            if age_h < 20:
                ok, reason = False, f"claude login problem {age_h:.0f}h ago ({str(last_err['error'])[:80]}); run `claude` and sign in, then trigger a session from the dashboard"
    digest = build_digest(db, settings, md, status, kind=kind, now=now, last_run_at=last_at)
    if not ok:
        run_id = db.insert_run(kind, cfg.model, digest)
        db.update_run(run_id, status="skipped", finished_at=now_iso(), summary=reason, error=reason)
        db.add_event("research", f"{kind} session skipped: {reason}", "info", ts=to_iso(now))
        return run_id
    run_id = db.insert_run(kind, cfg.model, digest)
    run_dir = settings.lab_dir / "runs" / f"{run_id:05d}-{kind}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "digest.md").write_text(digest, encoding="utf-8")
    task = ANALYST_TASK if kind == "analyst" else STRATEGIST_TASK
    prompt = f"{digest}\n\n---\n\n{task}\n\nAnswer with the JSON object described by the schema."
    (run_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    db.add_event("research", f"{kind} session #{run_id} started ({cfg.model}, cap ${cfg.max_budget_usd:.2f}, {cfg.max_turns} turns)", "info", ts=to_iso(now))
    if dry_run:
        db.update_run(run_id, status="skipped", finished_at=now_iso(), summary="dry run: prompt written, claude not called")
        return run_id
    # the prompt goes through stdin (it can exceed the Windows command-line limit) and the system prompt through a file
    (run_dir / "system.md").write_text(SYSTEM_PROMPT, encoding="utf-8")
    cmd = [
        _claude_cmd(settings), "-p",
        "--output-format", "json", "--model", cfg.model, "--max-turns", str(cfg.max_turns), "--max-budget-usd", f"{cfg.max_budget_usd:.2f}",
        "--json-schema", json.dumps(RESULT_SCHEMA), "--system-prompt-file", str(run_dir / "system.md"),
        "--permission-mode", "dontAsk", "--tools", "Read,Glob,Grep,Bash",
        "--allowedTools", "Read", "Glob", "Grep", "Bash(python -m sentinel backtest:*)", "Bash(cat:*)",
        "--no-session-persistence", "--disable-slash-commands",
    ]
    env = {k: v for k, v in os.environ.items() if k not in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")}
    env["SENTINEL_LAB_SESSION"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    # make `python` resolve to this interpreter for the subprocess
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    started = datetime.now(UTC)
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), env=env, timeout=60 * 40)
        raw = proc.stdout
        (run_dir / "stdout.json").write_text(raw, encoding="utf-8")
        if proc.stderr:
            (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        data = parse_cli_result(raw)
    except subprocess.TimeoutExpired:
        db.update_run(run_id, status="error", finished_at=now_iso(), error="timeout after 40 minutes", raw_path=str(run_dir))
        db.add_event("research", f"{kind} session #{run_id} timed out", "error", ts=to_iso(now))
        return run_id
    except Exception as exc:  # noqa: BLE001
        db.update_run(run_id, status="error", finished_at=now_iso(), error=str(exc)[:500], raw_path=str(run_dir))
        db.add_event("research", f"{kind} session #{run_id} failed: {exc}", "error", ts=to_iso(now))
        return run_id
    usage = data.get("usage") or {}
    cost = float(data.get("total_cost_usd") or 0.0)
    fields = {
        "cost_usd": cost, "input_tokens": int(usage.get("input_tokens") or 0) + int(usage.get("cache_read_input_tokens") or 0) + int(usage.get("cache_creation_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0), "num_turns": int(data.get("num_turns") or 0), "duration_ms": int(data.get("duration_ms") or 0),
        "session_id": data.get("session_id"), "raw_path": str(run_dir), "finished_at": now_iso(),
    }
    if data.get("is_error") or data.get("subtype") not in (None, "success"):
        err = str(data.get("result") or data.get("subtype") or "unknown error")[:500]
        db.update_run(run_id, status="error", error=err, summary=err[:200], **fields)
        db.add_event("research", f"{kind} session #{run_id} error: {err[:140]}", "error", ts=to_iso(now))
        return run_id
    try:
        answer = extract_structured(data)
    except ValueError as exc:
        db.update_run(run_id, status="error", error=str(exc)[:500], **fields)
        db.add_event("research", f"{kind} session #{run_id}: {exc}", "error", ts=to_iso(now))
        return run_id
    analysis = str(answer.get("analysis_md") or "")
    memory = answer.get("memory_update_md")
    if isinstance(memory, str) and memory.strip():
        (settings.lab_dir / "memory.md").write_text(memory.strip()[:8000] + "\n", encoding="utf-8")
    proposals = [p for p in (answer.get("proposals") or []) if isinstance(p, dict)]
    db.update_run(run_id, status="ok", analysis_md=analysis, memory_update_md=memory if isinstance(memory, str) else None,
                  summary=re.sub(r"\s+", " ", analysis)[:240], **fields)
    pids = []
    for p in proposals[:6]:
        pids.append(db.insert_proposal(run_id, p.get("type", "?"), p.get("target") or p.get("family"), p, str(p.get("rationale") or "")[:1000]))
    db.add_event("research", f"{kind} session #{run_id} done: ${cost:.2f}, {fields['num_turns']} turns, {len(pids)} proposals", "info",
                 {"run_id": run_id, "cost_usd": cost}, ts=to_iso(now))
    log(f"{kind} session #{run_id}: ${cost:.2f}, {len(pids)} proposals")
    for pid in pids:
        try:
            decide_proposal(db, settings, md, pid, now=now, log=log)
        except Exception as exc:  # noqa: BLE001
            db.update_proposal(pid, status="rejected", decision_reason=f"gating error: {exc}"[:400], decided_at=now_iso())
    return run_id


# ---- gating ------------------------------------------------------------------------------
def decide_proposal(db: Database, settings: Settings, md: MarketData, pid: int, *, now: datetime | None = None, force_accept: bool = False, log=print) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    p = db.proposal(pid)
    if not p:
        raise KeyError(pid)
    payload = p["payload"]
    ptype = p["type"]
    days, splits = settings.learn.backtest_days, settings.learn.walk_forward_splits
    db.update_proposal(pid, status="testing")
    control_cache: dict[str, dict] = {}

    def reject(reason: str, bt: dict | None = None) -> dict[str, Any]:
        db.update_proposal(pid, status="rejected", decision_reason=reason[:400], backtest=bt, decided_at=now_iso())
        db.add_event("research", f"proposal #{pid} ({ptype}) rejected: {reason}", "info", {"proposal_id": pid}, ts=to_iso(now))
        return db.proposal(pid)

    def accept(reason: str, child: dict[str, Any] | None, bt: dict | None = None) -> dict[str, Any]:
        db.update_proposal(pid, status="accepted", decision_reason=reason[:400], backtest=bt, created_variant_id=child["id"] if child else None, decided_at=now_iso())
        db.add_event("research", f"proposal #{pid} ({ptype}) accepted: {reason}" + (f" -> {child['id']}" if child else ""), "info", {"proposal_id": pid}, ts=to_iso(now))
        return db.proposal(pid)

    target = db.variant(p["target"]) if p.get("target") else None
    if ptype == "retire":
        if not target:
            return reject("target variant not found")
        if target["is_control"]:
            return reject("control variants cannot be retired")
        row = db.one("SELECT AVG(pnl_r) AS e, COUNT(*) AS n FROM trades WHERE variant_id=? AND is_backtest=0", (target["id"],)) or {}
        if not force_accept and (int(row.get("n") or 0) < 15 or float(row.get("e") or 0.0) > 0):
            return reject(f"no evidence to retire: {row.get('n')} trades, {float(row.get('e') or 0):+.2f}R")
        db.update_variant(target["id"], status="retired", status_reason=f"claude proposal #{pid}: {p['rationale'][:120]}", retired_at=now_iso(), allocation=0.0)
        return accept("retired on evidence", None)

    if ptype in ("param_change", "filter"):
        if not target:
            return reject("target variant not found")
        family = target["family"]
        params = dict(target["params"])
        if ptype == "param_change":
            params.update({k: v for k, v in (payload.get("params") or {}).items() if k != "filters"})
            if params == target["params"]:
                return reject("no parameter actually changed")
        else:
            f = payload.get("filter") or {}
            if not f.get("feature") or f.get("op") not in ("in", "not_in", "lt", "le", "gt", "ge", "between"):
                return reject("malformed filter")
            params["filters"] = list(params.get("filters") or []) + [f]
        markets = target.get("markets") or REGISTRY[family].markets
        incumbent_params = target["params"]
    elif ptype == "new_variant":
        family = payload.get("family")
        if family not in REGISTRY:
            return reject(f"unknown family {family}")
        params = dict(payload.get("params") or {})
        markets = payload.get("markets") or REGISTRY[family].markets
        best = db.one("SELECT v.id, v.params FROM variants v WHERE v.family=? AND v.status!='retired' ORDER BY (SELECT AVG(pnl_r) FROM trades t WHERE t.variant_id=v.id) DESC", (family,))
        target = db.variant(best["id"]) if best else None
        incumbent_params = target["params"] if target else None
    elif ptype == "new_strategy_code":
        if not settings.claude.allow_code_proposals:
            return reject("code proposals are disabled (claude.allow_code_proposals)")
        code = payload.get("code") or ""
        name = re.sub(r"[^a-z0-9_]", "", str(payload.get("module_name") or payload.get("family") or "evolved").lower())[:40] or "evolved"
        problems = check_evolved_source(code)
        if problems:
            return reject("static check failed: " + "; ".join(problems[:4]))
        EVOLVED_DIR.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix="sentinel-evolved-")) / f"{name}.py"
        tmp.write_text(code, encoding="utf-8")
        before = set(REGISTRY)
        loaded = load_evolved(tmp.parent, log=log)
        new = [f for f in loaded if f not in before]
        if not new:
            return reject("module did not register a new strategy family")
        family = new[0]
        params = dict(payload.get("params") or {})
        markets = payload.get("markets") or REGISTRY[family].markets
        incumbent_params = None
        target = None
        dest = EVOLVED_DIR / f"{name}.py"
        if dest.exists():
            dest = EVOLVED_DIR / f"{name}_{pid}.py"
        shutil.copy(tmp, dest)
        (EVOLVED_DIR / "__init__.py").touch()
    else:
        return reject(f"unknown proposal type {ptype}")

    cls = REGISTRY[family]
    try:
        params = cls.validate_params({**cls.DEFAULTS, **params})
    except Exception as exc:  # noqa: BLE001
        return reject(f"invalid params: {exc}")
    try:
        cand = walk_forward(settings, md, family, params, days=days, splits=splits, end=now, markets=markets)
        inc = walk_forward(settings, md, family, incumbent_params, days=days, splits=splits, end=now, markets=markets) if incumbent_params else None
        ctrl = control_baseline(settings, md, markets, days, splits, now, control_cache)
    except Exception as exc:  # noqa: BLE001
        return reject(f"backtest failed: {exc}")
    bt = {**slim(cand), "incumbent_expectancy_r": inc.get("expectancy_r") if inc else None, "control_expectancy_r": ctrl.get("expectancy_r")}
    ok, reason = gate(cand, slim(inc) if inc else None, slim(ctrl))
    if not ok and not force_accept:
        return reject(reason, bt)
    child = create_variant(db, settings, family, params, origin="claude", parent_id=target["id"] if target else None, markets=markets, status="incubating",
                           notes=f"claude proposal #{pid}: {p['rationale'][:200]}")
    return accept(reason if ok else f"accepted by operator override ({reason})", child, bt)
