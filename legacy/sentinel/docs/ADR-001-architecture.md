# ADR-001 — Spine vs Scaffolding, and the Two-Workflow Model

**Status:** Accepted · 2026-04-30

## Context

Sentinel Alpha runs two distinct loops in production:

1. A **scaffolding loop** that asks Claude to draft one or more exploratory
   Python modules per day. These commits inflate the repo's surface area and
   give a buffet of one-shot implementations to draw from later.
2. A **prediction loop** that runs hand-written, tested code end-to-end against
   real market data, makes directional bets per ticker, resolves them after a
   fixed horizon, and writes the results to disk for the dashboard.

Mixing the two — letting AI-generated code run as production logic — produces
unreviewed, untested behaviour that drifts daily. Keeping them entirely
separate sacrifices the value of the scaffolding (it becomes write-only).

## Decision

The codebase is split into two strict zones inside each pillar directory:

```
sentinel/{pillar}/                 ← spine: hand-written, imported, tested
sentinel/{pillar}/_generated/      ← scaffolding: AI-written, never imported
```

`sentinel/pipeline.py` may only import from spine paths. CI fails if
`_generated/` modules appear in `pipeline.py`'s import graph.

A promotion path connects them: `scripts/promote.py <pillar> <file>` moves a
generated module out of `_generated/` into the spine, after which the engineer
manually wires it into `pipeline.py` and adds a smoke test.

Two workflows enforce the cadence:

| Workflow | Schedule | Job |
|---|---|---|
| `daily_code.yml` | 09:00 UTC | scaffolding — Claude writes 1–4 stub modules, commits each |
| `daily_pipeline.yml` | 11:00 UTC | spine — runs predictions, resolves, writes dashboard data |
| `weekly_retro.yml` | 12:00 UTC Sunday | retrospective — Claude analyses week's hits and misses |
| `spine_tests.yml` | every push | pytest over spine modules |

## Consequences

**Positive**

- Generated noise is bounded; spine is provably tested.
- Daily contribution graph stays green without compromising production code.
- Promotion pathway gives generated work a non-zero chance of becoming real.

**Negative**

- Two TODO models needed: scaffolding-completed (auto, in TODO.md) and
  spine-completed (manual). Easy to confuse without discipline.
- Doubles the number of workflows the engineer must keep healthy.
- Cost: spine workflow makes one Claude + Gemini call set per ticker per day,
  ~$0.02/day at current pricing.

## Strategy comparison ledger

The Judge tracks four strategies in parallel — Claude, always-up,
always-neutral, and 5-day momentum — so the dashboard always answers the
question: "is the LLM actually adding value beyond a trivial baseline?"
