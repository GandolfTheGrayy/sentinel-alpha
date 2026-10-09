# Sentinel v2

A self-improving paper-trading lab. Sentinel trades a diversified population of strategy
*variants* around the clock (US equities during the session, crypto 24/7), records a
feature snapshot for every trade, works out which conditions led to good and bad
outcomes, reallocates capital toward what is working, and uses **Claude Code on your own
subscription** — under a hard weekly budget — to reason about the evidence and propose
the next generation of experiments.

Claude never places an order. It reads a digest, runs offline backtests, and writes
proposals; every proposal is backtested walk-forward against the incumbent and a
random-entry control before it gets a cent of paper capital.

```
engine (every minute)      learning loop (deterministic)         Claude Code (budgeted)
bars -> features ->        metrics -> tournament -> factor       analyst  (daily, Sonnet)
strategies -> risk ->      attribution -> optimizer ->  digest ->  strategist (weekly, Opus)
Alpaca paper / sim         allocation + lifecycle                    -> proposals -> gate
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and
[docs/API.md](docs/API.md) for the dashboard API.

## Quick start

Requirements: Python 3.11+, Node 20+, [Claude Code](https://claude.com/claude-code) logged in
(`claude` on your PATH), and an [Alpaca](https://alpaca.markets) account for paper
trading (free). Nothing is needed for simulation mode.

```bash
# 1. Python environment
uv venv --python 3.12 .venv            # or: python -m venv .venv
.venv/Scripts/activate                 # Windows;  source .venv/bin/activate on macOS/Linux
uv pip install -e ".[dev]"             # or: pip install -e ".[dev]"

# 2. Dashboard
python -m sentinel ui-build            # npm install + build into ui/dist

# 3. Try it with no keys: a synthetic market at 120x speed
python -m sentinel --sim run --speed 120
#    -> http://127.0.0.1:8787
```

Paper trading on real data:

```bash
cp .env.example .env                   # add ALPACA_API_KEY / ALPACA_SECRET_KEY (paper keys)
cp config.example.yaml config.yaml     # optional: universe, risk, schedules, Claude plan
python -m sentinel doctor              # checks keys, the Claude CLI login, the UI build
python -m sentinel run
```

Leave it running. The engine polls minute bars, trades the population, writes every
fill and exit to `data/sentinel.db`, and the scheduled jobs (America/New_York):

| Job | When | Claude |
|---|---|---|
| metrics | every 15 min | no |
| tournament (lifecycle + reallocation) | 16:35 daily | no |
| factor attribution | 16:40 daily | no |
| analyst session | 16:45 daily | Sonnet, capped at $1.50 |
| parameter optimizer | Sat 09:00 | no |
| strategist session | Sun 10:00 | Opus, capped at $6.00 |

## The Claude budget

Every research session is a headless `claude -p` call with `--max-budget-usd`,
`--max-turns`, a read-only tool surface (plus the offline backtester) and a JSON schema
for the answer. Reported costs are summed over a rolling 7 days and compared with a cap of
**`weekly_share` (default 25 %) of an estimated plan allowance** (`claude.plan`,
default `max5`). Sessions that would exceed the cap are skipped; the deterministic loop
keeps learning regardless.

The plan allowance is an estimate in API-equivalent dollars. To calibrate it, pick a
window in which you do not use Claude Code yourself (one quiet day is enough): note the
weekly percentage shown by `/usage` at the start and at the end, then tell Sentinel how
many points its sessions moved it:

```bash
python -m sentinel calibrate --observed-pct 1.8 --hours 24
```

The same thing is available on the Settings page (7-day window) and as
`POST /api/budget/calibrate`. The cap re-scales to match; the Research page shows every
session's dollar cost so you can sanity-check the estimate.

Claude needs a login on the machine that runs Sentinel: run `claude` once and sign in, or
create a long-lived token with `claude setup-token` and put it in `.env` as
`CLAUDE_CODE_OAUTH_TOKEN`.

## Strategy population

Eleven families seed the population (see `sentinel/strategies/`): EMA trend, Bollinger
mean reversion, opening-range breakout, Donchian breakout, VWAP reversion, volatility
squeeze, cross-sectional momentum rotation, overnight premium, multi-timeframe crypto
trend, plus two controls (random entries with the standard exits, and buy-and-hold).
Variants move through `incubating -> active -> probation -> retired` on confidence
intervals, not vibes; capital is allocated by Thompson sampling with an exploration floor.

Claude's proposals can change parameters, add entry filters on any recorded feature,
retire variants, create new variants, or (weekly) write a brand-new family into
`sentinel/strategies/evolved/`, which is loaded only after an AST allow-list check and a
passing backtest. Its running notebook lives in `lab/memory.md`.

## Command line

```bash
python -m sentinel run [--sim] [--speed N] [--port 8787]
python -m sentinel backtest --family trend_ema --params '{"fast": 9}' --days 45 --json
python -m sentinel backtest --variant trend_ema#2 --walk-forward 3
python -m sentinel research --kind analyst [--dry-run]     # one session now (dry-run writes the prompt only)
python -m sentinel tournament | attribution | optimize     # run a learning job once
python -m sentinel doctor
pytest                                                     # 30-second offline test suite
```

Simulation mode (`--sim`) uses a deterministic synthetic market and its own database
(`data/sentinel-sim.db`, `lab-sim/`), so it never mixes with paper results.

## Running 24/7 on Windows

`scripts/install_task.ps1` registers a Scheduled Task that starts Sentinel at logon and
restarts it if it stops; `scripts/uninstall_task.ps1` removes it. On a server without an
interactive Claude login, use `claude setup-token`. Logs go to `logs/`.

## Safety

* Paper trading by default. Live trading needs `broker.mode: live` **and**
  `SENTINEL_CONFIRM_LIVE=yes` in the environment.
* Daily loss limits pause entries (3 %) and flatten + halt (6 %); per-variant drawdown
  pauses a variant (15 %). All tunable in `config.yaml`.
* Pause / resume / halt / flatten from the dashboard or `POST /api/engine/{action}`.

This is a research tool, not investment advice. The hypothesis under test is whether a
disciplined, self-correcting process can find and keep an edge; expect most variants to
be retired.
