# Sentinel v2 — Architecture

Sentinel is a self-improving paper-trading lab. It trades a diversified population of
strategy *variants* around the clock (US equities during the session, crypto 24/7),
records a rich feature snapshot for every trade, measures what worked and what did not,
reallocates capital toward what is working, and uses Claude Code — on the owner's
subscription, under a hard budget — to reason about the evidence and propose the next
generation of experiments.

The guiding principle: **deterministic core, LLM at the edges.** Execution, risk,
accounting, backtesting and statistics are plain Python. Claude never places an order;
it reads digests and writes proposals, and every proposal must pass a backtest gate
before it gets a cent of (paper) capital.

```
                 +----------------------------------------------------------+
                 |                      ENGINE (asyncio)                     |
   Alpaca ------>|  data.poll -> bars cache -> features -> strategies ->     |
   (bars, quotes)|                                           risk -> broker  |--> Alpaca paper
                 |  exits (stop / target / time) each minute                 |    (or SimBroker)
                 |  ledger: per-variant lots, equity snapshots, events       |
                 +---------------+------------------------------+-----------+
                                 | SQLite                        | SSE / REST
                 +---------------v---------------+   +----------v----------+
                 |        LEARNING LOOP          |   |   FastAPI + React UI |
                 | metrics > tournament > factor |   |  overview, tournament|
                 | attribution > optimizer >     |   |  trades, factors,    |
                 | digest -> Claude lab -> gate  |   |  research, budget    |
                 +-------------------------------+   +---------------------+
```

## 1. Trading engine

* **Clock.** One `tick()` per minute. Equity strategies fire only while the US session is
  open (regular hours by default); crypto strategies fire every bar, 24/7. Each strategy
  declares a timeframe (5m, 15m, 1h, 4h, 1d); it is invoked when a bar of that timeframe
  closes.
* **Data.** Minute bars are polled from Alpaca (one multi-symbol request for equities,
  one for crypto), appended to the SQLite `bars` table and resampled in memory. On
  startup the engine backfills history (default 30 days of minute bars + 400 days of
  daily bars) so every strategy has warm indicators and every backtest runs offline.
* **Signals.** A variant emits `Signal(symbol, side, stop, target, max_hold, strength,
  reason)`. The risk layer sizes it from the variant's *sleeve* (its allocation x total
  equity), caps exposure, and sends a market order tagged with the variant id.
* **Exits** are managed by the engine, not the broker: ATR stop, R-multiple target, time
  stop, end-of-session flatten for intraday families. The same code path runs in
  backtests (using bar highs/lows) and live (using the latest price).
* **Ledger.** The engine is the source of truth for *which variant owns which lot*. The
  broker holds the net position. One direction per symbol at a time (first variant wins;
  a conflicting signal is recorded as an event, which is itself useful data).
* **Safety.** Daily loss limit pauses entries; a larger limit flattens and halts; a
  per-variant drawdown limit pauses the variant. Live trading requires
  `broker.mode: live` **and** `SENTINEL_CONFIRM_LIVE=yes` in the environment.

## 2. Strategy population

Each family is a parameterised `Strategy` subclass with a `PARAM_SPACE` (bounds for
mutation) and `DEFAULTS`. A *variant* is a family + a concrete parameter set + a
lifecycle status. The seed population:

| Family | Idea | Timeframe | Markets |
|---|---|---|---|
| `trend_ema` | EMA crossover + ADX trend filter, ATR trailing stop | 15m / 1h | both |
| `meanrev_bb` | Bollinger + RSI(2) extremes, exit at mid-band | 5m / 15m | equities |
| `breakout_orb` | Opening-range breakout with volume confirmation, flat at close | 5m | equities |
| `breakout_donchian` | Donchian channel breakout, chandelier trail | 1h | both |
| `vwap_revert` | Fade stretched moves from session VWAP | 5m | equities |
| `squeeze` | Bollinger-inside-Keltner squeeze, trade the expansion | 15m / 1h | both |
| `xs_momentum` | Cross-sectional 20/60-day momentum rotation | 1d | equities |
| `overnight` | Hold the close-to-open premium with a trend filter | 1d | equities |
| `crypto_mtf` | 4h trend, 15m pullback entry | 15m | crypto |
| `random_entry` | **Control**: random entries, standard exits | 15m | both |
| `buy_hold` | **Control**: benchmark sleeve | 1d | both |

Variants have a lifecycle: `incubating -> active -> probation -> retired`, driven by
trade-count and expectancy confidence intervals (see `learn/tournament.py`). Capital is
allocated by Thompson sampling over recent risk-adjusted performance with an exploration
floor, so every variant keeps producing data.

Claude may also propose *new families* as Python modules under
`sentinel/strategies/evolved/`. Those are loaded only after an AST allow-list check
(no I/O, no network, only numpy/pandas/the strategy base) and a passing backtest.

## 3. Feature snapshot

Every entry records ~30 numeric features: multi-horizon returns, RSI(2/14), ADX, ATR%,
Bollinger position, volume ratio, VWAP distance, distance to 50/200-day SMA, realised
vol, gap, hour/day-of-week/session, trend and vol regime labels, market context (SPY/BTC
returns and vol), signal strength, open-position count and sleeve drawdown. Exits record
the outcome (P&L, R-multiple, hold time, exit reason, MAE/MFE).

## 4. Learning loop

| Job | Cadence | What it does | Claude? |
|---|---|---|---|
| metrics | 15 min | per-variant n, win rate (Wilson CI), expectancy R (bootstrap CI), profit factor, Sharpe, max DD, rolling windows | no |
| tournament | daily + on demand | lifecycle transitions, Thompson allocation, population cap | no |
| attribution | daily | bucketed expectancy per feature with CIs, L2 logistic regression, gradient-boosting permutation importance, out-of-sample validated *filter candidates*, hour x weekday heatmaps | no |
| optimizer | weekly / on underperformance | random search in `PARAM_SPACE` with walk-forward backtests; spawns candidate variants | no |
| analyst | daily, after the US close | reads the digest, runs backtests to test ideas, writes analysis + proposals | **Sonnet** |
| strategist | weekly (Sunday) | broader review, may write a new family, updates lab memory | **Opus** |

Proposals (`param_change`, `new_variant`, `filter`, `retire`, `new_strategy_code`) are
validated, backtested walk-forward against the incumbent and the random-entry control,
and either become incubating variants or are rejected with a reason. Outcomes feed the
next digest, so Claude sees what happened to its last ideas. `lab/memory.md` is Claude's
running notebook (capped size) and is included in every session.

## 5. Budget governor

Every Claude run is a headless `claude -p` call with `--max-budget-usd`, `--max-turns`,
`--json-schema` and `--output-format json`. The reported `total_cost_usd` is logged and
summed per rolling week. The weekly cap defaults to **25 % of an estimated plan
allowance** (`claude.plan` + `claude.weekly_share`), and the UI exposes a calibration
control: enter the percentage shown by `/usage` after a week and the estimate re-scales.
When the cap is reached, research sessions are skipped (the deterministic loop keeps
learning); the daily session is also skipped when fewer than `min_new_trades` closed
since the last one.

## 6. Storage and API

SQLite (WAL) in `data/sentinel.db`: `bars`, `variants`, `allocations`, `lots`,
`trades`, `equity_snapshots`, `evaluations`, `attribution_reports`, `claude_runs`,
`proposals`, `events`, `kv`. FastAPI serves REST + an SSE stream (`docs/API.md`) and the
built React UI from `ui/dist`.

## 7. Modes

* `python -m sentinel run` — live engine (Alpaca paper by default) + API + scheduler.
* `python -m sentinel run --sim` — same engine on a synthetic market (no keys needed);
  used for demos, UI work and tests.
* `python -m sentinel backtest ...`, `research ...`, `seed`, `doctor`.
