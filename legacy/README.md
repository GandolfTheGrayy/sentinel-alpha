# Sentinel Sentiment Engine

An autonomous research system that hunts for **Sentiment Arbitrage** — gaps between what corporate and social language signals and what markets price in.

🟢 **Live dashboard:** [sentinel.pletkalabs.dev](https://sentinel.pletkalabs.dev)

## Architecture

| Agent | Role |
|---|---|
| **Scout** | Live OHLCV (yfinance + stooq fallback), latest news headlines, SEC EDGAR 8-K/10-Q filings |
| **Linguist** | Claude-powered "certainty vs. hesitation" scoring on filings + headlines |
| **Historian** | Embedding-based RAG (Gemini `text-embedding-004`) over a curated seed of historical market events |
| **Judge** | Daily directional predictions (5-trading-day horizon), automatic resolution against actual price moves, weekly Claude retrospective |

## Daily Loop

```
05:00 ET ── Scout fetches prices, news, SEC filings per watchlist ticker
         ── Linguist scores certainty on the combined text per ticker
         ── Historian RAG-matches against seed events
         ── Judge asks Claude for a direction + magnitude prediction
         ── Three baseline strategies (always-up, always-neutral, momentum)
            also predict, in parallel
         ── Pipeline writes docs/predictions.json + docs/data.json
         ── Resolver checks predictions ≥ 7 days old, marks HIT/MISS
         ── Discord webhook fires on high-conviction HITs and big MISSes
07:00 ET ── User checks sentinel.pletkalabs.dev — fresh predictions live
```

The dashboard shows Claude's hit rate next to the three baselines. The system is only useful if Claude meaningfully beats them — that's the working hypothesis being tested daily.

## Repo Layout

```
sentinel/
├── pipeline.py                  ← entrypoint: predict + resolve + persist
├── scout/                       ← spine modules (live_prices, news, sec_filings)
│   └── _generated/              ← AI scaffolding (never imported)
├── linguist/
│   └── _generated/
├── historian/                   ← embeddings, rag_query
│   └── _generated/
├── judge/                       ← predictor, resolver, baselines, notify, postmortem
│   └── _generated/
├── tests/test_spine.py          ← pytest smoke tests
└── docs/                        ← architecture decision records
backtest_results/                ← daily markdown post-mortems + weekly retros
docs/                            ← Vercel-served dashboard + JSON state
scripts/
├── sentinel_daily_build.py      ← AI scaffolding generator
├── weekly_retro.py              ← Sunday retrospective generator
└── promote.py                   ← move _generated/foo.py into the spine
```

### Spine vs Scaffolding

Two separate zones inside every pillar:

- **Spine** (`sentinel/{pillar}/*.py`) — hand-written, imported by `pipeline.py`, covered by `tests/test_spine.py`, touched only when you actually want behaviour to change.
- **Scaffolding** (`sentinel/{pillar}/_generated/*.py`) — Claude-authored exploratory modules from the daily build. Never imported, never run. Read them, copy ideas you like, promote interesting ones with `python scripts/promote.py <pillar> <file>`.

See [`sentinel/docs/ADR-001-architecture.md`](sentinel/docs/ADR-001-architecture.md) for rationale.

## Workflows

| Workflow | Cron | Purpose |
|---|---|---|
| `daily_code.yml` | 09:00 UTC | Scaffolding — Claude generates 1–4 stub modules into `_generated/` |
| `daily_pipeline.yml` | 11:00 UTC | Spine — predict, resolve, notify, publish |
| `weekly_retro.yml` | 12:00 UTC Sun | Claude retrospective on the past week of resolved predictions |
| `spine_tests.yml` | every push | pytest on `sentinel/tests/test_spine.py` |

## Required Secrets

| Secret | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Claude calls (predict, score, retrospective) |
| `GEMINI_API_KEY` | yes | Gemini embeddings for Historian RAG |
| `GIT_USER_NAME` | yes | Identity on autonomous commits |
| `GIT_USER_EMAIL` | yes | Must match a verified GitHub email for contributions to count |
| `SEC_USER_AGENT` | recommended | EDGAR rejects requests without one. Format: `"Your Name your@email"` |
| `DISCORD_WEBHOOK_URL` | optional | HIT/MISS pings to a Discord channel |

## Local Development

```bash
pip install anthropic google-generativeai yfinance pandas numpy requests beautifulsoup4 pyyaml pytest
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export SEC_USER_AGENT="Your Name your@email.com"
python -m sentinel.pipeline
pytest sentinel/tests/test_spine.py
```

## Status

This is an active research project, not investment advice. Predictions are exploratory and will be wrong often — the experiment is whether they're wrong less often than a coin flip and the three trivial baselines.
