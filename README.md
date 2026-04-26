# Sentinel Sentiment Engine

An autonomous financial intelligence system that identifies and exploits Sentiment Arbitrage through high-frequency linguistic analysis and historical cross-referencing.

## Architecture

| Agent | Role |
|---|---|
| **Scout** | Modular scrapers for live stock data and niche sentiment sources |
| **Linguist** | LLM-powered "certainty vs. hesitation" analysis of corporate and social language |
| **Historian** | RAG pipeline querying a vector database of historical market events |
| **Judge** | Daily post-mortem recalibration comparing predicted vs. actual market moves |

## Operational Rules

- **Modular Commits** — every commit is one functional unit
- **Documentation First** — every new file includes a docstring or README entry
- **Self-Correcting** — each session reviews `TODO.md` and yesterday's backtest before choosing what to build

## Current Sprint

> Establish the base time-series database schema and the initial Live Data Microservice.

## Project Structure

```
sentinel-alpha/
├── .github/
│   └── workflows/              # GitHub Actions — daily autonomous build runs here
├── backtest_results/           # Daily Judge post-mortems (markdown reports)
├── scripts/                    # Build orchestration scripts (e.g. sentinel_daily_build.py)
├── sentinel/
│   ├── __init__.py
│   ├── scout/                  # Data ingestion agents (prices, filings, social, dev signals)
│   │   └── __init__.py
│   ├── linguist/               # LLM reasoning — certainty/hesitation, drift, regulatory whispers
│   │   └── __init__.py
│   ├── historian/              # RAG pipeline over historical market events (ChromaDB)
│   │   └── __init__.py
│   ├── judge/                  # Daily post-mortem + heuristic recalibration
│   │   └── __init__.py
│   ├── tests/                  # Unit and integration tests
│   └── docs/                   # Architecture decision records and design notes
├── README.md
└── TODO.md                     # Live backlog — read by the daily build script
```
