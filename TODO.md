# Sentinel — Task Backlog

> **Two checklist models.** Items in `## Spine Progress` are hand-edited and reflect modules actually wired into `sentinel/pipeline.py`. Items under `## Completed` are auto-appended by the daily build script — they represent AI-generated stubs in `_generated/`, not production code. Don't confuse the two.

## Spine Progress (production)

- [x] Live price fetcher (yfinance + stooq fallback) — `sentinel/scout/live_prices.py`
- [x] News headline fetcher — `sentinel/scout/news.py`
- [x] SEC EDGAR 8-K/10-Q scraper — `sentinel/scout/sec_filings.py`
- [x] Linguist certainty scorer — `sentinel/linguist/sample_score.py`
- [x] Embedding-based RAG (Gemini) — `sentinel/historian/rag_query.py`
- [x] Per-ticker prediction (Claude) — `sentinel/judge/predictor.py`
- [x] Three baseline strategies — `sentinel/judge/baselines.py`
- [x] Auto-resolver — `sentinel/judge/resolver.py`
- [x] Discord notifier — `sentinel/judge/notify.py`
- [x] Post-mortem renderer — `sentinel/judge/postmortem.py`
- [x] End-to-end pipeline orchestrator — `sentinel/pipeline.py`
- [x] Spine smoke tests — `sentinel/tests/test_spine.py`
- [x] Weekly retrospective generator — `scripts/weekly_retro.py`
- [x] Promote-to-spine CLI — `scripts/promote.py`
- [ ] Linguistic Drift detector (per-company tone shift over time)
- [ ] Earnings calendar awareness (weight predictions by event proximity)
- [ ] Embedding-based corpus expansion (ingest historical SEC + news at scale)
- [ ] Paper-trading simulator on top of Claude predictions
- [ ] Predictions.json yearly rotation
- [ ] Cache layer for SEC + yfinance to survive rate limits

## Current Sprint: Base Infrastructure

### High Priority

- [ ] Define time-series database schema (SQLite to start, swap-ready for TimescaleDB)
- [ ] Implement live price fetcher microservice (Yahoo Finance via yfinance)
- [ ] Build SEC EDGAR RSS scraper for 8-K and 10-Q filings
- [ ] Set up basic project config system (env vars + YAML config loader)

### Scout Agent

- [ ] Reddit sentiment scraper (PRAW, targeting r/wallstreetbets, r/stocks, r/investing)
- [ ] Hacker News "Ask HN" developer sentiment scraper
- [ ] GitHub repo health signal collector (stars, commit velocity, issue open rate)
- [ ] Data normalizer — unified schema across all scraper outputs

### Linguist Agent

- [ ] Base LLM prompt templates for "certainty vs. hesitation" scoring
- [ ] Corporate earnings call transcript parser
- [ ] Linguistic Drift detector — tracks tone shift over rolling 30-day window
- [ ] Regulatory Whispers detector — flags hedging language in SEC filings

### Historian Agent

- [ ] Vector database setup (ChromaDB, local)
- [ ] Historical market event ingestion pipeline
- [ ] RAG query interface — given a sentiment signal, find similar historical events
- [ ] Confidence score weighting system

### Judge Agent

- [ ] Post-mortem report generator (markdown output to backtest_results/)
- [ ] Predicted Residual vs. Actual Market Move comparator
- [ ] Heuristic update logger — tracks logic refinements over time
- [ ] Anomaly flagging system

### Tests & Docs

- [ ] Unit tests for each scraper
- [ ] Integration test for end-to-end pipeline
- [ ] Architecture decision records (ADRs) in sentinel/docs/

## Completed (AI scaffolding — `_generated/` only, not production)

### 2026-05-15
- [x] A 'tells' extractor — given a block of corporate text, uses Claude to identify s
- [x] A RAG query interface — given a current SentimentResidual, queries ChromaDB for 
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll

### 2026-05-14
- [x] A GitHub repository health signal collector measuring stars, commit velocity (co
- [x] An event schema module defining dataclasses for MarketEvent, HistoricalMatch, an
- [x] A base time-series SQLite schema module — creates tables for price history, sent

### 2026-05-13
- [x] A Predicted Residual vs. Actual Market Move comparator that calculates direction
- [x] A data normalizer that maps outputs from all scrapers into a unified SignalRecor
- [x] A daily summary printer that reads the latest post-mortem and prints a concise c
- [x] A Regulatory Whispers detector that scans SEC filings for hedging language patte

### 2026-05-12
- [x] A data normalizer that maps outputs from all scrapers into a unified SignalRecor
- [x] A Linguistic Drift detector that compares a company's current 10-Q language agai
- [x] A heuristic update logger that appends CalibrationResult entries to a JSONL file
- [x] A Reddit sentiment scraper using PRAW targeting r/wallstreetbets, r/stocks, and 

### 2026-05-11
- [x] A Reddit sentiment scraper using PRAW targeting r/wallstreetbets, r/stocks, and 

### 2026-05-10
- [x] A GitHub repository health signal collector measuring stars, commit velocity (co
- [x] A data normalizer that maps outputs from all scrapers into a unified SignalRecor
- [x] An SEC EDGAR RSS scraper that polls the 8-K and 10-Q feeds and extracts filing m
- [x] A sentiment aggregator that combines Scout signals and Linguist scores into a co

### 2026-05-09
- [x] A Hacker News scraper targeting 'Ask HN' posts about tech companies, scoring dev

### 2026-05-08
- [x] A RAG query interface — given a current SentimentResidual, queries ChromaDB for 
- [x] A Linguistic Drift detector that compares a company's current 10-Q language agai
- [x] A pytest unit test module for the config loader — tests env var overrides, missi
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll

### 2026-05-07
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll
- [x] A pytest unit test module for the config loader — tests env var overrides, missi
- [x] A sentiment aggregator that combines Scout signals and Linguist scores into a co
- [x] A historical market event ingestion pipeline that reads from a CSV of past event

### 2026-05-06
- [x] A confidence score weighting system that combines RAG similarity scores with rec
- [x] A pytest integration test that runs the Scout → Linguist pipeline end-to-end wit
- [x] A pytest unit test module for the Linguistic Drift detector — uses fixture text 

### 2026-05-05
- [x] A pytest unit test module for the config loader — tests env var overrides, missi
- [x] A config loader that reads a YAML config file and environment variables, with a 
- [x] A modular yfinance-based live price fetcher that stores OHLCV data in SQLite wit

### 2026-05-04
- [x] A pytest unit test module for the Scout price fetcher — mocks yfinance responses

### 2026-05-03
- [x] A daily summary printer that reads the latest post-mortem and prints a concise c

### 2026-05-02
- [x] A Regulatory Whispers detector that scans SEC filings for hedging language patte
- [x] An anomaly flagging system that detects when actual market moves exceed 2x the p

### 2026-05-01
- [x] A Linguistic Drift detector that compares a company's current 10-Q language agai
- [x] An earnings call transcript parser that segments text by speaker role (CEO, CFO,
- [x] A 'tells' extractor — given a block of corporate text, uses Claude to identify s

### 2026-04-30
- [x] A sentiment aggregator that combines Scout signals and Linguist scores into a co
- [x] A base time-series SQLite schema module — creates tables for price history, sent

### 2026-04-29
- [x] A post-mortem report generator that reads yesterday's PredictionRecord from SQLi
- [x] A RAG query interface — given a current SentimentResidual, queries ChromaDB for 
- [x] A historical market event ingestion pipeline that reads from a CSV of past event
- [x] An SEC EDGAR RSS scraper that polls the 8-K and 10-Q feeds and extracts filing m

### 2026-04-28
- [x] A heuristic update logger that appends CalibrationResult entries to a JSONL file
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll
- [x] A Predicted Residual vs. Actual Market Move comparator that calculates direction

### 2026-04-27
- [x] An event schema module defining dataclasses for MarketEvent, HistoricalMatch, an
- [x] A prompt template system for LLM-based 'certainty vs. hesitation' scoring of cor
- [x] A GitHub repository health signal collector measuring stars, commit velocity (co

### 2026-04-26
- [x] An anomaly flagging system that detects when actual market moves exceed 2x the p
- [x] A pytest unit test module for the Linguistic Drift detector — uses fixture text 
- [x] A data normalizer that maps outputs from all scrapers into a unified SignalRecor
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll
- [x] A Predicted Residual vs. Actual Market Move comparator that calculates direction
- [x] A ChromaDB vector database setup module — initializes the local DB, defines coll

(nothing yet — this is day one)
