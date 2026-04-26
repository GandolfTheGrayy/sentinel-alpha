# Sentinel — Task Backlog

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

## Completed

(nothing yet — this is day one)
