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
