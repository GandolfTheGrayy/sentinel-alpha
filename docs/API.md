# Sentinel API contract (v1)

Base URL: `http://localhost:8787`. All responses are JSON. Timestamps are ISO-8601 UTC
strings. Money is USD floats. `R` means R-multiple (P&L divided by the initial risk of
the trade). Percentages are plain numbers (`0.23` means 0.23 %) unless the field name
ends in `_frac` (then `0.23` means 23 %).

## GET /api/status
```json
{
  "mode": "paper",
  "engine": {"running": true, "paused": false, "halted": false, "last_tick": "2026-10-09T14:31:00Z", "uptime_s": 1234, "tick_count": 99, "errors_1h": 0},
  "market": {"equities_open": true, "next_open": "…", "next_close": "…", "session": "regular", "crypto_open": true},
  "account": {"equity": 100432.1, "cash": 61000.0, "day_pnl": 231.4, "day_pnl_pct": 0.23, "week_pnl": 880.2, "week_pnl_pct": 0.88, "total_pnl": 432.1, "total_pnl_pct": 0.43, "gross_exposure_pct": 39.2, "open_positions": 6, "starting_equity": 100000.0},
  "population": {"active": 12, "incubating": 4, "probation": 1, "paused": 0, "retired": 7},
  "budget": {"weekly_cap_usd": 62.5, "spent_usd": 11.2, "remaining_usd": 51.3, "share_of_plan": 0.25, "next_analyst_run": "…", "next_strategist_run": "…"},
  "universe": {"equities": ["SPY", "QQQ"], "crypto": ["BTC/USD", "ETH/USD"]}
}
```
`mode` is one of `paper`, `live`, `sim`. `session` is one of `regular`, `pre`, `post`, `closed`.

## GET /api/equity?range=1d|1w|1m|all&variant=<id>
`[{"ts": "…", "equity": 100432.1, "cash": 61000.0, "benchmark": 100210.0}]`
`benchmark` is the buy-and-hold control rebased to the same starting equity. With
`variant`, `equity` is that variant's cumulative P&L added to its starting sleeve.

## GET /api/positions
```json
[{"lot_id": 12, "variant_id": "trend_ema#2", "family": "trend_ema", "symbol": "NVDA", "side": "long", "qty": 12, "entry_price": 231.4, "entry_ts": "…", "current_price": 234.1, "unrealized_pnl": 32.4, "unrealized_r": 0.6, "stop": 226.0, "target": 243.0, "max_hold_ts": "…", "reason": "EMA cross + ADX 31"}]
```

## GET /api/trades?limit=200&variant=&symbol=&since=
```json
[{"id": 501, "variant_id": "meanrev_bb#1", "family": "meanrev_bb", "symbol": "AAPL", "side": "long", "qty": 20, "entry_ts": "…", "entry_price": 229.1, "exit_ts": "…", "exit_price": 230.2, "pnl": 12.3, "pnl_r": 0.4, "fees": 0.0, "hold_minutes": 45, "exit_reason": "target", "mae_r": -0.3, "mfe_r": 0.9, "reason": "RSI2 oversold at lower band", "features": {"rsi_14": 28.1}}]
```
`exit_reason` is one of `target`, `stop`, `time`, `session_end`, `signal`, `kill`, `rebalance`.

## GET /api/variants
Leaderboard. Metrics are computed on closed trades (all-time and last 30).
```json
[{"id": "trend_ema#2", "family": "trend_ema", "name": "trend_ema fast=12 slow=40", "status": "active", "origin": "seed", "parent_id": null, "created_at": "…", "allocation": 0.11, "params": {"fast": 12}, "markets": ["equities"], "timeframe": "15m", "is_control": false,
  "metrics": {"n": 84, "win_rate": 0.52, "win_rate_ci": [0.41, 0.62], "expectancy_r": 0.18, "expectancy_ci": [-0.02, 0.37], "profit_factor": 1.4, "sharpe": 1.1, "max_dd_pct": 4.2, "pnl": 812.0, "avg_hold_minutes": 190, "last_30": {"n": 30, "win_rate": 0.56, "expectancy_r": 0.22, "pnl": 300.0}},
  "sparkline": [0, 12.1, 30.4],
  "open_positions": 2}]
```
`status` is one of `active`, `incubating`, `probation`, `paused`, `retired`. `origin` is one of `seed`, `optimizer`, `claude`.

## GET /api/variants/{id}
Same object plus `equity_curve: [{ts, pnl_cum}]`, `trades: [...]` (last 100), `lineage: [{id, origin, created_at, status}]`, `notes: "…"`.

## POST /api/variants/{id}/status
Body `{"status": "active" | "paused" | "retired"}`. Returns the updated variant.

## GET /api/attribution
Latest factor report (null fields when fewer than 30 closed trades exist).
```json
{"generated_at": "…", "n_trades": 640, "window_days": 30,
 "features": [{"name": "rsi_14", "importance": 0.08, "buckets": [{"label": "<30", "n": 90, "expectancy_r": 0.21, "ci": [0.05, 0.37], "win_rate": 0.55}]}],
 "logistic": [{"name": "vol_ratio", "coef": 0.31}],
 "filters": [{"feature": "hour_et", "rule": "hour_et in [9]", "n_removed": 80, "expectancy_before": 0.05, "expectancy_after": 0.12, "oos_delta": 0.06, "verdict": "candidate"}],
 "heatmap": {"rows": ["Mon", "Tue"], "cols": ["9", "10"], "values": [[0.1, -0.2]], "counts": [[12, 8]]},
 "by_family": [{"family": "trend_ema", "n": 120, "expectancy_r": 0.1, "win_rate": 0.48, "pnl": 300.0}],
 "by_regime": [{"regime": "trend/high_vol", "n": 40, "expectancy_r": 0.3, "win_rate": 0.5}],
 "by_exit_reason": [{"exit_reason": "stop", "n": 200, "expectancy_r": -1.0}]}
```
`verdict` is `candidate` or `rejected`.

## GET /api/research?limit=20
```json
[{"id": 7, "kind": "analyst", "model": "sonnet", "started_at": "…", "finished_at": "…", "status": "ok", "cost_usd": 0.42, "input_tokens": 90000, "output_tokens": 4000, "num_turns": 9, "summary": "first 200 chars…", "n_proposals": 3, "n_accepted": 1}]
```
`kind` is `analyst` or `strategist`. `status` is one of `ok`, `error`, `skipped`, `running`.

## GET /api/research/{id}
`{…run, "analysis_md": "…", "memory_update_md": "…", "proposals": [proposal…], "digest_md": "…", "error": null}`

## GET /api/research/memory
`{"memory_md": "…", "updated_at": "…"}`

## GET /api/proposals?status=&limit=100
```json
[{"id": 31, "run_id": 7, "type": "param_change", "target": "trend_ema#2", "payload": {"params": {"fast": 9}}, "rationale": "…", "status": "accepted", "decision_reason": "OOS expectancy 0.21R vs incumbent 0.08R", "backtest": {"n": 60, "expectancy_r": 0.21, "profit_factor": 1.5, "max_dd_pct": 3.1, "control_expectancy_r": -0.05, "incumbent_expectancy_r": 0.08}, "created_variant_id": "trend_ema#7", "created_at": "…"}]
```
`type` is one of `param_change`, `new_variant`, `filter`, `retire`, `new_strategy_code`. `status` is one of `pending`, `testing`, `accepted`, `rejected`.

## POST /api/proposals/{id}/decision
Body `{"decision": "accept" | "reject"}`.

## GET /api/budget
```json
{"plan": "max5", "weekly_share": 0.25, "weekly_allowance_usd_est": 250, "weekly_cap_usd": 62.5, "spent_usd": 11.2, "remaining_usd": 51.3, "week_start": "…", "runs_this_week": 6, "calibration": {"observed_pct": null, "note": "enter the weekly % from /usage to rescale"}, "models": {"analyst": "sonnet", "strategist": "opus"}, "schedule": {"analyst": "16:45 America/New_York daily", "strategist": "Sunday 10:00 America/New_York"}, "history": [{"week_start": "…", "spent_usd": 40.1, "runs": 8}]}
```

## POST /api/budget/calibrate
Body `{"observed_weekly_pct": 7.5}` — the weekly percentage shown by Claude Code `/usage` that is attributable to Sentinel. Rescales the allowance estimate.

## GET /api/events?limit=100&since=<iso>
`[{"id": 1, "ts": "…", "level": "info", "kind": "fill", "message": "…", "data": {}}]`
`level` is `info`, `warn` or `error`. `kind` is one of `fill`, `signal`, `exit`, `risk`, `research`, `tournament`, `system`, `data`.

## GET /api/stream  (Server-Sent Events)
Event types: `tick` (the `/api/status` object), `event` (one event row), `equity` (one equity point). Clients should reconnect automatically.

## POST /api/engine/{pause|resume|halt|flatten}
Returns `{"ok": true, "engine": {...}}`.

## POST /api/research/run
Body `{"kind": "analyst" | "strategist"}` → `{"queued": true}` or `{"queued": false, "reason": "budget exhausted"}`.

## GET /api/config
Sanitized config (no secrets).

## GET /api/bars?symbol=SPY&timeframe=15m&limit=300
`[{"t": "…", "o": 1, "h": 1, "l": 1, "c": 1, "v": 100}]`

## GET /api/backtests?limit=20 · POST /api/backtests
POST body `{"variant_id": "trend_ema#2"}` or `{"family": "trend_ema", "params": {}, "days": 45}` → runs synchronously (a few seconds) and returns
`{"id": 3, "family": "…", "params": {}, "days": 45, "n": 60, "expectancy_r": 0.2, "win_rate": 0.5, "profit_factor": 1.4, "max_dd_pct": 3.0, "pnl": 200.0, "equity_curve": [{"ts": "…", "pnl_cum": 1.0}], "trades": [...]}`.
