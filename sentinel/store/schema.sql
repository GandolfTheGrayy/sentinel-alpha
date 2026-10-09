-- Sentinel v2 SQLite schema. Timestamps are ISO-8601 UTC strings unless noted; bar ts is epoch seconds.
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL,
  tf     TEXT NOT NULL,          -- '1m' or '1d' (other timeframes are resampled in memory)
  ts     INTEGER NOT NULL,       -- epoch seconds UTC of bar open
  o REAL NOT NULL, h REAL NOT NULL, l REAL NOT NULL, c REAL NOT NULL,
  v REAL NOT NULL DEFAULT 0,
  vwap REAL,
  PRIMARY KEY (symbol, tf, ts)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS variants (
  id TEXT PRIMARY KEY,
  family TEXT NOT NULL,
  name TEXT NOT NULL,
  params TEXT NOT NULL,          -- JSON
  status TEXT NOT NULL,          -- incubating | active | probation | paused | retired
  status_reason TEXT,
  origin TEXT NOT NULL,          -- seed | optimizer | claude
  parent_id TEXT,
  markets TEXT NOT NULL,         -- JSON list: ["equities"], ["crypto"], or both
  timeframe TEXT NOT NULL,
  is_control INTEGER NOT NULL DEFAULT 0,
  allocation REAL NOT NULL DEFAULT 0,
  notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  retired_at TEXT
);

CREATE TABLE IF NOT EXISTS allocations (
  ts TEXT NOT NULL,
  variant_id TEXT NOT NULL,
  weight REAL NOT NULL,
  PRIMARY KEY (ts, variant_id)
);

CREATE TABLE IF NOT EXISTS lots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variant_id TEXT NOT NULL,
  family TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,            -- long | short
  qty REAL NOT NULL,
  entry_ts TEXT NOT NULL,
  entry_price REAL NOT NULL,
  stop REAL,
  target REAL,
  max_hold_ts TEXT,
  flat_at_session_end INTEGER NOT NULL DEFAULT 0,
  trail_atr REAL,                -- trailing stop distance (price units), NULL = fixed stop
  risk_per_unit REAL NOT NULL,   -- |entry - stop| at entry (R unit)
  reason TEXT,
  features TEXT,                 -- JSON feature snapshot at entry
  signal TEXT,                   -- JSON raw signal
  broker_order_id TEXT,
  hwm REAL, lwm REAL,            -- high/low water marks since entry (for MAE/MFE, trailing)
  fees REAL NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS lots_open_idx ON lots(status, variant_id);
CREATE INDEX IF NOT EXISTS lots_symbol_idx ON lots(symbol, status);

CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lot_id INTEGER,
  variant_id TEXT NOT NULL,
  family TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  qty REAL NOT NULL,
  entry_ts TEXT NOT NULL,
  entry_price REAL NOT NULL,
  exit_ts TEXT NOT NULL,
  exit_price REAL NOT NULL,
  pnl REAL NOT NULL,
  pnl_r REAL NOT NULL,
  fees REAL NOT NULL DEFAULT 0,
  hold_minutes REAL NOT NULL,
  exit_reason TEXT NOT NULL,
  mae_r REAL, mfe_r REAL,
  reason TEXT,
  features TEXT,                 -- JSON
  is_backtest INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS trades_variant_idx ON trades(variant_id, exit_ts);
CREATE INDEX IF NOT EXISTS trades_exit_idx ON trades(exit_ts);

CREATE TABLE IF NOT EXISTS equity_snapshots (
  ts TEXT PRIMARY KEY,
  equity REAL NOT NULL,
  cash REAL NOT NULL,
  benchmark REAL,
  per_variant TEXT               -- JSON {variant_id: cumulative pnl}
);

CREATE TABLE IF NOT EXISTS evaluations (
  ts TEXT NOT NULL,
  variant_id TEXT NOT NULL,
  window TEXT NOT NULL,          -- 'all' | 'last_30' | ...
  metrics TEXT NOT NULL,         -- JSON
  PRIMARY KEY (ts, variant_id, window)
);

CREATE TABLE IF NOT EXISTS attribution_reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  report TEXT NOT NULL,          -- JSON
  digest_md TEXT
);

CREATE TABLE IF NOT EXISTS claude_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,            -- analyst | strategist
  model TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL,          -- running | ok | error | skipped
  cost_usd REAL NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  num_turns INTEGER NOT NULL DEFAULT 0,
  duration_ms INTEGER NOT NULL DEFAULT 0,
  summary TEXT,
  analysis_md TEXT,
  memory_update_md TEXT,
  digest_md TEXT,
  error TEXT,
  session_id TEXT,
  raw_path TEXT
);

CREATE TABLE IF NOT EXISTS proposals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  type TEXT NOT NULL,            -- param_change | new_variant | filter | retire | new_strategy_code
  target TEXT,
  payload TEXT NOT NULL,         -- JSON
  rationale TEXT,
  status TEXT NOT NULL,          -- pending | testing | accepted | rejected
  decision_reason TEXT,
  backtest TEXT,                 -- JSON
  created_variant_id TEXT,
  created_at TEXT NOT NULL,
  decided_at TEXT
);

CREATE TABLE IF NOT EXISTS backtests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  family TEXT NOT NULL,
  variant_id TEXT,
  params TEXT NOT NULL,
  days INTEGER NOT NULL,
  result TEXT NOT NULL           -- JSON
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  level TEXT NOT NULL,
  kind TEXT NOT NULL,
  message TEXT NOT NULL,
  data TEXT
);
CREATE INDEX IF NOT EXISTS events_ts_idx ON events(ts);

CREATE TABLE IF NOT EXISTS kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
