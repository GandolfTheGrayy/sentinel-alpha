/**
 * TypeScript types for the Sentinel API (docs/API.md, contract v1).
 * Timestamps are ISO-8601 UTC strings. Money is USD. `R` = R-multiple.
 * Percentages are in percent units (0.23 => 0.23 %) unless the field ends in `_frac`.
 */

export type Mode = 'paper' | 'live' | 'sim'
export type Session = 'regular' | 'pre' | 'post' | 'closed'
export type VariantStatus = 'active' | 'incubating' | 'probation' | 'paused' | 'retired'
export type VariantOrigin = 'seed' | 'optimizer' | 'claude'
export type Side = 'long' | 'short'
export type ExitReason = 'target' | 'stop' | 'time' | 'session_end' | 'signal' | 'kill' | 'rebalance'
export type Market = 'equities' | 'crypto'
export type Timeframe = '5m' | '15m' | '1h' | '4h' | '1d'
export type RunKind = 'analyst' | 'strategist'
export type RunStatus = 'ok' | 'error' | 'skipped' | 'running'
export type ProposalType = 'param_change' | 'new_variant' | 'filter' | 'retire' | 'new_strategy_code'
export type ProposalStatus = 'pending' | 'testing' | 'accepted' | 'rejected'
export type EventLevel = 'info' | 'warn' | 'error'
export type EventKind = 'fill' | 'signal' | 'exit' | 'risk' | 'research' | 'tournament' | 'system' | 'data'
export type EquityRange = '1d' | '1w' | '1m' | 'all'
export type Verdict = 'candidate' | 'rejected'

export const VARIANT_STATUSES: VariantStatus[] = ['active', 'incubating', 'probation', 'paused', 'retired']
export const EXIT_REASONS: ExitReason[] = ['target', 'stop', 'time', 'session_end', 'signal', 'kill', 'rebalance']
export const PROPOSAL_STATUSES: ProposalStatus[] = ['pending', 'testing', 'accepted', 'rejected']
export const EVENT_KINDS: EventKind[] = ['fill', 'signal', 'exit', 'risk', 'research', 'tournament', 'system', 'data']

export interface EngineState {
  running: boolean
  paused: boolean
  halted: boolean
  last_tick: string | null
  uptime_s: number
  tick_count: number
  errors_1h: number
  /**
   * Sim mode only: the engine's virtual clock (ISO UTC). Every timestamp the backend emits
   * (positions, events, equity, next runs, market open/close, research runs) is in this
   * clock, which can be days behind the wall clock. Absent/null in paper and live modes.
   */
  sim_time?: string | null
}

export interface MarketState {
  equities_open: boolean
  next_open: string | null
  next_close: string | null
  session: Session
  crypto_open: boolean
}

export interface AccountState {
  equity: number
  cash: number
  day_pnl: number
  day_pnl_pct: number
  week_pnl: number
  week_pnl_pct: number
  total_pnl: number
  total_pnl_pct: number
  gross_exposure_pct: number
  open_positions: number
  starting_equity: number
}

export interface PopulationState {
  active: number
  incubating: number
  probation: number
  paused: number
  retired: number
}

export interface BudgetState {
  weekly_cap_usd: number
  spent_usd: number
  remaining_usd: number
  share_of_plan: number
  next_analyst_run: string | null
  next_strategist_run: string | null
}

export interface Universe {
  equities: string[]
  crypto: string[]
}

export interface StatusResponse {
  mode: Mode
  engine: EngineState
  market: MarketState
  account: AccountState
  population: PopulationState
  budget: BudgetState
  universe: Universe
}

export interface EquityPoint {
  ts: string
  equity: number
  cash: number
  benchmark: number
}

export interface Position {
  lot_id: number
  variant_id: string
  family: string
  symbol: string
  side: Side
  qty: number
  entry_price: number
  entry_ts: string
  current_price: number
  unrealized_pnl: number
  unrealized_r: number
  stop: number | null
  target: number | null
  max_hold_ts: string | null
  reason: string
}

/** Feature snapshot: mostly numeric, but regime/session labels may be strings. */
export type FeatureSnapshot = Record<string, number | string | boolean | null>

export interface Trade {
  id: number
  variant_id: string
  family: string
  symbol: string
  side: Side
  qty: number
  entry_ts: string
  entry_price: number
  exit_ts: string
  exit_price: number
  pnl: number
  pnl_r: number
  fees: number
  hold_minutes: number
  exit_reason: ExitReason
  mae_r: number
  mfe_r: number
  reason: string
  features: FeatureSnapshot
}

export interface TradeQuery {
  limit?: number
  variant?: string
  symbol?: string
  since?: string
}

export interface VariantMetricsWindow {
  n: number
  win_rate: number | null
  expectancy_r: number | null
  pnl: number
}

/** Metrics on closed trades. Statistics are null until the variant has closed a trade (n = 0). */
export interface VariantMetrics {
  n: number
  win_rate: number | null
  win_rate_ci: [number, number] | null
  expectancy_r: number | null
  expectancy_ci: [number, number] | null
  profit_factor: number | null
  sharpe: number | null
  max_dd_pct: number | null
  pnl: number
  avg_hold_minutes: number | null
  last_30: VariantMetricsWindow
}

/** True when the metrics are backed by at least one closed trade. */
export function hasTrades(m: VariantMetrics | null | undefined): m is VariantMetrics & { expectancy_r: number } {
  return !!m && m.n > 0 && m.expectancy_r !== null && m.expectancy_r !== undefined
}

export type ParamValue = number | string | boolean | null

export interface Variant {
  id: string
  family: string
  name: string
  status: VariantStatus
  origin: VariantOrigin
  parent_id: string | null
  created_at: string
  allocation: number
  params: Record<string, ParamValue>
  markets: Market[]
  timeframe: Timeframe | string
  is_control: boolean
  metrics: VariantMetrics
  sparkline: number[]
  open_positions: number
}

export interface PnlCurvePoint {
  ts: string
  pnl_cum: number
}

export interface LineageEntry {
  id: string
  origin: VariantOrigin
  created_at: string
  status: VariantStatus
}

export interface VariantDetail extends Variant {
  equity_curve: PnlCurvePoint[]
  trades: Trade[]
  lineage: LineageEntry[]
  notes: string
}

export type VariantStatusChange = 'active' | 'paused' | 'retired'

export interface FeatureBucket {
  label: string
  n: number
  expectancy_r: number
  ci: [number, number]
  win_rate: number
}

export interface FeatureReport {
  name: string
  importance: number
  buckets: FeatureBucket[]
}

export interface LogisticCoef {
  name: string
  coef: number
}

export interface FilterCandidate {
  feature: string
  rule: string
  n_removed: number
  expectancy_before: number
  expectancy_after: number
  oos_delta: number
  verdict: Verdict
}

export interface Heatmap {
  rows: string[]
  cols: string[]
  values: (number | null)[][]
  counts: number[][]
}

export interface FamilyStat {
  family: string
  n: number
  expectancy_r: number
  win_rate: number
  pnl: number
}

export interface RegimeStat {
  regime: string
  n: number
  expectancy_r: number
  win_rate: number
}

export interface ExitReasonStat {
  exit_reason: ExitReason
  n: number
  expectancy_r: number
}

/** Latest factor report. Fields are null when fewer than 30 closed trades exist. */
export interface AttributionReport {
  generated_at: string | null
  n_trades: number
  window_days: number
  features: FeatureReport[] | null
  logistic: LogisticCoef[] | null
  filters: FilterCandidate[] | null
  heatmap: Heatmap | null
  by_family: FamilyStat[] | null
  by_regime: RegimeStat[] | null
  by_exit_reason: ExitReasonStat[] | null
}

export interface ResearchRun {
  id: number
  kind: RunKind
  model: string
  started_at: string
  finished_at: string | null
  status: RunStatus
  cost_usd: number
  input_tokens: number
  output_tokens: number
  num_turns: number
  summary: string
  n_proposals: number
  n_accepted: number
}

export interface ProposalBacktest {
  n: number
  expectancy_r: number
  profit_factor: number
  max_dd_pct: number
  control_expectancy_r: number
  incumbent_expectancy_r: number | null
}

export interface Proposal {
  id: number
  run_id: number
  type: ProposalType
  target: string | null
  payload: Record<string, unknown>
  rationale: string
  status: ProposalStatus
  decision_reason: string | null
  backtest: ProposalBacktest | null
  created_variant_id: string | null
  created_at: string
}

export interface ResearchRunDetail extends ResearchRun {
  analysis_md: string
  memory_update_md: string
  proposals: Proposal[]
  digest_md: string
  error: string | null
}

export interface LabMemory {
  memory_md: string
  updated_at: string | null
}

export interface ProposalQuery {
  status?: ProposalStatus
  limit?: number
}

export type Decision = 'accept' | 'reject'

export interface BudgetHistoryWeek {
  week_start: string
  spent_usd: number
  runs: number
}

export interface Budget {
  plan: string
  weekly_share: number
  weekly_allowance_usd_est: number
  weekly_cap_usd: number
  spent_usd: number
  remaining_usd: number
  week_start: string
  runs_this_week: number
  calibration: { observed_pct: number | null; note: string }
  models: { analyst: string; strategist: string }
  schedule: { analyst: string; strategist: string }
  history: BudgetHistoryWeek[]
}

export interface EventRow {
  id: number
  ts: string
  level: EventLevel
  kind: EventKind
  message: string
  data: Record<string, unknown>
}

export interface EventQuery {
  limit?: number
  since?: string
}

export type EngineAction = 'pause' | 'resume' | 'halt' | 'flatten'

export interface EngineActionResponse {
  ok: boolean
  engine: EngineState
}

export type ResearchRunResponse = { queued: true } | { queued: false; reason: string }

/** Sanitized config: arbitrary nested JSON (no secrets). */
export type ConfigTree = { [key: string]: ConfigValue }
export type ConfigValue = string | number | boolean | null | ConfigValue[] | { [key: string]: ConfigValue }

export interface Bar {
  t: string
  o: number
  h: number
  l: number
  c: number
  v: number
}

export interface BacktestRequest {
  variant_id?: string
  family?: string
  params?: Record<string, ParamValue>
  days?: number
}

export interface BacktestResult {
  id: number
  family: string
  params: Record<string, ParamValue>
  days: number
  n: number
  expectancy_r: number
  win_rate: number
  profit_factor: number
  max_dd_pct: number
  pnl: number
  equity_curve: PnlCurvePoint[]
  trades: Trade[]
}

/** SSE stream payloads keyed by event type. */
export interface StreamEvents {
  tick: StatusResponse
  event: EventRow
  equity: EquityPoint
}
