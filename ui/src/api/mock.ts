/**
 * Deterministic mock implementation of the Sentinel API.
 * Active when VITE_MOCK === '1' or when the backend is unreachable.
 * Data is generated once from a fixed seed; timestamps are anchored to "now" so the
 * dashboard always looks current. A fake stream emits a tick every 2 s plus events.
 */
import type { ApiClient, StreamHandlers, Unsubscribe } from './contract'
import type {
  AttributionReport,
  BacktestRequest,
  BacktestResult,
  Bar,
  Budget,
  ConfigTree,
  Decision,
  EngineAction,
  EngineActionResponse,
  EngineState,
  EquityPoint,
  EquityRange,
  EventKind,
  EventLevel,
  EventQuery,
  EventRow,
  ExitReason,
  FeatureBucket,
  FeatureReport,
  FeatureSnapshot,
  FilterCandidate,
  LabMemory,
  LineageEntry,
  Market,
  MarketState,
  ParamValue,
  Position,
  Proposal,
  ProposalQuery,
  ResearchRun,
  ResearchRunDetail,
  ResearchRunResponse,
  RunKind,
  Session,
  Side,
  StatusResponse,
  Trade,
  TradeQuery,
  Variant,
  VariantDetail,
  VariantMetrics,
  VariantOrigin,
  VariantStatus,
  VariantStatusChange,
} from './types'
import { nyParts, nyWallToInstant } from '../lib/time'

// ---------------------------------------------------------------------------
// PRNG
// ---------------------------------------------------------------------------

class Rng {
  private s: number
  constructor(seed: number) {
    this.s = seed >>> 0
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0
    let t = this.s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next()
  }
  int(lo: number, hi: number): number {
    return Math.floor(this.range(lo, hi + 1))
  }
  pick<T>(arr: readonly T[]): T {
    return arr[Math.min(arr.length - 1, Math.floor(this.next() * arr.length))] as T
  }
  normal(mean = 0, sd = 1): number {
    let u = 0
    let v = 0
    while (u === 0) u = this.next()
    while (v === 0) v = this.next()
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  chance(p: number): boolean {
    return this.next() < p
  }
}

const SEED = 20261009
const DAY = 86_400_000
const HOUR = 3_600_000
const MIN = 60_000
const HISTORY_DAYS = 45
const STARTING_EQUITY = 100_000

const NOW = Date.now()
const START = NOW - HISTORY_DAYS * DAY

const iso = (ms: number) => new Date(ms).toISOString()
const round = (n: number, d = 2) => Math.round(n * 10 ** d) / 10 ** d
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

// ---------------------------------------------------------------------------
// Universe & families
// ---------------------------------------------------------------------------

const EQUITIES = ['SPY', 'QQQ', 'AAPL', 'NVDA', 'MSFT', 'AMZN', 'TSLA', 'META', 'AMD', 'GOOGL', 'IWM', 'XLE', 'COIN']
const CRYPTO = ['BTC/USD', 'ETH/USD', 'SOL/USD']

const BASE_PRICE: Record<string, number> = {
  SPY: 612.4,
  QQQ: 545.1,
  AAPL: 231.8,
  NVDA: 182.3,
  MSFT: 512.7,
  AMZN: 219.5,
  TSLA: 412.9,
  META: 718.2,
  AMD: 214.6,
  GOOGL: 236.4,
  IWM: 243.1,
  XLE: 92.4,
  COIN: 341.7,
  'BTC/USD': 118_420,
  'ETH/USD': 4_315,
  'SOL/USD': 214.3,
}

interface FamilySpec {
  family: string
  timeframes: string[]
  markets: Market[]
  control?: boolean
  intraday: boolean
  holdMinutes: number
  params: Record<string, ParamValue>
  sd: number
  targetR: number
  reasons: string[]
}

const FAMILIES: Record<string, FamilySpec> = {
  trend_ema: {
    family: 'trend_ema',
    timeframes: ['15m', '1h'],
    markets: ['equities', 'crypto'],
    intraday: false,
    holdMinutes: 240,
    params: { fast: 12, slow: 40, adx_min: 22, atr_mult: 2.0, target_r: 2.0, max_hold_bars: 48 },
    sd: 0.95,
    targetR: 2.0,
    reasons: ['EMA cross + ADX {adx}', 'EMA pullback hold, ADX {adx}', 'Trend continuation, ADX {adx}'],
  },
  meanrev_bb: {
    family: 'meanrev_bb',
    timeframes: ['5m', '15m'],
    markets: ['equities'],
    intraday: true,
    holdMinutes: 55,
    params: { bb_len: 20, bb_std: 2.0, rsi_len: 2, rsi_buy: 8, atr_mult: 1.2, target_r: 1.0 },
    sd: 0.75,
    targetR: 1.0,
    reasons: ['RSI2 oversold at lower band', 'RSI2 overbought at upper band', 'Band pierce + RSI2 extreme'],
  },
  breakout_orb: {
    family: 'breakout_orb',
    timeframes: ['5m'],
    markets: ['equities'],
    intraday: true,
    holdMinutes: 150,
    params: { or_minutes: 30, vol_mult: 1.5, atr_mult: 1.5, target_r: 2.0 },
    sd: 1.05,
    targetR: 2.0,
    reasons: ['ORB high break, vol {vol}x', 'ORB low break, vol {vol}x'],
  },
  breakout_donchian: {
    family: 'breakout_donchian',
    timeframes: ['1h'],
    markets: ['equities', 'crypto'],
    intraday: false,
    holdMinutes: 540,
    params: { len: 20, chandelier_mult: 3.0, atr_len: 14 },
    sd: 1.1,
    targetR: 3.0,
    reasons: ['Donchian 20 break', 'Channel breakout with expansion'],
  },
  vwap_revert: {
    family: 'vwap_revert',
    timeframes: ['5m'],
    markets: ['equities'],
    intraday: true,
    holdMinutes: 40,
    params: { stretch_pct: 0.8, atr_mult: 1.0, target_r: 1.0 },
    sd: 0.7,
    targetR: 1.0,
    reasons: ['Stretched {dist}% below VWAP', 'Stretched {dist}% above VWAP'],
  },
  squeeze: {
    family: 'squeeze',
    timeframes: ['15m', '1h'],
    markets: ['equities', 'crypto'],
    intraday: false,
    holdMinutes: 200,
    params: { bb_len: 20, kc_mult: 1.5, atr_mult: 1.8 },
    sd: 1.0,
    targetR: 2.0,
    reasons: ['Squeeze release, momentum up', 'Squeeze release, momentum down'],
  },
  xs_momentum: {
    family: 'xs_momentum',
    timeframes: ['1d'],
    markets: ['equities'],
    intraday: false,
    holdMinutes: 3 * 1440,
    params: { lookback_fast: 20, lookback_slow: 60, top_n: 3 },
    sd: 1.0,
    targetR: 2.5,
    reasons: ['Top-3 momentum rotation', 'Momentum rank entry'],
  },
  overnight: {
    family: 'overnight',
    timeframes: ['1d'],
    markets: ['equities'],
    intraday: false,
    holdMinutes: 17 * 60,
    params: { trend_len: 50, min_gap_pct: -0.5 },
    sd: 0.8,
    targetR: 1.5,
    reasons: ['Close-to-open premium, trend up'],
  },
  crypto_mtf: {
    family: 'crypto_mtf',
    timeframes: ['15m'],
    markets: ['crypto'],
    intraday: false,
    holdMinutes: 300,
    params: { htf_ema: 50, ltf_pullback: 0.5, atr_mult: 1.6 },
    sd: 1.0,
    targetR: 2.0,
    reasons: ['4h trend up, 15m pullback', '4h trend down, 15m pullback'],
  },
  random_entry: {
    family: 'random_entry',
    timeframes: ['15m'],
    markets: ['equities', 'crypto'],
    control: true,
    intraday: false,
    holdMinutes: 180,
    params: { atr_mult: 1.5, target_r: 1.5 },
    sd: 0.9,
    targetR: 1.5,
    reasons: ['Random entry (control)'],
  },
  buy_hold: {
    family: 'buy_hold',
    timeframes: ['1d'],
    markets: ['equities', 'crypto'],
    control: true,
    intraday: false,
    holdMinutes: 5 * 1440,
    params: {},
    sd: 0.9,
    targetR: 3.0,
    reasons: ['Benchmark sleeve rebalance'],
  },
}

interface VariantSeed {
  id: string
  family: string
  status: VariantStatus
  origin: VariantOrigin
  parent: string | null
  ageDays: number
  allocation: number
  edge: number
  tf: string
  paramOverride?: Record<string, ParamValue>
  markets?: Market[]
}

const VARIANT_SEEDS: VariantSeed[] = [
  { id: 'trend_ema#1', family: 'trend_ema', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.11, edge: 0.12, tf: '15m' },
  { id: 'trend_ema#2', family: 'trend_ema', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.13, edge: 0.22, tf: '1h', paramOverride: { fast: 12, slow: 40 } },
  { id: 'trend_ema#5', family: 'trend_ema', status: 'incubating', origin: 'optimizer', parent: 'trend_ema#2', ageDays: 6, allocation: 0.02, edge: 0.05, tf: '1h', paramOverride: { fast: 15, slow: 55, adx_min: 25 } },
  { id: 'trend_ema#7', family: 'trend_ema', status: 'active', origin: 'claude', parent: 'trend_ema#2', ageDays: 14, allocation: 0.09, edge: 0.28, tf: '1h', paramOverride: { fast: 9, slow: 40, adx_min: 25 } },
  { id: 'meanrev_bb#1', family: 'meanrev_bb', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.1, edge: 0.15, tf: '5m' },
  { id: 'meanrev_bb#3', family: 'meanrev_bb', status: 'probation', origin: 'optimizer', parent: 'meanrev_bb#1', ageDays: 20, allocation: 0.03, edge: -0.1, tf: '15m', paramOverride: { bb_std: 2.5, rsi_buy: 12 } },
  { id: 'meanrev_bb#4', family: 'meanrev_bb', status: 'retired', origin: 'claude', parent: 'meanrev_bb#1', ageDays: 28, allocation: 0, edge: -0.3, tf: '5m', paramOverride: { bb_len: 10, bb_std: 1.5 } },
  { id: 'breakout_orb#1', family: 'breakout_orb', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.08, edge: 0.18, tf: '5m' },
  { id: 'breakout_orb#2', family: 'breakout_orb', status: 'retired', origin: 'optimizer', parent: 'breakout_orb#1', ageDays: 33, allocation: 0, edge: -0.2, tf: '5m', paramOverride: { or_minutes: 15, vol_mult: 1.0 } },
  { id: 'breakout_donchian#1', family: 'breakout_donchian', status: 'probation', origin: 'seed', parent: null, ageDays: 45, allocation: 0.03, edge: -0.05, tf: '1h' },
  { id: 'vwap_revert#1', family: 'vwap_revert', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.07, edge: 0.1, tf: '5m' },
  { id: 'vwap_revert#2', family: 'vwap_revert', status: 'incubating', origin: 'claude', parent: 'vwap_revert#1', ageDays: 4, allocation: 0.02, edge: 0.3, tf: '5m', paramOverride: { stretch_pct: 1.1 } },
  { id: 'squeeze#1', family: 'squeeze', status: 'paused', origin: 'seed', parent: null, ageDays: 45, allocation: 0, edge: 0.0, tf: '15m' },
  { id: 'squeeze#3', family: 'squeeze', status: 'incubating', origin: 'optimizer', parent: 'squeeze#1', ageDays: 5, allocation: 0.02, edge: 0.1, tf: '1h', paramOverride: { kc_mult: 1.2 } },
  { id: 'xs_momentum#1', family: 'xs_momentum', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.09, edge: 0.2, tf: '1d' },
  { id: 'overnight#1', family: 'overnight', status: 'retired', origin: 'seed', parent: null, ageDays: 45, allocation: 0, edge: -0.25, tf: '1d' },
  { id: 'crypto_mtf#1', family: 'crypto_mtf', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.1, edge: 0.14, tf: '15m' },
  { id: 'crypto_mtf#2', family: 'crypto_mtf', status: 'incubating', origin: 'claude', parent: 'crypto_mtf#1', ageDays: 3, allocation: 0.02, edge: 0.2, tf: '15m', paramOverride: { htf_ema: 100 } },
  { id: 'random_entry#1', family: 'random_entry', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.03, edge: -0.05, tf: '15m' },
  { id: 'buy_hold#1', family: 'buy_hold', status: 'active', origin: 'seed', parent: null, ageDays: 45, allocation: 0.05, edge: 0.1, tf: '1d' },
]

// ---------------------------------------------------------------------------
// Market clock helpers
// ---------------------------------------------------------------------------

function sessionAt(ms: number): { session: Session; open: boolean } {
  const p = nyParts(new Date(ms))
  if (p.weekday === 0 || p.weekday === 6) return { session: 'closed', open: false }
  const m = p.hour * 60 + p.minute
  if (m >= 570 && m < 960) return { session: 'regular', open: true }
  if (m >= 240 && m < 570) return { session: 'pre', open: false }
  if (m >= 960 && m < 1200) return { session: 'post', open: false }
  return { session: 'closed', open: false }
}

function nextOpenClose(ms: number): { next_open: string; next_close: string } {
  const base = new Date(ms)
  const p = nyParts(base)
  const m = p.hour * 60 + p.minute
  const isWeekday = (wd: number) => wd >= 1 && wd <= 5
  // next close
  let close: Date
  if (isWeekday(p.weekday) && m < 960) {
    close = nyWallToInstant(base, 0, 16, 0)
  } else {
    let d = 1
    while (!isWeekday((p.weekday + d) % 7)) d++
    close = nyWallToInstant(base, d, 16, 0)
  }
  let open: Date
  if (isWeekday(p.weekday) && m < 570) {
    open = nyWallToInstant(base, 0, 9, 30)
  } else {
    let d = 1
    while (!isWeekday((p.weekday + d) % 7)) d++
    open = nyWallToInstant(base, d, 9, 30)
  }
  return { next_open: open.toISOString(), next_close: close.toISOString() }
}

/** Move an instant onto a weekday during regular hours (9:35–15:30 ET). */
function snapToSession(ms: number, rng: Rng): number {
  let t = ms
  for (let i = 0; i < 10; i++) {
    const p = nyParts(new Date(t))
    if (p.weekday === 0) t += 1 * DAY
    else if (p.weekday === 6) t += 2 * DAY
    else break
  }
  // A quarter of entries land in the first half hour (open-driven strategies), the rest spread over the day.
  const minute = rng.chance(0.25) ? rng.int(571, 599) : rng.int(600, 930)
  return nyWallToInstant(new Date(t), 0, Math.floor(minute / 60), minute % 60).getTime()
}

// ---------------------------------------------------------------------------
// Trades
// ---------------------------------------------------------------------------

interface VariantRuntime {
  seed: VariantSeed
  spec: FamilySpec
  createdAt: number
  params: Record<string, ParamValue>
  markets: Market[]
}

function tradeCount(status: VariantStatus, rng: Rng): number {
  switch (status) {
    case 'active':
      return rng.int(26, 36)
    case 'incubating':
      return rng.int(6, 11)
    case 'probation':
      return rng.int(20, 26)
    case 'paused':
      return rng.int(14, 18)
    case 'retired':
      return rng.int(16, 22)
  }
}

function genFeatures(rng: Rng, entryMs: number, market: Market, trendFamily: boolean): FeatureSnapshot {
  const p = nyParts(new Date(entryMs))
  const sess = market === 'crypto' ? (sessionAt(entryMs).open ? 'regular' : 'crypto') : sessionAt(entryMs).session
  const trend = rng.chance(trendFamily ? 0.6 : 0.45) ? 'trend' : 'range'
  const vol = rng.chance(0.4) ? 'high_vol' : 'low_vol'
  const atrPct = Math.max(0.12, rng.normal(vol === 'high_vol' ? 1.3 : 0.6, 0.3))
  return {
    ret_5m: round(rng.normal(0, 0.12), 3),
    ret_1h: round(rng.normal(0, 0.35), 3),
    ret_1d: round(rng.normal(0.05, 1.1), 3),
    ret_5d: round(rng.normal(0.2, 2.4), 3),
    rsi_2: round(Math.min(99, Math.max(1, rng.normal(50, 30))), 1),
    rsi_14: round(Math.min(95, Math.max(5, rng.normal(50, 14))), 1),
    adx_14: round(Math.max(5, rng.normal(trend === 'trend' ? 29 : 17, 7)), 1),
    atr_pct: round(atrPct, 3),
    bb_pos: round(rng.normal(0.5, 0.4), 3),
    vol_ratio: round(Math.max(0.2, rng.normal(1.15, 0.55)), 2),
    vwap_dist_pct: round(rng.normal(0, 0.6), 3),
    dist_sma50_pct: round(rng.normal(1.5, 3.5), 3),
    dist_sma200_pct: round(rng.normal(6, 8), 3),
    rvol_20: round(Math.max(5, rng.normal(vol === 'high_vol' ? 32 : 17, 6)), 2),
    gap_pct: round(rng.normal(0, 0.5), 3),
    hour_et: p.hour,
    dow: p.weekday,
    session: sess,
    trend_regime: trend,
    vol_regime: vol,
    spy_ret_1d: round(rng.normal(0.04, 0.8), 3),
    btc_ret_1d: round(rng.normal(0.1, 2.5), 3),
    spy_vol_20: round(Math.max(6, rng.normal(14, 4)), 2),
    signal_strength: round(Math.min(1, Math.max(0.05, rng.normal(0.6, 0.22))), 3),
    open_positions: rng.int(0, 7),
    sleeve_dd_pct: round(-Math.abs(rng.normal(1.2, 1.1)), 3),
  }
}

function genTrades(rng: Rng, runtimes: VariantRuntime[]): Trade[] {
  const trades: Trade[] = []
  let nextId = 1
  for (const rt of runtimes) {
    const { seed, spec } = rt
    const n = tradeCount(seed.status, rng)
    const from = Math.max(rt.createdAt + HOUR, START)
    const endAt = seed.status === 'retired' ? NOW - 4 * DAY : seed.status === 'paused' ? NOW - 2 * DAY : NOW - 3 * HOUR
    const isTrend = ['trend_ema', 'breakout_donchian', 'squeeze', 'crypto_mtf'].includes(spec.family)
    for (let i = 0; i < n; i++) {
      const market: Market = rt.markets.length === 2 ? (rng.chance(0.65) ? 'equities' : 'crypto') : rt.markets[0]!
      let hold = Math.max(5, spec.holdMinutes * Math.exp(rng.normal(0, 0.45)))
      // Sample the entry so that entry + hold lands before "now" (long-hold families need headroom).
      const latestEntry = endAt - hold * MIN - (spec.intraday ? 0 : 2.5 * DAY)
      let entry = rng.range(from, Math.max(from + HOUR, latestEntry))
      if (market === 'equities') entry = snapToSession(entry, rng)
      const symbol = market === 'equities' ? rng.pick(EQUITIES) : rng.pick(CRYPTO)
      const side: Side = rng.chance(spec.family === 'overnight' || spec.family === 'buy_hold' || spec.family === 'xs_momentum' ? 0.95 : 0.62)
        ? 'long'
        : 'short'
      let clamped = false
      if (market === 'equities' && spec.intraday) {
        const closeAt = nyWallToInstant(new Date(entry), 0, 15, 58).getTime()
        const maxHold = (closeAt - entry) / MIN
        if (hold > maxHold) {
          hold = Math.max(3, maxHold)
          clamped = true
        }
      }
      hold = Math.round(hold)
      const exit = Math.min(entry + hold * MIN, NOW - 10 * MIN)
      const features = genFeatures(rng, entry, market, isTrend)
      let pnlR = rng.normal(seed.edge, spec.sd)
      // Inject structure so attribution has something to find.
      if (features.hour_et === 9) pnlR -= 0.45
      if ((features.vol_ratio as number) > 1.5) pnlR += 0.18
      if ((features.vol_ratio as number) < 0.7) pnlR -= 0.22
      if (isTrend && (features.adx_14 as number) > 25) pnlR += 0.15
      if (spec.family === 'meanrev_bb' && (features.rsi_14 as number) < 30) pnlR += 0.2
      if (features.dow === 5) pnlR -= 0.08
      pnlR = Math.max(-1.25, Math.min(spec.targetR * 1.6, pnlR))

      let exitReason: ExitReason
      if (pnlR >= spec.targetR * 0.97) {
        pnlR = spec.targetR * rng.range(0.97, 1.03)
        exitReason = 'target'
      } else if (pnlR <= -0.95) {
        pnlR = -rng.range(0.95, 1.12)
        exitReason = 'stop'
      } else if (clamped) exitReason = 'session_end'
      else if (rng.chance(0.03)) exitReason = 'kill'
      else if (spec.family === 'xs_momentum' || spec.family === 'buy_hold') exitReason = rng.chance(0.6) ? 'rebalance' : 'time'
      else exitReason = rng.pick(['time', 'signal', 'time', 'signal', 'time'] as const)

      const sleeve = Math.max(2500, seed.allocation * STARTING_EQUITY, 4000)
      const riskUsd = sleeve * 0.0075
      const pnl = round(pnlR * riskUsd, 2)
      const base = BASE_PRICE[symbol] ?? 100
      const entryPrice = round(base * (1 + rng.normal(0, 0.03)), market === 'crypto' && base < 1000 ? 2 : 2)
      const stopDist = entryPrice * ((features.atr_pct as number) / 100) * 1.4
      const qtyRaw = riskUsd / Math.max(stopDist, entryPrice * 0.002)
      const qty = market === 'crypto' ? round(qtyRaw, 4) : Math.max(1, Math.round(qtyRaw))
      const dir = side === 'long' ? 1 : -1
      const exitPrice = round(entryPrice + (dir * pnl) / qty, 2)
      const maeR = exitReason === 'stop' ? round(pnlR, 2) : round(Math.min(0, pnlR) - Math.abs(rng.normal(0, 0.22)), 2)
      const mfeR = exitReason === 'target' ? round(pnlR + Math.abs(rng.normal(0, 0.08)), 2) : round(Math.max(0, pnlR) + Math.abs(rng.normal(0, 0.3)), 2)
      const reasonTpl = rng.pick(spec.reasons)
      const reason = reasonTpl
        .replace('{adx}', String(Math.round(features.adx_14 as number)))
        .replace('{vol}', (features.vol_ratio as number).toFixed(1))
        .replace('{dist}', Math.abs(features.vwap_dist_pct as number).toFixed(2))

      trades.push({
        id: nextId++,
        variant_id: seed.id,
        family: spec.family,
        symbol,
        side,
        qty,
        entry_ts: iso(entry),
        entry_price: entryPrice,
        exit_ts: iso(exit),
        exit_price: exitPrice,
        pnl,
        pnl_r: round(pnlR, 2),
        fees: 0,
        hold_minutes: Math.max(1, Math.round((exit - entry) / MIN)),
        exit_reason: exitReason,
        mae_r: Math.max(-1.3, maeR),
        mfe_r: mfeR,
        reason,
        features,
      })
    }
  }
  trades.sort((a, b) => a.exit_ts.localeCompare(b.exit_ts))
  // Re-number ids chronologically.
  trades.forEach((t, i) => {
    t.id = i + 1
  })
  return trades
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0
}
function stdev(xs: number[]): number {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
}
function wilson(wins: number, n: number): [number, number] {
  if (n === 0) return [0, 0]
  const z = 1.96
  const p = wins / n
  const denom = 1 + (z * z) / n
  const center = (p + (z * z) / (2 * n)) / denom
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom
  return [round(Math.max(0, center - half), 3), round(Math.min(1, center + half), 3)]
}
function expectancyCi(rs: number[]): [number, number] {
  if (rs.length < 2) return [0, 0]
  const m = mean(rs)
  const se = stdev(rs) / Math.sqrt(rs.length)
  return [round(m - 1.96 * se, 3), round(m + 1.96 * se, 3)]
}

function computeMetrics(trades: Trade[], sleeve: number): VariantMetrics {
  const rs = trades.map((t) => t.pnl_r)
  const n = trades.length
  const wins = trades.filter((t) => t.pnl > 0)
  const grossWin = wins.reduce((a, t) => a + t.pnl, 0)
  const grossLoss = Math.abs(trades.filter((t) => t.pnl <= 0).reduce((a, t) => a + t.pnl, 0))
  const pnl = round(trades.reduce((a, t) => a + t.pnl, 0), 2)
  // Daily P&L for Sharpe (calendar days from the first exit to now, zero-filled)
  const byDay = new Map<string, number>()
  if (trades.length) {
    const first = new Date(trades[0]!.exit_ts).getTime()
    for (let d = first; d <= NOW; d += DAY) byDay.set(new Date(d).toISOString().slice(0, 10), 0)
  }
  for (const t of trades) {
    const d = t.exit_ts.slice(0, 10)
    byDay.set(d, (byDay.get(d) ?? 0) + t.pnl)
  }
  const daily = [...byDay.values()].map((v) => v / Math.max(sleeve, 1))
  const sharpe = daily.length > 2 && stdev(daily) > 0 ? round((mean(daily) / stdev(daily)) * Math.sqrt(252), 2) : 0
  // Max drawdown on cumulative pnl path
  let peak = 0
  let cum = 0
  let maxDd = 0
  for (const t of trades) {
    cum += t.pnl
    peak = Math.max(peak, cum)
    maxDd = Math.max(maxDd, peak - cum)
  }
  const last30 = trades.slice(-30)
  return {
    n,
    win_rate: n ? round(wins.length / n, 3) : 0,
    win_rate_ci: wilson(wins.length, n),
    expectancy_r: n ? round(mean(rs), 3) : 0,
    expectancy_ci: expectancyCi(rs),
    profit_factor: grossLoss > 0 ? round(grossWin / grossLoss, 2) : grossWin > 0 ? 9.99 : 0,
    sharpe,
    max_dd_pct: round((maxDd / Math.max(sleeve, 1)) * 100, 2),
    pnl,
    avg_hold_minutes: n ? Math.round(mean(trades.map((t) => t.hold_minutes))) : 0,
    last_30: {
      n: last30.length,
      win_rate: last30.length ? round(last30.filter((t) => t.pnl > 0).length / last30.length, 3) : 0,
      expectancy_r: last30.length ? round(mean(last30.map((t) => t.pnl_r)), 3) : 0,
      pnl: round(last30.reduce((a, t) => a + t.pnl, 0), 2),
    },
  }
}

function sparklineOf(trades: Trade[]): number[] {
  const cum: number[] = [0]
  let c = 0
  for (const t of trades) {
    c += t.pnl
    cum.push(round(c, 1))
  }
  if (cum.length <= 32) return cum
  const out: number[] = []
  for (let i = 0; i < 32; i++) out.push(cum[Math.round((i * (cum.length - 1)) / 31)] as number)
  return out
}

// ---------------------------------------------------------------------------
// World state
// ---------------------------------------------------------------------------

interface World {
  rng: Rng
  runtimes: VariantRuntime[]
  variants: Variant[]
  trades: Trade[]
  equity: EquityPoint[]
  positions: Position[]
  events: EventRow[]
  runs: ResearchRunDetail[]
  proposals: Proposal[]
  budget: Budget
  memory: LabMemory
  engine: EngineState
  nextEventId: number
  noise: number
  exposureFrac: number
  equityNow: number
}

function buildVariants(rng: Rng): { runtimes: VariantRuntime[]; byId: Map<string, VariantRuntime> } {
  const runtimes: VariantRuntime[] = VARIANT_SEEDS.map((seed) => {
    const spec = FAMILIES[seed.family]!
    const params: Record<string, ParamValue> = { ...spec.params, ...(seed.paramOverride ?? {}) }
    if (seed.origin === 'optimizer' && !seed.paramOverride) {
      for (const k of Object.keys(params)) {
        const v = params[k]
        if (typeof v === 'number') params[k] = round(v * rng.range(0.8, 1.25), Number.isInteger(v) ? 0 : 2)
      }
    }
    return {
      seed,
      spec,
      createdAt: NOW - seed.ageDays * DAY - rng.range(0, 6 * HOUR),
      params,
      markets: seed.markets ?? spec.markets,
    }
  })
  const byId = new Map(runtimes.map((r) => [r.seed.id, r]))
  return { runtimes, byId }
}

function variantName(rt: VariantRuntime): string {
  const keys = Object.keys(rt.params).slice(0, 2)
  const desc = keys.map((k) => `${k}=${String(rt.params[k])}`).join(' ')
  return desc ? `${rt.spec.family} ${desc}` : rt.spec.family
}

function materializeVariants(runtimes: VariantRuntime[], trades: Trade[]): Variant[] {
  return runtimes.map((rt) => {
    const vt = trades.filter((t) => t.variant_id === rt.seed.id)
    const sleeve = Math.max(2500, rt.seed.allocation * STARTING_EQUITY, 4000)
    return {
      id: rt.seed.id,
      family: rt.spec.family,
      name: variantName(rt),
      status: rt.seed.status,
      origin: rt.seed.origin,
      parent_id: rt.seed.parent,
      created_at: iso(rt.createdAt),
      allocation: rt.seed.allocation,
      params: rt.params,
      markets: rt.markets,
      timeframe: rt.seed.tf,
      is_control: !!rt.spec.control,
      metrics: computeMetrics(vt, sleeve),
      sparkline: sparklineOf(vt),
      open_positions: 0,
    }
  })
}

function buildEquity(rng: Rng, trades: Trade[]): EquityPoint[] {
  const step = 5 * MIN
  const points: EquityPoint[] = []
  let ti = 0
  let realized = 0
  let noise = 0
  let bench = Math.log(STARTING_EQUITY)
  let exposure = 0.35
  for (let t = START; t <= NOW; t += step) {
    while (ti < trades.length && new Date(trades[ti]!.exit_ts).getTime() <= t) {
      realized += trades[ti]!.pnl
      ti++
    }
    noise = 0.97 * noise + rng.normal(0, 55)
    const sess = sessionAt(t)
    bench += sess.open ? rng.normal(0.000015, 0.0011) : rng.normal(0.000002, 0.00025)
    exposure = Math.min(0.6, Math.max(0.1, exposure + rng.normal(0, 0.01)))
    const equity = round(STARTING_EQUITY + realized + noise, 2)
    points.push({
      ts: iso(t),
      equity,
      cash: round(equity * (1 - (sess.open ? exposure : exposure * 0.6)), 2),
      benchmark: round(Math.exp(bench), 2),
    })
  }
  return points
}

function buildPositions(rng: Rng, runtimes: VariantRuntime[]): Position[] {
  const active = runtimes.filter((r) => r.seed.status === 'active' && !r.spec.control)
  const out: Position[] = []
  const used = new Set<string>()
  let lot = 1180
  for (let i = 0; i < 6; i++) {
    const rt = rng.pick(active)
    const market: Market = rt.markets.length === 2 ? (rng.chance(0.6) ? 'equities' : 'crypto') : rt.markets[0]!
    let symbol = market === 'equities' ? rng.pick(EQUITIES) : rng.pick(CRYPTO)
    let guard = 0
    while (used.has(symbol) && guard++ < 10) symbol = market === 'equities' ? rng.pick(EQUITIES) : rng.pick(CRYPTO)
    used.add(symbol)
    const side: Side = rng.chance(0.7) ? 'long' : 'short'
    const base = BASE_PRICE[symbol] ?? 100
    const entryPrice = round(base * (1 + rng.normal(0, 0.004)), 2)
    const atr = entryPrice * rng.range(0.004, 0.012)
    const sleeve = rt.seed.allocation * STARTING_EQUITY
    const riskUsd = sleeve * 0.0075
    const qtyRaw = riskUsd / (atr * 1.5)
    const qty = market === 'crypto' ? round(qtyRaw, 4) : Math.max(1, Math.round(qtyRaw))
    const dir = side === 'long' ? 1 : -1
    const current = round(entryPrice * (1 + dir * rng.normal(0.002, 0.005)), 2)
    const stop = round(entryPrice - dir * atr * 1.5, 2)
    const target = round(entryPrice + dir * atr * 1.5 * rt.spec.targetR, 2)
    const entryTs = NOW - rng.range(15 * MIN, 6 * HOUR)
    const unreal = round(dir * (current - entryPrice) * qty, 2)
    out.push({
      lot_id: lot++,
      variant_id: rt.seed.id,
      family: rt.spec.family,
      symbol,
      side,
      qty,
      entry_price: entryPrice,
      entry_ts: iso(entryTs),
      current_price: current,
      unrealized_pnl: unreal,
      unrealized_r: round(unreal / riskUsd, 2),
      stop,
      target,
      max_hold_ts: iso(entryTs + rt.spec.holdMinutes * 2 * MIN),
      reason: rng.pick(rt.spec.reasons).replace('{adx}', '31').replace('{vol}', '1.8').replace('{dist}', '0.92'),
    })
  }
  return out
}

function buildEvents(trades: Trade[], runtimes: VariantRuntime[]): EventRow[] {
  const events: EventRow[] = []
  let id = 1
  const push = (ts: number, level: EventLevel, kind: EventKind, message: string, data: Record<string, unknown> = {}) => {
    events.push({ id: id++, ts: iso(ts), level, kind, message, data })
  }
  const recent = trades.filter((t) => new Date(t.exit_ts).getTime() > NOW - 36 * HOUR)
  for (const t of recent) {
    const e = new Date(t.entry_ts).getTime()
    const x = new Date(t.exit_ts).getTime()
    if (e > NOW - 36 * HOUR) {
      push(e - 20_000, 'info', 'signal', `${t.variant_id} ${t.side} ${t.symbol}: ${t.reason}`, { variant: t.variant_id, symbol: t.symbol })
      push(e, 'info', 'fill', `Filled ${t.side.toUpperCase()} ${t.qty} ${t.symbol} @ ${t.entry_price} for ${t.variant_id}`, {
        qty: t.qty,
        price: t.entry_price,
      })
    }
    push(
      x,
      t.exit_reason === 'stop' || t.exit_reason === 'kill' ? 'warn' : 'info',
      'exit',
      `${t.variant_id} closed ${t.symbol} on ${t.exit_reason}: ${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)} (${t.pnl_r >= 0 ? '+' : ''}${t.pnl_r.toFixed(2)}R)`,
      { pnl: t.pnl, pnl_r: t.pnl_r, exit_reason: t.exit_reason },
    )
  }
  push(NOW - 35 * HOUR, 'info', 'system', 'Engine started (paper mode). Backfilled 30d minute bars for 16 symbols.')
  push(NOW - 34.9 * HOUR, 'info', 'data', 'Bars cache warm: 16 symbols, 30 days, 1m + 1d')
  push(NOW - 26 * HOUR, 'info', 'tournament', 'Tournament: trend_ema#7 promoted incubating -> active (n=31, exp CI [0.04, 0.51])')
  push(NOW - 26 * HOUR + MIN, 'warn', 'tournament', 'Tournament: meanrev_bb#3 moved to probation (expectancy CI upper < 0)')
  push(NOW - 25.9 * HOUR, 'info', 'tournament', 'Thompson allocation updated: 16 sleeves, exploration floor 2%')
  push(NOW - 25 * HOUR, 'info', 'research', 'Analyst session #6 finished: 3 proposals, 1 accepted, $0.42')
  push(NOW - 25.1 * HOUR, 'info', 'research', 'Backtest gate: proposal #31 (trend_ema#2 fast=9) PASSED, OOS 0.21R vs 0.08R incumbent')
  push(NOW - 23 * HOUR, 'warn', 'risk', 'Conflicting signal ignored: squeeze#3 short NVDA while trend_ema#2 holds long')
  push(NOW - 20 * HOUR, 'warn', 'data', 'Alpaca bars poll took 4.8s (slow); retried OK')
  push(NOW - 9 * HOUR, 'error', 'data', 'Alpaca crypto quote request failed (HTTP 429); backing off 30s')
  push(NOW - 8.98 * HOUR, 'info', 'data', 'Crypto quotes recovered')
  push(NOW - 6 * HOUR, 'info', 'system', 'Attribution report regenerated (n=' + trades.length + ', window 30d)')
  push(NOW - 3 * HOUR, 'warn', 'risk', 'Daily loss watch: -0.6% intraday, limit -2.0%')
  push(NOW - 40 * MIN, 'info', 'tournament', 'Metrics refreshed for ' + runtimes.length + ' variants')
  push(NOW - 12 * MIN, 'info', 'system', 'Scheduler: analyst run scheduled 16:45 ET')
  events.sort((a, b) => a.ts.localeCompare(b.ts))
  events.forEach((e, i) => {
    e.id = i + 1
  })
  return events
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

interface BucketDef {
  name: string
  edges: number[]
  labels: string[]
}

const BUCKET_DEFS: BucketDef[] = [
  { name: 'rsi_14', edges: [30, 45, 55, 70], labels: ['<30', '30-45', '45-55', '55-70', '>70'] },
  { name: 'adx_14', edges: [15, 25, 35], labels: ['<15', '15-25', '25-35', '>35'] },
  { name: 'vol_ratio', edges: [0.7, 1.0, 1.5, 2.5], labels: ['<0.7', '0.7-1', '1-1.5', '1.5-2.5', '>2.5'] },
  { name: 'hour_et', edges: [10, 11, 12, 13, 14, 15], labels: ['9', '10', '11', '12', '13', '14', '15+'] },
  { name: 'atr_pct', edges: [0.4, 0.8, 1.4], labels: ['<0.4%', '0.4-0.8%', '0.8-1.4%', '>1.4%'] },
  { name: 'bb_pos', edges: [0, 0.25, 0.75, 1], labels: ['<0', '0-0.25', '0.25-0.75', '0.75-1', '>1'] },
  { name: 'vwap_dist_pct', edges: [-1, -0.3, 0.3, 1], labels: ['<-1%', '-1..-0.3', '-0.3..0.3', '0.3..1', '>1%'] },
  { name: 'ret_1d', edges: [-1, 0, 1], labels: ['<-1%', '-1..0', '0..1', '>1%'] },
  { name: 'rsi_2', edges: [10, 30, 70, 90], labels: ['<10', '10-30', '30-70', '70-90', '>90'] },
  { name: 'signal_strength', edges: [0.3, 0.6, 0.8], labels: ['<0.3', '0.3-0.6', '0.6-0.8', '>0.8'] },
  { name: 'spy_ret_1d', edges: [-0.5, 0, 0.5], labels: ['<-0.5%', '-0.5..0', '0..0.5', '>0.5%'] },
  { name: 'dow', edges: [2, 3, 4, 5], labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'] },
  { name: 'dist_sma50_pct', edges: [-2, 0, 2, 5], labels: ['<-2%', '-2..0', '0..2', '2..5', '>5%'] },
  { name: 'rvol_20', edges: [12, 20, 30], labels: ['<12', '12-20', '20-30', '>30'] },
]

function bucketIndex(v: number, edges: number[]): number {
  let i = 0
  while (i < edges.length && v >= edges[i]!) i++
  return i
}

function buildAttribution(rng: Rng, trades: Trade[]): AttributionReport {
  const windowTrades = trades.filter((t) => new Date(t.exit_ts).getTime() > NOW - 30 * DAY)
  const n = windowTrades.length
  if (n < 30) {
    return {
      generated_at: iso(NOW - 6 * HOUR),
      n_trades: n,
      window_days: 30,
      features: null,
      logistic: null,
      filters: null,
      heatmap: null,
      by_family: null,
      by_regime: null,
      by_exit_reason: null,
    }
  }
  const overall = mean(windowTrades.map((t) => t.pnl_r))

  const features: FeatureReport[] = BUCKET_DEFS.map((def) => {
    const groups: Trade[][] = def.labels.map(() => [])
    for (const t of windowTrades) {
      const v = t.features[def.name]
      if (typeof v !== 'number') continue
      if (def.name === 'hour_et' && (v < 9 || v > 16)) continue
      if (def.name === 'dow' && (v < 1 || v > 5)) continue
      groups[bucketIndex(v, def.edges)]!.push(t)
    }
    const buckets: FeatureBucket[] = groups.map((g, i) => {
      const e = g.length ? round(mean(g.map((t) => t.pnl_r)), 3) : 0
      return {
        label: def.labels[i]!,
        n: g.length,
        expectancy_r: e,
        ci: g.length > 1 ? expectancyCi(g.map((t) => t.pnl_r)) : [e, e],
        win_rate: g.length ? round(g.filter((t) => t.pnl > 0).length / g.length, 3) : 0,
      }
    })
    // importance proxy: n-weighted spread of bucket expectancies around the overall mean
    const spread = buckets.reduce((a, b) => a + (b.n / Math.max(n, 1)) * Math.abs(b.expectancy_r - overall), 0)
    return { name: def.name, importance: spread * rng.range(0.85, 1.15) + 0.004, buckets }
  })
  const totalImp = features.reduce((a, f) => a + f.importance, 0)
  for (const f of features) f.importance = round(f.importance / totalImp, 4)
  features.sort((a, b) => b.importance - a.importance)

  const numericNames = BUCKET_DEFS.map((d) => d.name).filter((nm) => nm !== 'dow')
  const logistic = numericNames
    .map((name) => {
      const xs = windowTrades.map((t) => t.features[name] as number)
      const ys = windowTrades.map((t) => (t.pnl > 0 ? 1 : 0))
      const mx = mean(xs)
      const my = mean(ys)
      const sx = stdev(xs)
      const sy = stdev(ys)
      let cov = 0
      for (let i = 0; i < xs.length; i++) cov += ((xs[i] as number) - mx) * ((ys[i] as number) - my)
      cov /= Math.max(1, xs.length - 1)
      const corr = sx > 0 && sy > 0 ? cov / (sx * sy) : 0
      return { name, coef: round(corr * 2.2, 3) }
    })
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
    .slice(0, 10)

  const filterDefs: { feature: string; rule: string; pred: (t: Trade) => boolean }[] = [
    { feature: 'hour_et', rule: 'hour_et in [9]', pred: (t) => t.features.hour_et === 9 },
    { feature: 'vol_ratio', rule: 'vol_ratio < 0.7', pred: (t) => (t.features.vol_ratio as number) < 0.7 },
    { feature: 'adx_14', rule: 'adx_14 < 15', pred: (t) => (t.features.adx_14 as number) < 15 },
    { feature: 'dow', rule: 'dow in [Fri]', pred: (t) => t.features.dow === 5 },
    { feature: 'spy_ret_1d', rule: 'spy_ret_1d < -0.5', pred: (t) => (t.features.spy_ret_1d as number) < -0.5 },
  ]
  const filters: FilterCandidate[] = filterDefs.map((f, i) => {
    const removed = windowTrades.filter(f.pred)
    const kept = windowTrades.filter((t) => !f.pred(t))
    const before = round(overall, 3)
    const after = round(mean(kept.map((t) => t.pnl_r)), 3)
    const inSample = after - before
    const oos = round(inSample * rng.range(0.35, 1.1) - (i >= 3 ? 0.06 : 0), 3)
    return {
      feature: f.feature,
      rule: f.rule,
      n_removed: removed.length,
      expectancy_before: before,
      expectancy_after: after,
      oos_delta: oos,
      verdict: oos > 0.02 && removed.length >= 8 ? 'candidate' : 'rejected',
    }
  })

  const rows = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
  const cols = ['9', '10', '11', '12', '13', '14', '15']
  const cells: Trade[][][] = rows.map(() => cols.map(() => []))
  for (const t of windowTrades) {
    const dow = t.features.dow as number
    const hour = t.features.hour_et as number
    if (dow < 1 || dow > 5 || hour < 9 || hour > 15) continue
    cells[dow - 1]![hour - 9]!.push(t)
  }
  const heatmap = {
    rows,
    cols,
    values: cells.map((r) => r.map((c) => (c.length >= 3 ? round(mean(c.map((t) => t.pnl_r)), 3) : null))),
    counts: cells.map((r) => r.map((c) => c.length)),
  }

  const famMap = new Map<string, Trade[]>()
  for (const t of windowTrades) famMap.set(t.family, [...(famMap.get(t.family) ?? []), t])
  const by_family = [...famMap.entries()]
    .map(([family, ts]) => ({
      family,
      n: ts.length,
      expectancy_r: round(mean(ts.map((t) => t.pnl_r)), 3),
      win_rate: round(ts.filter((t) => t.pnl > 0).length / ts.length, 3),
      pnl: round(ts.reduce((a, t) => a + t.pnl, 0), 2),
    }))
    .sort((a, b) => b.expectancy_r - a.expectancy_r)

  const regMap = new Map<string, Trade[]>()
  for (const t of windowTrades) {
    const key = `${String(t.features.trend_regime)}/${String(t.features.vol_regime)}`
    regMap.set(key, [...(regMap.get(key) ?? []), t])
  }
  const by_regime = [...regMap.entries()]
    .map(([regime, ts]) => ({
      regime,
      n: ts.length,
      expectancy_r: round(mean(ts.map((t) => t.pnl_r)), 3),
      win_rate: round(ts.filter((t) => t.pnl > 0).length / ts.length, 3),
    }))
    .sort((a, b) => b.n - a.n)

  const exitMap = new Map<ExitReason, Trade[]>()
  for (const t of windowTrades) exitMap.set(t.exit_reason, [...(exitMap.get(t.exit_reason) ?? []), t])
  const by_exit_reason = [...exitMap.entries()]
    .map(([exit_reason, ts]) => ({ exit_reason, n: ts.length, expectancy_r: round(mean(ts.map((t) => t.pnl_r)), 3) }))
    .sort((a, b) => b.n - a.n)

  return {
    generated_at: iso(NOW - 6 * HOUR),
    n_trades: n,
    window_days: 30,
    features,
    logistic,
    filters,
    heatmap,
    by_family,
    by_regime,
    by_exit_reason,
  }
}

// ---------------------------------------------------------------------------
// Research, proposals, memory, budget, config
// ---------------------------------------------------------------------------

const ANALYSIS_1 = `# Daily analyst review

## Headline
Population expectancy over the last 30 days is **+0.11R** (n=412), with the control \`random_entry#1\` at **-0.05R**. The edge is concentrated in the trend families when ADX is above 25; mean-reversion is paying only at RSI(14) extremes.

## What the attribution says
- \`hour_et = 9\` is the single worst bucket (**-0.18R**, n=61). The first 30 minutes are noisy for every intraday family; the OOS filter test holds up (+0.06R delta).
- \`vol_ratio > 1.5\` improves expectancy by roughly **+0.2R** across families. Volume confirmation is doing real work.
- Friday entries underperform modestly; the CI straddles zero, so I am *not* proposing a filter yet.

## Experiments run this session
1. Backtested \`trend_ema#2\` with \`fast=9\` (walk-forward, 45d): **0.21R** vs incumbent 0.08R, PF 1.5, max DD 3.1%.
2. Re-tested \`meanrev_bb#3\` with a wider band (\`bb_std=2.5\`): still negative. Recommend retiring.
3. Checked whether the \`hour_et in [9]\` filter hurts \`breakout_orb\`, which by construction trades the open. It does, so the filter is scoped to non-ORB families.

\`\`\`text
family         n   exp_r   pf    dd%
trend_ema     118  +0.19  1.52   4.1
meanrev_bb     96  +0.06  1.12   5.3
breakout_orb   58  +0.17  1.41   3.8
crypto_mtf     64  +0.13  1.33   4.9
random_entry   30  -0.05  0.93   6.2
\`\`\`

## Proposals
See the proposals list below. The parameter change on \`trend_ema#2\` is the highest-conviction item.
`

const ANALYSIS_2 = `# Weekly strategist review

## Population health
Sixteen live sleeves, four incubating. Capital is drifting toward \`trend_ema\` and \`crypto_mtf\`, which is consistent with the regime (SPY 20d realised vol 14%, BTC trending). The overnight family has been retired after 38 trades at **-0.25R**: the close-to-open premium has not shown up in this sample and the trend filter did not help.

## Structural observations
- The lab is **short on short-side evidence**. 71% of trades are long. I added a note to memory to favour symmetric experiments.
- Session-end exits account for 14% of closes and are roughly break-even. Intraday families would benefit from a *time-of-day aware* target.
- Two incubating optimizer children of \`squeeze#1\` are near-duplicates (\`kc_mult\` 1.2 vs 1.3). The optimizer should enforce a minimum parameter distance.

## New family proposal
I drafted \`gap_fade\`: fade opening gaps larger than 1.2 ATR when the gap is *against* the 50-day trend, flat at close. Walk-forward backtest (60d): **n=44, +0.24R, PF 1.6, DD 2.9%**. Submitted as \`new_strategy_code\` for the AST gate.

## Memory updates
- Record the \`hour_et = 9\` finding as *confirmed*.
- Mark \`overnight\` as retired with reason.
`

const ANALYSIS_3 = `# Daily analyst review

## Headline
Quiet session: 11 new closed trades since the last run, all intraday. Nothing crosses the proposal bar today.

## Notes
- \`vwap_revert#2\` (incubating, Claude origin) is 6-for-8 but n is far too small to say anything. Leave it.
- \`breakout_donchian#1\` remains on probation; expectancy CI is **[-0.41, 0.12]**. One more losing week and the tournament will retire it on its own.
- Data note: the crypto 429 back-off at 03:14 ET cost one 15m bar for ETH/USD; the engine back-filled it correctly.

*No proposals this session.*
`

const MEMORY_MD = `# Lab memory

_Last updated by strategist run #4._

## Confirmed findings
- **First-30-minutes effect**: entries at \`hour_et = 9\` are -0.18R vs population; confirmed OOS twice. Filter applied to all intraday families except \`breakout_orb\`.
- **Volume confirmation** (\`vol_ratio > 1.5\`) adds about +0.2R for breakout and trend families. Not useful for mean reversion.
- **ADX gate** at 25 is the right neighbourhood for \`trend_ema\`; 22 lets in too much chop, 30 starves the family.

## Open questions
- Does \`meanrev_bb\` have any edge outside RSI(14) < 30? Current evidence says no.
- Short-side performance: only 29% of trades are short; too few to judge. Prefer symmetric experiments.
- Is \`crypto_mtf\` edge real or a BTC trend artefact? Compare against \`buy_hold#1\` crypto sleeve over the next 2 weeks.

## Retired ideas (do not re-propose)
- \`overnight\` family (38 trades, -0.25R, no premium in sample).
- \`meanrev_bb\` with \`bb_len=10\`: too many signals, negative expectancy.
- ORB with 15-minute range: far noisier than 30 minutes.

## House rules
1. Proposals need a walk-forward backtest with n >= 40.
2. Never propose a filter that removes more than 30% of trades.
3. Prefer one change per proposal so attribution stays clean.
`

function buildResearch(rng: Rng): { runs: ResearchRunDetail[]; proposals: Proposal[] } {
  const runAt = (daysAgo: number, hour: number, minute: number) => nyWallToInstant(new Date(NOW - daysAgo * DAY), 0, hour, minute).getTime()
  const mk = (
    id: number,
    kind: RunKind,
    status: ResearchRunDetail['status'],
    start: number,
    durationMin: number,
    cost: number,
    inTok: number,
    outTok: number,
    turns: number,
    analysis: string,
    summary: string,
    error: string | null = null,
  ): ResearchRunDetail => ({
    id,
    kind,
    model: kind === 'analyst' ? 'sonnet' : 'opus',
    started_at: iso(start),
    finished_at: iso(start + durationMin * MIN),
    status,
    cost_usd: cost,
    input_tokens: inTok,
    output_tokens: outTok,
    num_turns: turns,
    summary,
    n_proposals: 0,
    n_accepted: 0,
    analysis_md: analysis,
    memory_update_md: status === 'ok' ? '- Noted ' + (kind === 'analyst' ? 'daily findings' : 'weekly review') + ' in lab memory.' : '',
    proposals: [],
    digest_md: `## Digest\n\nClosed trades since last run: **${rng.int(9, 28)}**. Population expectancy 30d: +0.11R. Budget remaining before run: $${(62.5 - cost * 8).toFixed(2)}.`,
    error,
  })

  const runs: ResearchRunDetail[] = [
    mk(6, 'analyst', 'ok', runAt(1, 16, 45), 7, 0.42, 91_400, 4_120, 9, ANALYSIS_1, 'Population +0.11R (n=412); hour_et=9 is the worst bucket; trend_ema#2 fast=9 backtests at 0.21R vs 0.08R incumbent.'),
    mk(5, 'analyst', 'skipped', runAt(2, 16, 45), 0, 0, 0, 0, 0, '', 'Skipped: only 4 closed trades since the last session (min_new_trades=8).'),
    mk(4, 'strategist', 'ok', runAt(3, 10, 0), 19, 2.86, 412_000, 11_900, 23, ANALYSIS_2, 'Weekly review: capital drifting to trend families; overnight retired; drafted gap_fade family (+0.24R walk-forward).'),
    mk(3, 'analyst', 'ok', runAt(4, 16, 45), 5, 0.31, 64_200, 2_300, 6, ANALYSIS_3, 'Quiet session, 11 new trades, no proposals. breakout_donchian#1 still on probation.'),
    mk(2, 'analyst', 'error', runAt(5, 16, 45), 3, 0.12, 22_000, 400, 2, '', 'Run aborted: claude -p exited 1 (max turns reached before JSON output).', 'claude -p exited with code 1: output did not match --json-schema after 2 attempts'),
    mk(1, 'analyst', 'ok', runAt(6, 16, 45), 8, 0.47, 102_800, 4_900, 10, ANALYSIS_1.replace('+0.11R', '+0.09R').replace('(n=412)', '(n=361)'), 'Population +0.09R (n=361); proposed retiring meanrev_bb#4 and a vol_ratio filter for breakout_orb.'),
  ]

  const proposals: Proposal[] = [
    {
      id: 31,
      run_id: 6,
      type: 'param_change',
      target: 'trend_ema#2',
      payload: { params: { fast: 9, slow: 40, adx_min: 25 } },
      rationale: 'Faster EMA improves entry timing in trending regimes; ADX 25 gate removes chop. Walk-forward OOS 0.21R vs 0.08R.',
      status: 'accepted',
      decision_reason: 'OOS expectancy 0.21R vs incumbent 0.08R; control -0.05R; n=60 >= 40',
      backtest: { n: 60, expectancy_r: 0.21, profit_factor: 1.5, max_dd_pct: 3.1, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.08 },
      created_variant_id: 'trend_ema#7',
      created_at: iso(runAt(1, 16, 51)),
    },
    {
      id: 32,
      run_id: 6,
      type: 'filter',
      target: 'meanrev_bb#1',
      payload: { rule: 'hour_et not in [9]', scope: 'intraday_families_except_orb' },
      rationale: 'First 30 minutes are -0.18R for mean reversion; removing them lifts expectancy with 12% fewer trades.',
      status: 'pending',
      decision_reason: null,
      backtest: { n: 71, expectancy_r: 0.14, profit_factor: 1.31, max_dd_pct: 4.0, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.06 },
      created_variant_id: null,
      created_at: iso(runAt(1, 16, 52)),
    },
    {
      id: 33,
      run_id: 6,
      type: 'retire',
      target: 'meanrev_bb#3',
      payload: { reason: 'expectancy CI upper bound below zero after 24 trades' },
      rationale: 'Wider band variant has not produced a positive week; parent already covers the idea.',
      status: 'pending',
      decision_reason: null,
      backtest: null,
      created_variant_id: null,
      created_at: iso(runAt(1, 16, 52)),
    },
    {
      id: 28,
      run_id: 4,
      type: 'new_strategy_code',
      target: null,
      payload: { module: 'gap_fade', loc: 142, ast_check: 'passed' },
      rationale: 'Fade 1.2+ ATR opening gaps against the 50-day trend; flat at close. 60d walk-forward +0.24R.',
      status: 'testing',
      decision_reason: null,
      backtest: { n: 44, expectancy_r: 0.24, profit_factor: 1.6, max_dd_pct: 2.9, control_expectancy_r: -0.05, incumbent_expectancy_r: null },
      created_variant_id: null,
      created_at: iso(runAt(3, 10, 17)),
    },
    {
      id: 27,
      run_id: 4,
      type: 'new_variant',
      target: 'crypto_mtf#1',
      payload: { family: 'crypto_mtf', params: { htf_ema: 100, ltf_pullback: 0.5, atr_mult: 1.6 } },
      rationale: 'Slower higher-timeframe trend filter should cut whipsaws on ETH; parent is 0.14R with 28% of losses from HTF flips.',
      status: 'accepted',
      decision_reason: 'Backtest 0.19R (n=52) beats incumbent 0.14R and control; spawned as incubating',
      backtest: { n: 52, expectancy_r: 0.19, profit_factor: 1.42, max_dd_pct: 4.4, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.14 },
      created_variant_id: 'crypto_mtf#2',
      created_at: iso(runAt(3, 10, 18)),
    },
    {
      id: 26,
      run_id: 4,
      type: 'param_change',
      target: 'squeeze#1',
      payload: { params: { kc_mult: 1.2 } },
      rationale: 'Tighter Keltner band produces earlier squeeze releases.',
      status: 'rejected',
      decision_reason: 'Backtest 0.03R (n=47) does not beat incumbent 0.00R by the required 0.1R margin',
      backtest: { n: 47, expectancy_r: 0.03, profit_factor: 1.04, max_dd_pct: 5.8, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.0 },
      created_variant_id: null,
      created_at: iso(runAt(3, 10, 19)),
    },
    {
      id: 22,
      run_id: 1,
      type: 'retire',
      target: 'meanrev_bb#4',
      payload: { reason: '-0.3R after 20 trades; bb_len=10 is noise' },
      rationale: 'Short Bollinger window generates many marginal signals; expectancy decisively negative.',
      status: 'accepted',
      decision_reason: 'Accepted by operator; variant retired',
      backtest: null,
      created_variant_id: null,
      created_at: iso(runAt(6, 16, 53)),
    },
    {
      id: 23,
      run_id: 1,
      type: 'filter',
      target: 'breakout_orb#1',
      payload: { rule: 'vol_ratio >= 1.5' },
      rationale: 'Volume confirmation is the strongest single feature for ORB; the rule removes 22% of trades.',
      status: 'rejected',
      decision_reason: 'Removes 31% of trades in OOS window (limit 30%); re-propose with a lower threshold',
      backtest: { n: 41, expectancy_r: 0.26, profit_factor: 1.7, max_dd_pct: 2.6, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.17 },
      created_variant_id: null,
      created_at: iso(runAt(6, 16, 54)),
    },
    {
      id: 24,
      run_id: 1,
      type: 'new_variant',
      target: 'vwap_revert#1',
      payload: { family: 'vwap_revert', params: { stretch_pct: 1.1, atr_mult: 1.0, target_r: 1.0 } },
      rationale: 'Larger stretch threshold trades fewer but cleaner reversions.',
      status: 'accepted',
      decision_reason: 'Backtest 0.22R (n=48) vs incumbent 0.10R; spawned as incubating',
      backtest: { n: 48, expectancy_r: 0.22, profit_factor: 1.48, max_dd_pct: 3.3, control_expectancy_r: -0.05, incumbent_expectancy_r: 0.1 },
      created_variant_id: 'vwap_revert#2',
      created_at: iso(runAt(6, 16, 55)),
    },
  ]

  for (const r of runs) {
    r.proposals = proposals.filter((p) => p.run_id === r.id)
    r.n_proposals = r.proposals.length
    r.n_accepted = r.proposals.filter((p) => p.status === 'accepted').length
  }
  return { runs, proposals }
}

function buildBudget(runs: ResearchRun[]): Budget {
  const weekStart = (() => {
    const p = nyParts(new Date(NOW))
    const daysSinceMonday = (p.weekday + 6) % 7
    return nyWallToInstant(new Date(NOW), -daysSinceMonday, 0, 0).getTime()
  })()
  const thisWeek = runs.filter((r) => new Date(r.started_at).getTime() >= weekStart)
  const spent = round(thisWeek.reduce((a, r) => a + r.cost_usd, 0), 2)
  const history = []
  const spends = [38.4, 44.9, 51.2, 47.7, 55.3, 40.1, 58.9, 49.6]
  for (let i = 8; i >= 1; i--) {
    history.push({ week_start: iso(weekStart - i * 7 * DAY), spent_usd: spends[8 - i]!, runs: 7 + ((i * 3) % 3) })
  }
  history.push({ week_start: iso(weekStart), spent_usd: spent, runs: thisWeek.length })
  return {
    plan: 'max5',
    weekly_share: 0.25,
    weekly_allowance_usd_est: 250,
    weekly_cap_usd: 62.5,
    spent_usd: spent,
    remaining_usd: round(62.5 - spent, 2),
    week_start: iso(weekStart),
    runs_this_week: thisWeek.length,
    calibration: { observed_pct: null, note: 'enter the weekly % from /usage to rescale' },
    models: { analyst: 'sonnet', strategist: 'opus' },
    schedule: { analyst: '16:45 America/New_York daily', strategist: 'Sunday 10:00 America/New_York' },
    history,
  }
}

const CONFIG: ConfigTree = {
  broker: { mode: 'paper', provider: 'alpaca', api_key: '***redacted***', base_url: 'https://paper-api.alpaca.markets', data_feed: 'iex' },
  engine: { tick_seconds: 60, backfill_days: 30, daily_backfill_days: 400, timezone: 'America/New_York', regular_hours_only: true },
  risk: {
    risk_per_trade_frac: 0.0075,
    max_gross_exposure_frac: 0.6,
    max_positions: 10,
    daily_loss_limit_pct: 2.0,
    daily_halt_limit_pct: 4.0,
    variant_dd_pause_pct: 8.0,
    one_direction_per_symbol: true,
  },
  tournament: {
    population_cap: 24,
    incubation_min_trades: 20,
    probation_ci_upper: 0.0,
    retire_after_probation_days: 10,
    exploration_floor: 0.02,
    thompson_window_days: 20,
  },
  attribution: { window_days: 30, min_trades: 30, min_bucket_n: 8, oos_split: 0.3 },
  optimizer: { cadence: 'weekly', random_search_samples: 40, walk_forward_folds: 3, min_param_distance: 0.15 },
  claude: {
    plan: 'max5',
    weekly_share: 0.25,
    models: { analyst: 'sonnet', strategist: 'opus' },
    schedule: { analyst: '16:45 America/New_York daily', strategist: 'Sunday 10:00 America/New_York' },
    max_turns: { analyst: 12, strategist: 30 },
    max_budget_usd: { analyst: 1.5, strategist: 6.0 },
    min_new_trades: 8,
    memory_cap_chars: 12000,
  },
  universe: { equities: EQUITIES, crypto: CRYPTO },
  storage: { db_path: 'data/sentinel.db', wal: true, vacuum_days: 7 },
  api: { host: '127.0.0.1', port: 8787, serve_ui: true, ui_dist: 'ui/dist' },
}

// ---------------------------------------------------------------------------
// Build world
// ---------------------------------------------------------------------------

function buildWorld(): World {
  const rng = new Rng(SEED)
  const { runtimes } = buildVariants(rng)
  const trades = genTrades(rng, runtimes)
  const variants = materializeVariants(runtimes, trades)
  const equity = buildEquity(rng, trades)
  const positions = buildPositions(rng, runtimes)
  for (const p of positions) {
    const v = variants.find((x) => x.id === p.variant_id)
    if (v) v.open_positions += 1
  }
  const events = buildEvents(trades, runtimes)
  const { runs, proposals } = buildResearch(rng)
  const budget = buildBudget(runs)
  const last = equity[equity.length - 1]!
  return {
    rng,
    runtimes,
    variants,
    trades,
    equity,
    positions,
    events,
    runs,
    proposals,
    budget,
    memory: { memory_md: MEMORY_MD, updated_at: runs.find((r) => r.kind === 'strategist')?.finished_at ?? iso(NOW - 3 * DAY) },
    engine: { running: true, paused: false, halted: false, last_tick: iso(NOW), uptime_s: 35 * 3600 + 1240, tick_count: 2120, errors_1h: 0 },
    nextEventId: events.length + 1,
    noise: 0,
    exposureFrac: 0.39,
    equityNow: last.equity,
  }
}

let world: World | null = null
function W(): World {
  if (!world) world = buildWorld()
  return world
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

function equityAt(points: EquityPoint[], ms: number): number {
  let lo = 0
  let hi = points.length - 1
  if (hi < 0) return STARTING_EQUITY
  if (new Date(points[0]!.ts).getTime() >= ms) return points[0]!.equity
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (new Date(points[mid]!.ts).getTime() <= ms) lo = mid
    else hi = mid - 1
  }
  return points[lo]!.equity
}

function buildStatus(w: World): StatusResponse {
  const now = Date.now()
  const equity = round(w.equityNow + w.positions.reduce((a, p) => a + p.unrealized_pnl, 0), 2)
  const p = nyParts(new Date(now))
  const dayStart = nyWallToInstant(new Date(now), 0, 0, 0).getTime()
  const weekStart = nyWallToInstant(new Date(now), -((p.weekday + 6) % 7), 0, 0).getTime()
  const dayBase = equityAt(w.equity, dayStart)
  const weekBase = equityAt(w.equity, weekStart)
  const gross = w.positions.reduce((a, pos) => a + Math.abs(pos.qty * pos.current_price), 0)
  const sess = sessionAt(now)
  const { next_open, next_close } = nextOpenClose(now)
  const market: MarketState = { equities_open: sess.open, next_open, next_close, session: sess.session, crypto_open: true }
  const counts = { active: 0, incubating: 0, probation: 0, paused: 0, retired: 0 }
  for (const v of w.variants) counts[v.status] += 1
  const nextAnalyst = (() => {
    const t = nyWallToInstant(new Date(now), 0, 16, 45)
    return t.getTime() > now ? t : nyWallToInstant(new Date(now), 1, 16, 45)
  })()
  const nextStrategist = (() => {
    const daysToSunday = (7 - p.weekday) % 7
    const t = nyWallToInstant(new Date(now), daysToSunday, 10, 0)
    return t.getTime() > now ? t : nyWallToInstant(new Date(now), daysToSunday + 7, 10, 0)
  })()
  return {
    mode: 'paper',
    engine: { ...w.engine },
    market,
    account: {
      equity,
      cash: round(equity - gross, 2),
      day_pnl: round(equity - dayBase, 2),
      day_pnl_pct: round(((equity - dayBase) / dayBase) * 100, 2),
      week_pnl: round(equity - weekBase, 2),
      week_pnl_pct: round(((equity - weekBase) / weekBase) * 100, 2),
      total_pnl: round(equity - STARTING_EQUITY, 2),
      total_pnl_pct: round(((equity - STARTING_EQUITY) / STARTING_EQUITY) * 100, 2),
      gross_exposure_pct: round((gross / equity) * 100, 1),
      open_positions: w.positions.length,
      starting_equity: STARTING_EQUITY,
    },
    population: counts,
    budget: {
      weekly_cap_usd: w.budget.weekly_cap_usd,
      spent_usd: w.budget.spent_usd,
      remaining_usd: w.budget.remaining_usd,
      share_of_plan: w.budget.weekly_share,
      next_analyst_run: nextAnalyst.toISOString(),
      next_strategist_run: nextStrategist.toISOString(),
    },
    universe: { equities: EQUITIES, crypto: CRYPTO },
  }
}

// ---------------------------------------------------------------------------
// Live stream
// ---------------------------------------------------------------------------

const subscribers = new Set<StreamHandlers>()
let timer: ReturnType<typeof setInterval> | null = null
let tickN = 0

const LIVE_EVENTS: { level: EventLevel; kind: EventKind; msg: () => string }[] = [
  { level: 'info', kind: 'signal', msg: () => `trend_ema#2 long ${pickSym()}: EMA cross + ADX ${W().rng.int(24, 38)}` },
  { level: 'info', kind: 'fill', msg: () => `Filled LONG ${W().rng.int(4, 40)} ${pickSym()} @ market for ${W().rng.pick(['meanrev_bb#1', 'breakout_orb#1', 'vwap_revert#1'])}` },
  { level: 'info', kind: 'exit', msg: () => `${W().rng.pick(['trend_ema#1', 'crypto_mtf#1', 'xs_momentum#1'])} closed ${pickSym()} on target: +${W().rng.range(20, 140).toFixed(2)} (+${W().rng.range(0.8, 2.1).toFixed(2)}R)` },
  { level: 'warn', kind: 'exit', msg: () => `${W().rng.pick(['meanrev_bb#1', 'breakout_donchian#1'])} closed ${pickSym()} on stop: -${W().rng.range(30, 110).toFixed(2)} (-1.0${W().rng.int(0, 9)}R)` },
  { level: 'info', kind: 'data', msg: () => `Bars poll OK: 16 symbols in ${W().rng.range(0.4, 1.9).toFixed(1)}s` },
  { level: 'warn', kind: 'risk', msg: () => `Conflicting signal ignored: ${W().rng.pick(['squeeze#3', 'random_entry#1'])} short ${pickSym()} while another sleeve holds long` },
  { level: 'info', kind: 'tournament', msg: () => `Metrics refreshed for ${W().variants.length} variants` },
  { level: 'info', kind: 'system', msg: () => `Equity snapshot persisted (${W().rng.int(1, 9)} open lots)` },
]
function pickSym() {
  return W().rng.pick([...EQUITIES.slice(0, 8), ...CRYPTO])
}

function emitEvent(level: EventLevel, kind: EventKind, message: string, data: Record<string, unknown> = {}) {
  const w = W()
  const row: EventRow = { id: w.nextEventId++, ts: iso(Date.now()), level, kind, message, data }
  w.events.push(row)
  if (w.events.length > 600) w.events.splice(0, w.events.length - 600)
  for (const s of subscribers) s.event?.(row)
}

function liveTick() {
  const w = W()
  tickN++
  const now = Date.now()
  if (w.engine.running && !w.engine.halted) {
    w.engine.last_tick = iso(now)
    w.engine.uptime_s += 2
    if (tickN % 30 === 0) w.engine.tick_count += 1
  }
  // Random-walk prices and the unrealized component.
  if (!w.engine.halted) {
    for (const p of w.positions) {
      const riskUsd = Math.max(1, (w.variants.find((v) => v.id === p.variant_id)?.allocation ?? 0.05) * STARTING_EQUITY * 0.0075)
      p.current_price = round(p.current_price * (1 + w.rng.normal(0, 0.0006)), 2)
      const dir = p.side === 'long' ? 1 : -1
      p.unrealized_pnl = round(dir * (p.current_price - p.entry_price) * p.qty, 2)
      p.unrealized_r = round(p.unrealized_pnl / riskUsd, 2)
    }
    w.noise = 0.98 * w.noise + w.rng.normal(0, 12)
    const last = w.equity[w.equity.length - 1]!
    w.equityNow = round(last.equity + w.noise, 2)
  }
  const status = buildStatus(w)
  for (const s of subscribers) s.tick?.(status)
  if (tickN % 15 === 0 && !w.engine.halted) {
    const last = w.equity[w.equity.length - 1]!
    const point: EquityPoint = {
      ts: iso(now),
      equity: status.account.equity,
      cash: status.account.cash,
      benchmark: round(last.benchmark * (1 + w.rng.normal(0, 0.0004)), 2),
    }
    w.equity.push(point)
    for (const s of subscribers) s.equity?.(point)
  }
  if (w.rng.chance(0.18)) {
    const e = w.rng.pick(LIVE_EVENTS)
    emitEvent(e.level, e.kind, e.msg())
  }
}

function startStream() {
  if (timer) return
  timer = setInterval(liveTick, 2000)
}
function stopStream() {
  if (timer && subscribers.size === 0) {
    clearInterval(timer)
    timer = null
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

async function latency<T>(value: T, ms?: number): Promise<T> {
  await sleep(ms ?? 90 + W().rng.range(0, 220))
  return value
}

function downsample(points: EquityPoint[], stepMs: number): EquityPoint[] {
  if (points.length === 0) return points
  const out: EquityPoint[] = []
  let nextAt = -Infinity
  for (const p of points) {
    const t = new Date(p.ts).getTime()
    if (t >= nextAt) {
      out.push(p)
      nextAt = t + stepMs
    }
  }
  const last = points[points.length - 1]!
  if (out[out.length - 1] !== last) out.push(last)
  return out
}

const clone = <T,>(v: T): T => structuredClone(v)

export const mockClient: ApiClient = {
  getStatus: () => latency(buildStatus(W()), 60),

  getEquity: (range: EquityRange, variant?: string) => {
    const w = W()
    const now = Date.now()
    if (variant) {
      const vt = w.trades.filter((t) => t.variant_id === variant)
      const v = w.variants.find((x) => x.id === variant)
      const sleeve = (v?.allocation ?? 0.05) * STARTING_EQUITY
      let cum = 0
      const pts: EquityPoint[] = vt.map((t) => {
        cum += t.pnl
        return { ts: t.exit_ts, equity: round(sleeve + cum, 2), cash: round(sleeve + cum, 2), benchmark: sleeve }
      })
      return latency(pts)
    }
    const spans: Record<EquityRange, [number, number]> = {
      '1d': [24 * HOUR, 5 * MIN],
      '1w': [7 * DAY, 30 * MIN],
      '1m': [30 * DAY, 2 * HOUR],
      all: [Infinity, 4 * HOUR],
    }
    const [span, step] = spans[range]
    const from = now - span
    const sliced = w.equity.filter((p) => new Date(p.ts).getTime() >= from)
    return latency(clone(downsample(sliced, step)))
  },

  getPositions: () => latency(clone(W().positions)),

  getTrades: (q: TradeQuery = {}) => {
    const w = W()
    let ts = [...w.trades].sort((a, b) => b.exit_ts.localeCompare(a.exit_ts))
    if (q.variant) ts = ts.filter((t) => t.variant_id === q.variant)
    if (q.symbol) ts = ts.filter((t) => t.symbol === q.symbol)
    if (q.since) ts = ts.filter((t) => t.exit_ts > (q.since as string))
    return latency(clone(ts.slice(0, q.limit ?? 200)))
  },

  getVariants: () => latency(clone(W().variants)),

  getVariant: (id: string) => {
    const w = W()
    const v = w.variants.find((x) => x.id === id)
    if (!v) return Promise.reject(new Error(`variant ${id} not found`))
    const vt = w.trades.filter((t) => t.variant_id === id)
    let cum = 0
    const equity_curve = vt.map((t) => {
      cum += t.pnl
      return { ts: t.exit_ts, pnl_cum: round(cum, 2) }
    })
    const lineage: LineageEntry[] = []
    const seen = new Set<string>()
    let cur: Variant | undefined = v
    while (cur && !seen.has(cur.id)) {
      seen.add(cur.id)
      lineage.unshift({ id: cur.id, origin: cur.origin, created_at: cur.created_at, status: cur.status })
      cur = cur.parent_id ? w.variants.find((x) => x.id === cur!.parent_id) : undefined
    }
    for (const child of w.variants.filter((x) => x.parent_id === id)) {
      lineage.push({ id: child.id, origin: child.origin, created_at: child.created_at, status: child.status })
    }
    const notes =
      v.origin === 'claude'
        ? `Spawned from proposal by analyst run; parent ${v.parent_id}. Gate: walk-forward backtest beat incumbent and control.`
        : v.origin === 'optimizer'
          ? `Optimizer random-search child of ${v.parent_id} (walk-forward, 3 folds).`
          : v.is_control
            ? 'Control sleeve. Never allocated above the exploration floor; used as the baseline for every gate.'
            : 'Seed variant from the default population.'
    const detail: VariantDetail = { ...clone(v), equity_curve, trades: clone(vt.slice(-100).reverse()), lineage, notes }
    return latency(detail)
  },

  setVariantStatus: (id: string, status: VariantStatusChange) => {
    const w = W()
    const v = w.variants.find((x) => x.id === id)
    if (!v) return Promise.reject(new Error(`variant ${id} not found`))
    const prev = v.status
    v.status = status
    if (status !== 'active') v.allocation = 0
    else if (v.allocation === 0) v.allocation = 0.02
    emitEvent('info', 'tournament', `Operator set ${id} ${prev} -> ${status}`)
    return latency(clone(v), 150)
  },

  getAttribution: () => latency(buildAttribution(W().rng, W().trades), 220),

  getResearch: (limit = 20) => latency(clone(W().runs.slice(0, limit).map(stripRun))),

  getResearchRun: (id: number) => {
    const r = W().runs.find((x) => x.id === id)
    return r ? latency(clone(r)) : Promise.reject(new Error(`run ${id} not found`))
  },

  getMemory: () => latency(clone(W().memory)),

  getProposals: (q: ProposalQuery = {}) => {
    let ps = [...W().proposals].sort((a, b) => b.created_at.localeCompare(a.created_at))
    if (q.status) ps = ps.filter((p) => p.status === q.status)
    return latency(clone(ps.slice(0, q.limit ?? 100)))
  },

  decideProposal: (id: number, decision: Decision) => {
    const w = W()
    const p = w.proposals.find((x) => x.id === id)
    if (!p) return Promise.reject(new Error(`proposal ${id} not found`))
    p.status = decision === 'accept' ? 'accepted' : 'rejected'
    p.decision_reason = decision === 'accept' ? 'Accepted by operator' : 'Rejected by operator'
    if (decision === 'accept' && p.type === 'retire' && p.target) {
      const v = w.variants.find((x) => x.id === p.target)
      if (v) {
        v.status = 'retired'
        v.allocation = 0
      }
    }
    const run = w.runs.find((r) => r.id === p.run_id)
    if (run) run.n_accepted = run.proposals.filter((x) => x.status === 'accepted').length
    emitEvent('info', 'research', `Operator ${p.status} proposal #${id} (${p.type}${p.target ? ' ' + p.target : ''})`)
    return latency(clone(p), 150)
  },

  getBudget: () => latency(clone(W().budget)),

  calibrateBudget: (observedWeeklyPct: number) => {
    const w = W()
    const b = w.budget
    // Rescale: observed pct of plan usage vs actual spend implies the plan allowance.
    const impliedAllowance = observedWeeklyPct > 0 ? round((b.spent_usd / (observedWeeklyPct / 100)) * 1, 0) : b.weekly_allowance_usd_est
    b.calibration = { observed_pct: observedWeeklyPct, note: `calibrated from ${observedWeeklyPct}% observed usage` }
    b.weekly_allowance_usd_est = Math.max(20, impliedAllowance)
    b.weekly_cap_usd = round(b.weekly_allowance_usd_est * b.weekly_share, 2)
    b.remaining_usd = round(b.weekly_cap_usd - b.spent_usd, 2)
    emitEvent('info', 'system', `Budget calibrated: observed ${observedWeeklyPct}% -> allowance est $${b.weekly_allowance_usd_est}/wk, cap $${b.weekly_cap_usd}`)
    return latency(clone(b), 200)
  },

  getEvents: (q: EventQuery = {}) => {
    let es = [...W().events].sort((a, b) => b.ts.localeCompare(a.ts))
    if (q.since) es = es.filter((e) => e.ts > (q.since as string))
    return latency(clone(es.slice(0, q.limit ?? 100)))
  },

  engineAction: (action: EngineAction) => {
    const w = W()
    const e = w.engine
    switch (action) {
      case 'pause':
        e.paused = true
        emitEvent('warn', 'system', 'Operator paused entries')
        break
      case 'resume':
        e.paused = false
        e.halted = false
        e.running = true
        emitEvent('info', 'system', 'Operator resumed the engine')
        break
      case 'halt':
        e.halted = true
        e.paused = true
        e.running = false
        emitEvent('error', 'system', 'Operator HALTED the engine')
        break
      case 'flatten':
        w.positions.splice(0, w.positions.length)
        for (const v of w.variants) v.open_positions = 0
        emitEvent('warn', 'risk', 'Operator flattened all positions')
        break
    }
    const res: EngineActionResponse = { ok: true, engine: { ...e } }
    return latency(res, 200)
  },

  runResearch: (kind: RunKind) => {
    const w = W()
    if (w.runs.some((r) => r.status === 'running')) {
      return latency<ResearchRunResponse>({ queued: false, reason: 'a research session is already running' }, 150)
    }
    const estCost = kind === 'analyst' ? 0.5 : 3.0
    if (w.budget.remaining_usd < estCost) {
      return latency<ResearchRunResponse>({ queued: false, reason: 'budget exhausted' }, 150)
    }
    const id = Math.max(0, ...w.runs.map((r) => r.id)) + 1
    const run: ResearchRunDetail = {
      id,
      kind,
      model: kind === 'analyst' ? 'sonnet' : 'opus',
      started_at: iso(Date.now()),
      finished_at: null,
      status: 'running',
      cost_usd: 0,
      input_tokens: 0,
      output_tokens: 0,
      num_turns: 0,
      summary: 'Running…',
      n_proposals: 0,
      n_accepted: 0,
      analysis_md: '',
      memory_update_md: '',
      proposals: [],
      digest_md: '',
      error: null,
    }
    w.runs.unshift(run)
    emitEvent('info', 'research', `${kind} session #${id} started on demand (${run.model})`)
    setTimeout(() => {
      run.status = 'ok'
      run.finished_at = iso(Date.now())
      run.cost_usd = round(estCost * w.rng.range(0.7, 1.1), 2)
      run.input_tokens = w.rng.int(40_000, 120_000)
      run.output_tokens = w.rng.int(1_500, 6_000)
      run.num_turns = w.rng.int(4, 12)
      run.summary = 'On-demand session: no new proposals; confirmed the hour_et=9 finding on the latest trades.'
      run.analysis_md = ANALYSIS_3.replace('Quiet session', 'On-demand session')
      run.digest_md = '## Digest\n\nOn-demand run requested by operator.'
      w.budget.spent_usd = round(w.budget.spent_usd + run.cost_usd, 2)
      w.budget.remaining_usd = round(w.budget.weekly_cap_usd - w.budget.spent_usd, 2)
      w.budget.runs_this_week += 1
      const hist = w.budget.history[w.budget.history.length - 1]
      if (hist) {
        hist.spent_usd = w.budget.spent_usd
        hist.runs += 1
      }
      emitEvent('info', 'research', `${kind} session #${id} finished: 0 proposals, $${run.cost_usd.toFixed(2)}`)
    }, 20_000)
    return latency<ResearchRunResponse>({ queued: true }, 150)
  },

  getConfig: () => latency(clone(CONFIG)),

  getBars: (symbol: string, _timeframe: string, limit = 300) => {
    const w = W()
    const base = BASE_PRICE[symbol] ?? 100
    const bars: Bar[] = []
    let c = base * 0.98
    for (let i = limit; i > 0; i--) {
      const o = c
      c = o * (1 + w.rng.normal(0.0001, 0.004))
      const h = Math.max(o, c) * (1 + Math.abs(w.rng.normal(0, 0.002)))
      const l = Math.min(o, c) * (1 - Math.abs(w.rng.normal(0, 0.002)))
      bars.push({ t: iso(Date.now() - i * 15 * MIN), o: round(o), h: round(h), l: round(l), c: round(c), v: w.rng.int(1000, 90000) })
    }
    return latency(bars)
  },

  getBacktests: () => latency([]),

  runBacktest: (req: BacktestRequest) => {
    const w = W()
    const family = req.family ?? (req.variant_id ? req.variant_id.split('#')[0] ?? 'trend_ema' : 'trend_ema')
    const res: BacktestResult = {
      id: 1,
      family,
      params: req.params ?? {},
      days: req.days ?? 45,
      n: w.rng.int(30, 80),
      expectancy_r: round(w.rng.normal(0.1, 0.15), 3),
      win_rate: round(w.rng.range(0.4, 0.6), 3),
      profit_factor: round(w.rng.range(0.9, 1.7), 2),
      max_dd_pct: round(w.rng.range(2, 7), 2),
      pnl: round(w.rng.normal(200, 300), 2),
      equity_curve: [],
      trades: [],
    }
    return latency(res, 1500)
  },

  stream: (handlers: StreamHandlers): Unsubscribe => {
    subscribers.add(handlers)
    startStream()
    // Signal "open" asynchronously, then an immediate first tick so the UI fills in fast.
    setTimeout(() => {
      handlers.onOpen?.()
      handlers.tick?.(buildStatus(W()))
    }, 30)
    return () => {
      subscribers.delete(handlers)
      stopStream()
    }
  },
}

function stripRun(r: ResearchRunDetail): ResearchRun {
  const { analysis_md: _a, memory_update_md: _m, proposals: _p, digest_md: _d, error: _e, ...rest } = r
  return rest
}
