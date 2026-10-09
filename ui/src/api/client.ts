/**
 * Typed HTTP client for the Sentinel API plus the mock switch and the SSE hook.
 *
 * `api` always points at the active implementation: the HTTP client by default, or the
 * mock when VITE_MOCK === '1' or after the first /api/status request fails.
 */
import { useEffect, useRef, useState, useSyncExternalStore } from 'react'
import type { ApiClient, StreamHandlers, Unsubscribe } from './contract'
import { mockClient } from './mock'
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
  EquityPoint,
  EquityRange,
  EventQuery,
  EventRow,
  LabMemory,
  Position,
  Proposal,
  ProposalQuery,
  ResearchRun,
  ResearchRunDetail,
  ResearchRunResponse,
  RunKind,
  StatusResponse,
  Trade,
  TradeQuery,
  Variant,
  VariantDetail,
  VariantStatusChange,
} from './types'

// ---------------------------------------------------------------------------
// API mode store
// ---------------------------------------------------------------------------

export type ApiMode = 'real' | 'mock'
export type MockReason = 'env' | 'fallback' | null

interface ModeState {
  mode: ApiMode
  reason: MockReason
}

const envMock = import.meta.env.VITE_MOCK === '1'
let modeState: ModeState = envMock ? { mode: 'mock', reason: 'env' } : { mode: 'real', reason: null }
const modeListeners = new Set<() => void>()

export function getApiMode(): ModeState {
  return modeState
}

/** Switch to demo data (called when the backend is unreachable on first contact). */
export function activateMock(reason: Exclude<MockReason, null>) {
  if (modeState.mode === 'mock') return
  modeState = { mode: 'mock', reason }
  modeListeners.forEach((l) => l())
}

function subscribeMode(cb: () => void) {
  modeListeners.add(cb)
  return () => {
    modeListeners.delete(cb)
  }
}

export function useApiMode(): ModeState {
  return useSyncExternalStore(subscribeMode, getApiMode, getApiMode)
}

// ---------------------------------------------------------------------------
// HTTP implementation
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

type Query = Record<string, string | number | undefined | null>

function qs(params: Query): string {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

async function request<T>(path: string, init?: { method?: 'GET' | 'POST'; body?: unknown }): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  let body: string | undefined
  if (init?.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(init.body)
  }
  const res = await fetch(path, { method: init?.method ?? 'GET', headers, body, cache: 'no-store' })
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.text()).slice(0, 200)
    } catch {
      // ignore
    }
    throw new ApiError(res.status, `${res.status} ${res.statusText}${detail ? `: ${detail}` : ''}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

function parseEvent<T>(e: Event): T | null {
  const data = (e as MessageEvent<string>).data
  if (typeof data !== 'string') return null
  try {
    return JSON.parse(data) as T
  } catch {
    return null
  }
}

function httpStream(handlers: StreamHandlers): Unsubscribe {
  let es: EventSource | null = null
  let closed = false
  let attempt = 0
  let everConnected = false
  let timer: ReturnType<typeof setTimeout> | null = null

  const connect = () => {
    if (closed) return
    try {
      es = new EventSource('/api/stream')
    } catch {
      scheduleRetry()
      return
    }
    es.onopen = () => {
      attempt = 0
      everConnected = true
      handlers.onOpen?.()
    }
    es.addEventListener('tick', (e) => {
      const p = parseEvent<StatusResponse>(e)
      if (p) handlers.tick?.(p)
    })
    es.addEventListener('event', (e) => {
      const p = parseEvent<EventRow>(e)
      if (p) handlers.event?.(p)
    })
    es.addEventListener('equity', (e) => {
      const p = parseEvent<EquityPoint>(e)
      if (p) handlers.equity?.(p)
    })
    es.onerror = () => {
      es?.close()
      es = null
      scheduleRetry()
    }
  }

  const scheduleRetry = () => {
    if (closed) return
    attempt += 1
    const base = Math.min(30_000, 1000 * 2 ** Math.min(attempt - 1, 5))
    const retryInMs = Math.round(base + Math.random() * 400)
    handlers.onError?.({ attempt, everConnected, retryInMs })
    timer = setTimeout(connect, retryInMs)
  }

  connect()
  return () => {
    closed = true
    if (timer) clearTimeout(timer)
    es?.close()
    es = null
  }
}

export const httpClient: ApiClient = {
  getStatus: () => request<StatusResponse>('/api/status'),
  getEquity: (range: EquityRange, variant?: string) => request<EquityPoint[]>(`/api/equity${qs({ range, variant })}`),
  getPositions: () => request<Position[]>('/api/positions'),
  getTrades: (q: TradeQuery = {}) => request<Trade[]>(`/api/trades${qs({ limit: q.limit ?? 200, variant: q.variant, symbol: q.symbol, since: q.since })}`),
  getVariants: () => request<Variant[]>('/api/variants'),
  getVariant: (id: string) => request<VariantDetail>(`/api/variants/${encodeURIComponent(id)}`),
  setVariantStatus: (id: string, status: VariantStatusChange) =>
    request<Variant>(`/api/variants/${encodeURIComponent(id)}/status`, { method: 'POST', body: { status } }),
  getAttribution: () => request<AttributionReport>('/api/attribution'),
  getResearch: (limit = 20) => request<ResearchRun[]>(`/api/research${qs({ limit })}`),
  getResearchRun: (id: number) => request<ResearchRunDetail>(`/api/research/${id}`),
  getMemory: () => request<LabMemory>('/api/research/memory'),
  getProposals: (q: ProposalQuery = {}) => request<Proposal[]>(`/api/proposals${qs({ status: q.status, limit: q.limit ?? 100 })}`),
  decideProposal: (id: number, decision: Decision) => request<Proposal>(`/api/proposals/${id}/decision`, { method: 'POST', body: { decision } }),
  getBudget: () => request<Budget>('/api/budget'),
  calibrateBudget: (observedWeeklyPct: number) =>
    request<Budget>('/api/budget/calibrate', { method: 'POST', body: { observed_weekly_pct: observedWeeklyPct } }),
  getEvents: (q: EventQuery = {}) => request<EventRow[]>(`/api/events${qs({ limit: q.limit ?? 100, since: q.since })}`),
  engineAction: (action: EngineAction) => request<EngineActionResponse>(`/api/engine/${action}`, { method: 'POST' }),
  runResearch: (kind: RunKind) => request<ResearchRunResponse>('/api/research/run', { method: 'POST', body: { kind } }),
  getConfig: () => request<ConfigTree>('/api/config'),
  getBars: (symbol: string, timeframe: string, limit = 300) => request<Bar[]>(`/api/bars${qs({ symbol, timeframe, limit })}`),
  getBacktests: (limit = 20) => request<BacktestResult[]>(`/api/backtests${qs({ limit })}`),
  runBacktest: (req: BacktestRequest) => request<BacktestResult>('/api/backtests', { method: 'POST', body: req }),
  stream: httpStream,
}

// ---------------------------------------------------------------------------
// Active client
// ---------------------------------------------------------------------------

function current(): ApiClient {
  return modeState.mode === 'mock' ? mockClient : httpClient
}

/** The active API client. Resolves the implementation at call time. */
export const api: ApiClient = new Proxy({} as ApiClient, {
  get(_target, prop: string | symbol) {
    const impl = current() as unknown as Record<string | symbol, unknown>
    return impl[prop]
  },
})

/**
 * Probe the backend once. Returns the status when reachable; otherwise switches to the
 * mock and returns the mock status. Only the *first* failure triggers the fallback.
 */
export async function probeBackend(): Promise<{ status: StatusResponse; fellBack: boolean }> {
  if (modeState.mode === 'mock') {
    return { status: await mockClient.getStatus(), fellBack: false }
  }
  try {
    const status = await httpClient.getStatus()
    return { status, fellBack: false }
  } catch {
    activateMock('fallback')
    return { status: await mockClient.getStatus(), fellBack: true }
  }
}

// ---------------------------------------------------------------------------
// SSE hook
// ---------------------------------------------------------------------------

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting' | 'unreachable' | 'mock'

/**
 * Subscribe to /api/stream. Handlers are read through a ref so callers can pass fresh
 * closures on every render without re-subscribing. Reconnects with exponential backoff.
 */
export function useSSE(handlers: StreamHandlers, enabled = true): ConnectionState {
  const { mode } = useApiMode()
  const ref = useRef(handlers)
  useEffect(() => {
    ref.current = handlers
  })
  const [state, setState] = useState<ConnectionState>(mode === 'mock' ? 'mock' : 'connecting')

  useEffect(() => {
    if (!enabled) return
    const isMock = mode === 'mock'
    const unsub = api.stream({
      tick: (p) => ref.current.tick?.(p),
      event: (p) => ref.current.event?.(p),
      equity: (p) => ref.current.equity?.(p),
      onOpen: () => {
        setState(isMock ? 'mock' : 'connected')
        ref.current.onOpen?.()
      },
      onError: (info) => {
        if (isMock) return
        setState(info.everConnected && info.attempt < 4 ? 'reconnecting' : info.attempt >= 2 ? 'unreachable' : 'reconnecting')
        ref.current.onError?.(info)
      },
    })
    return unsub
  }, [mode, enabled])

  return state
}
