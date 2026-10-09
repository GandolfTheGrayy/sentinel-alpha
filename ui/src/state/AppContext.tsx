import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, probeBackend, useApiMode, useSSE, type ConnectionState, type MockReason } from '../api/client'
import { useNow } from '../api/hooks'
import type { EquityPoint, EventRow, StatusResponse } from '../api/types'
import { nowMs, setSimTime } from '../lib/clock'

interface AppState {
  status: StatusResponse | null
  statusError: Error | null
  connection: ConnectionState
  apiMode: 'real' | 'mock'
  mockReason: MockReason
  bannerDismissed: boolean
  dismissBanner: () => void
  /** Events received over the stream since page load (newest first, capped). */
  liveEvents: EventRow[]
  /** Equity points received over the stream since page load (oldest first, capped). */
  liveEquity: EquityPoint[]
  /** Increments whenever a tournament/research/system event arrives (pages may refetch). */
  changeTick: number
  refreshStatus: () => Promise<void>
  /** True while the engine reports a simulated clock (`engine.sim_time`). */
  simClock: boolean
  /** The app clock in ms: simulated time (advanced locally between ticks) in sim mode, else the wall clock. */
  nowMs: () => number
}

const AppContext = createContext<AppState | null>(null)

const MAX_LIVE_EVENTS = 300
const MAX_LIVE_EQUITY = 2000

export function AppProvider({ children }: { children: ReactNode }) {
  const { mode, reason } = useApiMode()
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [statusError, setStatusError] = useState<Error | null>(null)
  const [bannerDismissed, setBannerDismissed] = useState(false)
  const [liveEvents, setLiveEvents] = useState<EventRow[]>([])
  const [liveEquity, setLiveEquity] = useState<EquityPoint[]>([])
  const [changeTick, setChangeTick] = useState(0)
  const probed = useRef(false)

  /** Every status (probe, stream tick, poll) goes through here so the clock source stays in sync. */
  const applyStatus = useCallback((s: StatusResponse) => {
    setSimTime(s.engine?.sim_time ?? null)
    setStatus(s)
    setStatusError(null)
  }, [])

  // First contact: probe /api/status; fall back to the mock on failure.
  useEffect(() => {
    if (probed.current) return
    probed.current = true
    void probeBackend().then(({ status: s }) => applyStatus(s))
  }, [applyStatus])

  const connection = useSSE({
    tick: applyStatus,
    event: (e) => {
      setLiveEvents((prev) => {
        if (prev.some((x) => x.id === e.id)) return prev
        const next = [e, ...prev]
        return next.length > MAX_LIVE_EVENTS ? next.slice(0, MAX_LIVE_EVENTS) : next
      })
      if (e.kind === 'tournament' || e.kind === 'research' || e.kind === 'system' || e.kind === 'risk') {
        setChangeTick((t) => t + 1)
      }
    },
    equity: (p) => {
      setLiveEquity((prev) => {
        const next = [...prev, p]
        return next.length > MAX_LIVE_EQUITY ? next.slice(next.length - MAX_LIVE_EQUITY) : next
      })
    },
  })

  const refreshStatus = useCallback(async () => {
    try {
      applyStatus(await api.getStatus())
    } catch (e) {
      setStatusError(e instanceof Error ? e : new Error(String(e)))
    }
  }, [applyStatus])

  // Polling fallback: every 5 s whenever the stream is not delivering ticks.
  useEffect(() => {
    if (connection === 'connected' || connection === 'mock') return
    const id = setInterval(() => {
      if (document.hidden) return
      void refreshStatus()
    }, 5000)
    return () => clearInterval(id)
  }, [connection, refreshStatus])

  const simClock = status?.engine?.sim_time != null

  const value = useMemo<AppState>(
    () => ({
      status,
      statusError,
      connection,
      apiMode: mode,
      mockReason: reason,
      bannerDismissed,
      dismissBanner: () => setBannerDismissed(true),
      liveEvents,
      liveEquity,
      changeTick,
      refreshStatus,
      simClock,
      nowMs,
    }),
    [status, statusError, connection, mode, reason, bannerDismissed, liveEvents, liveEquity, changeTick, refreshStatus, simClock],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppState {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used within AppProvider')
  return ctx
}

export function useStatus(): StatusResponse | null {
  return useApp().status
}

/**
 * Ticking app clock for live displays (ET clock, countdowns). Re-renders every `intervalMs`.
 * `now` is simulated time in sim mode (see lib/clock.ts), otherwise the wall clock.
 */
export function useClock(intervalMs = 1000): { now: number; isSim: boolean } {
  const { simClock } = useApp()
  const now = useNow(intervalMs)
  return { now, isSim: simClock }
}

export { nowMs }
