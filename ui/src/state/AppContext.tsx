import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { api, probeBackend, useApiMode, useSSE, type ConnectionState, type MockReason } from '../api/client'
import type { EquityPoint, EventRow, StatusResponse } from '../api/types'

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

  // First contact: probe /api/status; fall back to the mock on failure.
  useEffect(() => {
    if (probed.current) return
    probed.current = true
    void probeBackend().then(({ status: s }) => {
      setStatus(s)
      setStatusError(null)
    })
  }, [])

  const connection = useSSE({
    tick: (s) => {
      setStatus(s)
      setStatusError(null)
    },
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
      const s = await api.getStatus()
      setStatus(s)
      setStatusError(null)
    } catch (e) {
      setStatusError(e instanceof Error ? e : new Error(String(e)))
    }
  }, [])

  // Polling fallback: every 5 s whenever the stream is not delivering ticks.
  useEffect(() => {
    if (connection === 'connected' || connection === 'mock') return
    const id = setInterval(() => {
      if (document.hidden) return
      void refreshStatus()
    }, 5000)
    return () => clearInterval(id)
  }, [connection, refreshStatus])

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
    }),
    [status, statusError, connection, mode, reason, bannerDismissed, liveEvents, liveEquity, changeTick, refreshStatus],
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
