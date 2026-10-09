import { useCallback, useEffect, useRef, useState } from 'react'
import { useApiMode } from './client'

export interface PollState<T> {
  data: T | null
  error: Error | null
  /** True only until the first successful (or failed) load. */
  loading: boolean
  /** True while a background refresh is in flight (previous data is kept). */
  refreshing: boolean
  refresh: () => Promise<void>
  /** Optimistically replace the data. */
  setData: (updater: (prev: T | null) => T | null) => void
}

/**
 * Fetch-and-poll hook. Keeps the previous data while refreshing (no skeleton flash),
 * pauses while the tab is hidden, and refetches when the API mode flips to mock.
 */
export function usePoll<T>(fetcher: () => Promise<T>, intervalMs: number | null, deps: ReadonlyArray<unknown> = []): PollState<T> {
  const { mode } = useApiMode()
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const fetcherRef = useRef(fetcher)
  useEffect(() => {
    fetcherRef.current = fetcher
  })
  const generation = useRef(0)

  const run = useCallback(async () => {
    const gen = ++generation.current
    setRefreshing(true)
    try {
      const result = await fetcherRef.current()
      if (gen !== generation.current) return
      setData(result)
      setError(null)
    } catch (e) {
      if (gen !== generation.current) return
      setError(e instanceof Error ? e : new Error(String(e)))
    } finally {
      if (gen === generation.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    generation.current++
    setLoading(true)
    void run()
    if (intervalMs === null) return
    const id = setInterval(() => {
      if (typeof document !== 'undefined' && document.hidden) return
      void run()
    }, intervalMs)
    return () => clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, intervalMs, run, ...deps])

  const setDataFn = useCallback((updater: (prev: T | null) => T | null) => {
    setData((prev) => updater(prev))
  }, [])

  return { data, error, loading, refreshing, refresh: run, setData: setDataFn }
}

export interface ActionState<A extends unknown[], R> {
  run: (...args: A) => Promise<R | undefined>
  pending: boolean
  error: Error | null
  result: R | null
  reset: () => void
}

/** Wrap a mutating call with pending/error state. */
export function useAction<A extends unknown[], R>(fn: (...args: A) => Promise<R>): ActionState<A, R> {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const [result, setResult] = useState<R | null>(null)
  const fnRef = useRef(fn)
  useEffect(() => {
    fnRef.current = fn
  })
  const run = useCallback(async (...args: A) => {
    setPending(true)
    setError(null)
    try {
      const r = await fnRef.current(...args)
      setResult(r)
      return r
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)))
      return undefined
    } finally {
      setPending(false)
    }
  }, [])
  const reset = useCallback(() => {
    setError(null)
    setResult(null)
  }, [])
  return { run, pending, error, result, reset }
}

/** Re-render on an interval (for clocks and countdowns). Returns Date.now(). */
export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), intervalMs)
    return () => clearInterval(id)
  }, [intervalMs])
  return now
}

/** Track a media query. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window !== 'undefined' ? window.matchMedia(query).matches : false))
  useEffect(() => {
    const mq = window.matchMedia(query)
    const handler = () => setMatches(mq.matches)
    handler()
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [query])
  return matches
}
