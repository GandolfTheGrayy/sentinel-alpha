import { useSyncExternalStore } from 'react'

export type Theme = 'dark' | 'light'

const STORAGE_KEY = 'sentinel.theme'

/** Chart colors per theme — mirrors the CSS tokens in index.css. */
export const CHART_COLORS: Record<
  Theme,
  {
    accent: string
    profit: string
    loss: string
    warn: string
    claude: string
    info: string
    muted: string
    faint: string
    grid: string
    panel: string
    text: string
    neutral: string
  }
> = {
  dark: {
    accent: '#2dd4bf',
    profit: '#2dd4bf',
    loss: '#fb7185',
    warn: '#fbbf24',
    claude: '#a78bfa',
    info: '#60a5fa',
    muted: '#9ca3af',
    faint: '#6b7280',
    grid: '#1c2432',
    panel: '#121826',
    text: '#e5e7eb',
    neutral: '#283142',
  },
  light: {
    accent: '#0d9488',
    profit: '#0d9488',
    loss: '#e11d48',
    warn: '#d97706',
    claude: '#7c3aed',
    info: '#2563eb',
    muted: '#64748b',
    faint: '#94a3b8',
    grid: '#e5e7eb',
    panel: '#ffffff',
    text: '#0f172a',
    neutral: '#e2e8f0',
  },
}

function readStored(): Theme | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'dark' || v === 'light' ? v : null
  } catch {
    return null
  }
}

export function getInitialTheme(): Theme {
  const stored = readStored()
  if (stored) return stored
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: light)').matches) {
    return 'light'
  }
  return 'dark'
}

let current: Theme = getInitialTheme()
const listeners = new Set<() => void>()

export function applyTheme(theme: Theme) {
  const root = document.documentElement
  root.classList.toggle('dark', theme === 'dark')
  root.setAttribute('data-theme', theme)
  root.style.colorScheme = theme
}

export function setTheme(theme: Theme) {
  current = theme
  applyTheme(theme)
  try {
    localStorage.setItem(STORAGE_KEY, theme)
  } catch {
    // ignore storage failures (private mode etc.)
  }
  listeners.forEach((l) => l())
}

export function toggleTheme() {
  setTheme(current === 'dark' ? 'light' : 'dark')
}

function subscribe(cb: () => void) {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

function getSnapshot() {
  return current
}

/** Reactive theme hook. Applies the initial theme on first use. */
export function useTheme() {
  const theme = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return { theme, colors: CHART_COLORS[theme], toggle: toggleTheme, set: setTheme }
}

// Ensure the DOM matches the store at module load (index.html already did this pre-paint).
if (typeof document !== 'undefined') applyTheme(current)
