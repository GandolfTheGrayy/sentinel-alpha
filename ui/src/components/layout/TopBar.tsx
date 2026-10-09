import clsx from 'clsx'
import { Clock, Moon, Sun, Wifi, WifiOff } from 'lucide-react'
import { useLocation } from 'react-router-dom'
import type { ConnectionState } from '../../api/client'
import type { EngineState, MarketState, Mode } from '../../api/types'
import { fmtUsd } from '../../lib/format'
import { navFor } from '../../lib/nav'
import { useTheme } from '../../lib/theme'
import { fmtCountdown, nyClock } from '../../lib/time'
import { useApp, useClock } from '../../state/AppContext'
import { Badge, Dot, type Tone } from '../ui/Badge'
import { IconButton } from '../ui/Button'
import { MiniRing } from '../ui/Gauge'
import { Skeleton } from '../ui/Skeleton'

export function TopBar() {
  const { status, connection, apiMode } = useApp()
  const { pathname } = useLocation()
  const nav = navFor(pathname)

  return (
    <header className="min-h-14 shrink-0 border-b border-border bg-panel/80 backdrop-blur flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-2 sm:px-4">
      <div className="min-w-0 flex-1 basis-[120px]">
        <h1 className="text-[14px] font-semibold text-text leading-5 truncate">{nav.label}</h1>
        <p className="text-[11px] text-muted leading-4 truncate hidden md:block">{nav.description}</p>
      </div>
      <div className="flex items-center gap-2 overflow-x-auto [scrollbar-width:none] [&::-webkit-scrollbar]:hidden shrink-0 max-w-full">
        {status ? (
          <>
            <ModeBadge mode={status.mode} />
            <EnginePill engine={status.engine} />
            <MarketPill market={status.market} />
            <BudgetMini spent={status.budget.spent_usd} cap={status.budget.weekly_cap_usd} />
          </>
        ) : (
          <>
            <Skeleton className="h-6 w-14" />
            <Skeleton className="h-6 w-20" />
            <Skeleton className="h-6 w-28" />
          </>
        )}
        <ConnectionPill state={connection} mock={apiMode === 'mock'} />
        <ThemeToggle />
      </div>
    </header>
  )
}

function ModeBadge({ mode }: { mode: Mode }) {
  const tone: Tone = mode === 'live' ? 'loss' : mode === 'sim' ? 'info' : 'profit'
  return (
    <Badge tone={tone} size="md" className="shrink-0 uppercase tracking-wide font-semibold">
      {mode}
    </Badge>
  )
}

function EnginePill({ engine }: { engine: EngineState }) {
  const state = engine.halted ? 'halted' : engine.paused ? 'paused' : engine.running ? 'running' : 'stopped'
  const tone: Tone = state === 'running' ? 'profit' : state === 'paused' ? 'warn' : 'loss'
  return (
    <Badge tone={tone} size="md" dot pulse={state === 'running'} className="shrink-0" title={`Engine ${state} · ${engine.tick_count} ticks · ${engine.errors_1h} errors in the last hour`}>
      <span className="hidden sm:inline">engine </span>
      {state}
      {engine.errors_1h > 0 && <span className="text-loss">· {engine.errors_1h} err</span>}
    </Badge>
  )
}

function MarketPill({ market }: { market: MarketState }) {
  const { now, isSim } = useClock(1000)
  const open = market.equities_open
  const label = market.session === 'regular' ? 'open' : market.session
  const countdown = open ? `closes ${fmtCountdown(market.next_close, now)}` : `opens ${fmtCountdown(market.next_open, now)}`
  return (
    <div
      className={clsx(
        'inline-flex items-center gap-2 rounded-md border border-border bg-panel-2 h-7 px-2 text-[11.5px] whitespace-nowrap shrink-0',
      )}
      title={`US equities ${label} · ${countdown} · crypto 24/7${isSim ? ' · engine runs on a simulated clock; all times are sim time' : ''}`}
    >
      <Dot tone={open ? 'profit' : market.session === 'closed' ? 'dim' : 'warn'} pulse={open} />
      <span className="hidden xl:inline text-muted">NYSE {label}</span>
      <span className="hidden xl:inline text-faint">· {countdown}</span>
      <span className="inline-flex items-center gap-1 text-text font-medium">
        <Clock className="h-3 w-3 text-faint" aria-hidden />
        {nyClock(new Date(now))}
        <span className="text-faint font-normal">ET</span>
      </span>
      {isSim && (
        <Badge tone="info" size="xs" className="uppercase tracking-wide" title="Simulated clock: the engine's virtual time, advanced locally between ticks">
          sim clock
        </Badge>
      )}
    </div>
  )
}

function BudgetMini({ spent, cap }: { spent: number; cap: number }) {
  const { colors } = useTheme()
  const frac = cap > 0 ? spent / cap : 0
  const color = frac >= 0.95 ? colors.loss : frac >= 0.8 ? colors.warn : colors.claude
  return (
    <div
      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel-2 h-7 px-2 text-[11.5px] whitespace-nowrap shrink-0"
      title={`Claude budget: ${fmtUsd(spent, { decimals: 2 })} of ${fmtUsd(cap, { decimals: 2 })} weekly cap`}
    >
      <MiniRing fraction={frac} color={color} />
      <span className="text-text font-medium">{fmtUsd(spent, { decimals: 2 })}</span>
      <span className="hidden lg:inline text-faint">/ {fmtUsd(cap, { decimals: 2 })}</span>
    </div>
  )
}

function ConnectionPill({ state, mock }: { state: ConnectionState; mock: boolean }) {
  const cfg = mock
    ? { tone: 'warn' as Tone, label: 'demo data', Icon: Wifi, pulse: false }
    : state === 'connected'
      ? { tone: 'profit' as Tone, label: 'live', Icon: Wifi, pulse: false }
      : state === 'reconnecting'
        ? { tone: 'warn' as Tone, label: 'reconnecting', Icon: Wifi, pulse: true }
        : state === 'unreachable'
          ? { tone: 'loss' as Tone, label: 'backend unreachable', Icon: WifiOff, pulse: false }
          : { tone: 'dim' as Tone, label: 'connecting', Icon: Wifi, pulse: true }
  return (
    <Badge tone={cfg.tone} size="md" className="shrink-0" title={mock ? 'Mock API active' : `SSE stream: ${cfg.label}`}>
      <cfg.Icon className={clsx('h-3.5 w-3.5', cfg.pulse && 'animate-pulse-dot')} aria-hidden />
      <span className="hidden sm:inline">{cfg.label}</span>
    </Badge>
  )
}

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <IconButton label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'} onClick={toggle} className="shrink-0 border border-border bg-panel-2">
      {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </IconButton>
  )
}
