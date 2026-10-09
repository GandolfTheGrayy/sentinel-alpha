import clsx from 'clsx'
import { Activity, Bitcoin, Briefcase, Gauge, Landmark, Percent, TrendingUp, Wallet } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { usePoll } from '../api/hooks'
import type { EquityPoint, EquityRange, PopulationState, VariantStatus } from '../api/types'
import { ActivityFeed } from '../components/ActivityFeed'
import { EquityChart, EquityLegend } from '../components/charts/EquityChart'
import { PositionsTable } from '../components/PositionsTable'
import { Dot, variantStatusTone } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { Segmented } from '../components/ui/Chips'
import { ArcGauge, Meter } from '../components/ui/Gauge'
import { KpiTile } from '../components/ui/KpiTile'
import { Skeleton } from '../components/ui/Skeleton'
import { fmtFrac, fmtNum, fmtPct, fmtUsd, signClass } from '../lib/format'
import { useTheme } from '../lib/theme'
import { fmtCountdown, fmtRelative, fmtTime, fmtWeekday, nyClock } from '../lib/time'
import { useApp, useClock } from '../state/AppContext'

const RANGES: { value: EquityRange; label: string }[] = [
  { value: '1d', label: '1d' },
  { value: '1w', label: '1w' },
  { value: '1m', label: '1m' },
  { value: 'all', label: 'All' },
]

export function OverviewPage() {
  const { status, liveEquity, liveEvents, connection } = useApp()
  const [range, setRange] = useState<EquityRange>('1d')
  const equity = usePoll(() => api.getEquity(range), 60_000, [range])
  const positions = usePoll(() => api.getPositions(), 10_000)
  const events = usePoll(() => api.getEvents({ limit: 60 }), 30_000)

  // Append streamed equity points newer than the last fetched one.
  const chartData = useMemo<EquityPoint[] | null>(() => {
    if (!equity.data) return null
    const last = equity.data[equity.data.length - 1]?.ts ?? ''
    const extra = liveEquity.filter((p) => p.ts > last)
    return extra.length ? [...equity.data, ...extra] : equity.data
  }, [equity.data, liveEquity])

  const acct = status?.account
  const loading = !status
  const variantCount = useMemo(() => new Set((positions.data ?? []).map((p) => p.variant_id)).size, [positions.data])

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-3">
        <KpiTile label="Equity" icon={Wallet} loading={loading} value={fmtUsd(acct?.equity)} sub={acct ? `Cash ${fmtUsd(acct.cash)}` : undefined} />
        <KpiTile
          label="Day P&L"
          icon={TrendingUp}
          loading={loading}
          value={fmtUsd(acct?.day_pnl, { sign: true, decimals: 2 })}
          valueClassName={signClass(acct?.day_pnl)}
          sub={fmtPct(acct?.day_pnl_pct, { sign: true })}
          subClassName={signClass(acct?.day_pnl_pct)}
        />
        <KpiTile
          label="Week P&L"
          icon={TrendingUp}
          loading={loading}
          value={fmtUsd(acct?.week_pnl, { sign: true, decimals: 2 })}
          valueClassName={signClass(acct?.week_pnl)}
          sub={fmtPct(acct?.week_pnl_pct, { sign: true })}
          subClassName={signClass(acct?.week_pnl_pct)}
        />
        <KpiTile
          label="Total P&L"
          icon={Landmark}
          loading={loading}
          value={fmtUsd(acct?.total_pnl, { sign: true, decimals: 2 })}
          valueClassName={signClass(acct?.total_pnl)}
          sub={acct ? `${fmtPct(acct.total_pnl_pct, { sign: true })} on ${fmtUsd(acct.starting_equity, { compact: true })} start` : undefined}
          subClassName={signClass(acct?.total_pnl_pct)}
        />
        <KpiTile
          label="Open positions"
          icon={Briefcase}
          loading={loading}
          value={fmtNum(acct?.open_positions)}
          sub={positions.data ? `across ${variantCount} variant${variantCount === 1 ? '' : 's'}` : undefined}
        />
        <KpiTile
          label="Gross exposure"
          icon={Percent}
          loading={loading}
          value={fmtPct(acct?.gross_exposure_pct, { digits: 1 })}
          sub="of equity · cap 60%"
          footer={acct ? <Meter fraction={acct.gross_exposure_pct / 100} color={acct.gross_exposure_pct > 60 ? 'var(--loss)' : 'var(--accent)'} markers={[0.6]} /> : undefined}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card
          className="xl:col-span-2"
          title="Equity curve"
          subtitle={<EquityLegend />}
          actions={<Segmented options={RANGES} value={range} onChange={setRange} />}
        >
          <EquityChart data={chartData} loading={equity.loading} height={280} />
        </Card>
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-4">
          <MarketCard />
          <PopulationCard population={status?.population ?? null} />
          <BudgetCard className="sm:col-span-2 xl:col-span-1" />
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <Card className="xl:col-span-2" title="Open positions" subtitle={positions.data ? `${positions.data.length} lots · refreshes every 10 s` : undefined} flush>
          <PositionsTable positions={positions.data} loading={positions.loading} compact />
        </Card>
        <Card
          title={
            <span className="inline-flex items-center gap-2">
              Activity
              {(connection === 'connected' || connection === 'mock') && <Dot tone="profit" pulse />}
            </span>
          }
          subtitle="Live feed from the engine"
          actions={
            <Link to="/trades" className="text-[11.5px] text-accent hover:underline">
              Trades
            </Link>
          }
          flush
        >
          <ActivityFeed fetched={events.data} live={liveEvents} loading={events.loading} maxHeight={420} />
        </Card>
      </div>
    </div>
  )
}

function MarketCard() {
  const { status } = useApp()
  const { now, isSim } = useClock(1000)
  const m = status?.market
  const open = m?.equities_open ?? false
  const sessionLabel = !m ? '' : m.session === 'regular' ? 'Regular session' : m.session === 'pre' ? 'Pre-market' : m.session === 'post' ? 'After hours' : 'Closed'
  return (
    <Card
      title="Market"
      subtitle={
        <span>
          {nyClock(new Date(now))} <span className="text-faint">New York{isSim ? ' · simulated clock' : ''}</span>
        </span>
      }
    >
      {!m ? (
        <div className="space-y-2">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-36" />
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex items-start gap-3">
            <div className={clsx('h-8 w-8 rounded-lg flex items-center justify-center shrink-0', open ? 'bg-profit/15 text-profit' : 'bg-muted/10 text-muted')}>
              <Landmark className="h-4 w-4" aria-hidden />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-text">US equities</span>
                <Dot tone={open ? 'profit' : m.session === 'closed' ? 'dim' : 'warn'} pulse={open} />
                <span className="text-[11.5px] text-muted">{sessionLabel}</span>
              </div>
              <div className="text-[12px] text-muted mt-0.5">
                {open ? (
                  <>
                    Closes in <span className="text-text font-medium">{fmtCountdown(m.next_close, now)}</span>
                    <span className="text-faint"> · {fmtTime(m.next_close)} ET</span>
                  </>
                ) : (
                  <>
                    Opens in <span className="text-text font-medium">{fmtCountdown(m.next_open, now)}</span>
                    <span className="text-faint"> · {fmtWeekday(m.next_open)} {fmtTime(m.next_open)} ET</span>
                  </>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-start gap-3">
            <div className="h-8 w-8 rounded-lg bg-profit/15 text-profit flex items-center justify-center shrink-0">
              <Bitcoin className="h-4 w-4" aria-hidden />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="text-[13px] font-medium text-text">Crypto</span>
                <Dot tone={m.crypto_open ? 'profit' : 'dim'} pulse={m.crypto_open} />
                <span className="text-[11.5px] text-muted">{m.crypto_open ? 'Trading 24/7' : 'Halted'}</span>
              </div>
              <div className="text-[12px] text-muted mt-0.5">
                {status?.universe.crypto.length ?? 0} pairs · {status?.universe.equities.length ?? 0} equities in universe
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}

const POP_ORDER: VariantStatus[] = ['active', 'incubating', 'probation', 'paused', 'retired']

function PopulationCard({ population }: { population: PopulationState | null }) {
  const total = population ? POP_ORDER.reduce((a, k) => a + population[k], 0) : 0
  const live = population ? population.active + population.incubating + population.probation : 0
  return (
    <Card title="Population" subtitle={population ? `${live} trading · ${total} total` : undefined} actions={<Link to="/tournament" className="text-[11.5px] text-accent hover:underline">Tournament</Link>}>
      {!population ? (
        <div className="space-y-2">
          <Skeleton className="h-2.5 w-full" />
          <Skeleton className="h-4 w-48" />
        </div>
      ) : (
        <div>
          <div className="flex h-2.5 rounded-full overflow-hidden gap-px bg-muted/10">
            {POP_ORDER.map((k) =>
              population[k] > 0 ? (
                <div key={k} className={clsx('h-full', barClass(k))} style={{ width: `${(population[k] / Math.max(total, 1)) * 100}%` }} title={`${k}: ${population[k]}`} />
              ) : null,
            )}
          </div>
          <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 mt-3">
            {POP_ORDER.map((k) => (
              <li key={k} className="flex items-center justify-between text-[12px]">
                <span className="inline-flex items-center gap-1.5 text-muted">
                  <Dot tone={variantStatusTone(k)} />
                  {k}
                </span>
                <span className="font-semibold text-text">{population[k]}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  )
}

function barClass(k: VariantStatus): string {
  switch (k) {
    case 'active':
      return 'bg-profit'
    case 'incubating':
      return 'bg-info'
    case 'probation':
      return 'bg-warn'
    case 'paused':
      return 'bg-muted'
    case 'retired':
      return 'bg-faint/50'
  }
}

function BudgetCard({ className }: { className?: string }) {
  const { status } = useApp()
  const { colors } = useTheme()
  const b = status?.budget
  const frac = b && b.weekly_cap_usd > 0 ? b.spent_usd / b.weekly_cap_usd : 0
  const color = frac >= 0.95 ? colors.loss : frac >= 0.8 ? colors.warn : colors.claude
  return (
    <Card className={className} title="Claude budget" subtitle={b ? `${fmtFrac(b.share_of_plan, { digits: 0 })} of plan allowance per week` : undefined} actions={<Link to="/settings" className="text-[11.5px] text-accent hover:underline">Budget</Link>}>
      {!b ? (
        <div className="flex gap-4">
          <Skeleton className="h-20 w-40" />
          <div className="flex-1 space-y-2">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row xl:flex-col 2xl:flex-row items-center gap-4">
          <ArcGauge fraction={frac} color={color} size={150} label={`Spent ${fmtUsd(b.spent_usd, { decimals: 2 })} of ${fmtUsd(b.weekly_cap_usd, { decimals: 2 })}`}>
            <div className="text-[18px] font-semibold text-text leading-6">{fmtUsd(b.spent_usd, { decimals: 2 })}</div>
            <div className="text-[11px] text-muted">of {fmtUsd(b.weekly_cap_usd, { decimals: 2 })} cap</div>
          </ArcGauge>
          <div className="flex-1 w-full space-y-2 text-[12px]">
            <div className="flex items-center justify-between">
              <span className="text-muted">Remaining</span>
              <span className={clsx('font-semibold', b.remaining_usd <= 0 ? 'text-loss' : 'text-text')}>{fmtUsd(b.remaining_usd, { decimals: 2 })}</span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-muted">Next analyst</span>
              <span className="text-text text-right">
                {fmtRelative(b.next_analyst_run)} <span className="text-faint">· {fmtWeekday(b.next_analyst_run)} {fmtTime(b.next_analyst_run)}</span>
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-muted">Next strategist</span>
              <span className="text-text text-right">
                {fmtRelative(b.next_strategist_run)} <span className="text-faint">· {fmtWeekday(b.next_strategist_run)} {fmtTime(b.next_strategist_run)}</span>
              </span>
            </div>
            {frac >= 0.8 && (
              <div className={clsx('flex items-center gap-1.5 text-[11.5px]', frac >= 0.95 ? 'text-loss' : 'text-warn')}>
                <Gauge className="h-3.5 w-3.5" aria-hidden />
                {frac >= 0.95 ? 'Cap reached — research sessions are skipped' : 'Approaching the weekly cap'}
              </div>
            )}
            {frac < 0.8 && (
              <div className="flex items-center gap-1.5 text-[11.5px] text-muted">
                <Activity className="h-3.5 w-3.5" aria-hidden />
                Research runs on schedule
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}
