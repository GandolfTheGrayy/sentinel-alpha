import clsx from 'clsx'
import { Trophy } from 'lucide-react'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import { usePoll } from '../api/hooks'
import { VARIANT_STATUSES, hasTrades, type Variant, type VariantStatus } from '../api/types'
import { Badge, OriginBadge, StatusBadge } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { Chips } from '../components/ui/Chips'
import { DataTable, type Column } from '../components/ui/DataTable'
import { ErrorState } from '../components/ui/EmptyState'
import { Meter } from '../components/ui/Gauge'
import { Sparkline } from '../components/ui/Sparkline'
import { VariantDrawer } from '../components/VariantDrawer'
import { DASH, fmtFrac, fmtNum, fmtPct, fmtR, fmtUsd, fmtX, signClass } from '../lib/format'
import { useApp } from '../state/AppContext'

export function TournamentPage() {
  const { changeTick } = useApp()
  const variants = usePoll(() => api.getVariants(), 30_000, [changeTick])
  const [family, setFamily] = useState<string | null>(null)
  const [status, setStatus] = useState<VariantStatus | null>(null)
  const [selected, setSelected] = useState<string | null>(null)

  const families = useMemo(() => {
    const counts = new Map<string, number>()
    for (const v of variants.data ?? []) counts.set(v.family, (counts.get(v.family) ?? 0) + 1)
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([value, count]) => ({ value, label: value, count }))
  }, [variants.data])

  const statusCounts = useMemo(() => {
    const counts: Record<VariantStatus, number> = { active: 0, incubating: 0, probation: 0, paused: 0, retired: 0 }
    for (const v of variants.data ?? []) counts[v.status] += 1
    return counts
  }, [variants.data])

  const rows = useMemo(() => {
    let r = variants.data ?? []
    if (family) r = r.filter((v) => v.family === family)
    if (status) r = r.filter((v) => v.status === status)
    return r
  }, [variants.data, family, status])

  const totals = useMemo(() => {
    const live = rows.filter((v) => v.status !== 'retired')
    return {
      alloc: live.reduce((a, v) => a + v.allocation, 0),
      pnl: rows.reduce((a, v) => a + v.metrics.pnl, 0),
      n: rows.reduce((a, v) => a + v.metrics.n, 0),
    }
  }, [rows])

  const columns = useMemo<Column<Variant>[]>(
    () => [
      {
        key: 'variant',
        header: 'Variant',
        render: (v) => (
          <div className="min-w-[140px]">
            <div className="flex items-center gap-1.5">
              <span className="font-mono font-semibold text-text">{v.id}</span>
              <OriginBadge origin={v.origin} size="xs" />
              {v.is_control && (
                <Badge tone="warn" size="xs">
                  control
                </Badge>
              )}
            </div>
            <div className="text-[11px] text-muted truncate max-w-[200px]" title={`${v.name} · ${v.timeframe} · ${v.markets.join('+')}`}>
              {v.name.replace(`${v.family} `, '')} · {v.timeframe} · {v.markets.join('+')}
            </div>
          </div>
        ),
        sortValue: (v) => v.id,
      },
      { key: 'status', header: 'Status', render: (v) => <StatusBadge status={v.status} size="xs" />, sortValue: (v) => VARIANT_STATUSES.indexOf(v.status) },
      {
        key: 'allocation',
        header: 'Allocation',
        align: 'right',
        render: (v) => (
          <div className="flex items-center justify-end gap-2 min-w-[96px]">
            <Meter fraction={v.allocation / 0.2} color={v.status === 'active' ? 'var(--accent)' : 'var(--muted)'} className="w-12" height={5} />
            <span className={clsx('w-10 text-right', v.allocation === 0 ? 'text-faint' : 'text-text')}>{fmtFrac(v.allocation, { digits: 1 })}</span>
          </div>
        ),
        sortValue: (v) => v.allocation,
      },
      {
        key: 'exp',
        header: 'Exp R',
        align: 'right',
        render: (v) =>
          hasTrades(v.metrics) ? (
            <div>
              <div className={clsx('font-semibold', signClass(v.metrics.expectancy_r))}>{fmtR(v.metrics.expectancy_r)}</div>
              {v.metrics.expectancy_ci && (
                <div className="text-[10px] text-faint">
                  {fmtR(v.metrics.expectancy_ci[0], 2)} … {fmtR(v.metrics.expectancy_ci[1], 2)}
                </div>
              )}
            </div>
          ) : (
            <div>
              <div className="text-muted">{DASH}</div>
              <div className="text-[10px] text-faint">no trades yet</div>
            </div>
          ),
        sortValue: (v) => v.metrics.expectancy_r,
      },
      {
        key: 'win',
        header: 'Win rate',
        align: 'right',
        render: (v) =>
          hasTrades(v.metrics) ? (
            <div>
              <div className="text-text">{fmtFrac(v.metrics.win_rate)}</div>
              {v.metrics.win_rate_ci && (
                <div className="text-[10px] text-faint">
                  {fmtFrac(v.metrics.win_rate_ci[0])} … {fmtFrac(v.metrics.win_rate_ci[1])}
                </div>
              )}
            </div>
          ) : (
            <span className="text-muted">{DASH}</span>
          ),
        sortValue: (v) => v.metrics.win_rate,
        hideBelow: 'xl',
      },
      { key: 'pf', header: 'PF', align: 'right', render: (v) => fmtX(v.metrics.profit_factor), sortValue: (v) => v.metrics.profit_factor, hideBelow: 'md' },
      { key: 'sharpe', header: 'Sharpe', align: 'right', render: (v) => fmtNum(v.metrics.sharpe, 2), sortValue: (v) => v.metrics.sharpe, hideBelow: '2xl' },
      { key: 'dd', header: 'Max DD', align: 'right', render: (v) => <span className="text-loss">{fmtPct(v.metrics.max_dd_pct, { digits: 1 })}</span>, sortValue: (v) => v.metrics.max_dd_pct, hideBelow: 'xl' },
      { key: 'n', header: 'n', align: 'right', render: (v) => fmtNum(v.metrics.n), sortValue: (v) => v.metrics.n, hideBelow: 'lg' },
      {
        key: 'pnl',
        header: 'P&L',
        align: 'right',
        render: (v) => <span className={clsx('font-semibold', signClass(v.metrics.pnl))}>{fmtUsd(v.metrics.pnl, { sign: true })}</span>,
        sortValue: (v) => v.metrics.pnl,
      },
      { key: 'spark', header: 'Curve', align: 'right', render: (v) => <Sparkline data={v.sparkline} />, hideBelow: '2xl' },
      { key: 'open', header: 'Open', align: 'right', render: (v) => <span className={v.open_positions ? 'text-text' : 'text-faint'}>{v.open_positions}</span>, sortValue: (v) => v.open_positions, hideBelow: '2xl' },
    ],
    [],
  )

  return (
    <div className="space-y-4">
      <Card
        title="Leaderboard"
        subtitle={
          variants.data
            ? `${rows.length} of ${variants.data.length} variants · ${fmtFrac(totals.alloc, { digits: 0 })} allocated · ${fmtNum(totals.n)} trades · ${fmtUsd(totals.pnl, { sign: true })}`
            : 'Loading variants…'
        }
        flush
      >
        <div className="px-4 pb-3 flex flex-col gap-2">
          <Chips options={families} value={family} onChange={setFamily} allLabel="All families" />
          <Chips
            options={VARIANT_STATUSES.map((s) => ({ value: s, label: s, count: statusCounts[s] }))}
            value={status}
            onChange={setStatus}
            allLabel="Any status"
            size="xs"
          />
        </div>
        {variants.error && !variants.data ? (
          <ErrorState error={variants.error} onRetry={() => void variants.refresh()} />
        ) : (
          <DataTable
            columns={columns}
            rows={rows}
            rowKey={(v) => v.id}
            loading={variants.loading}
            onRowClick={(v) => setSelected(v.id)}
            activeKey={selected}
            defaultSort={{ key: 'allocation', dir: 'desc' }}
            empty="No variants match"
            emptyDescription="Try clearing the family or status filter."
            skeletonRows={10}
          />
        )}
      </Card>
      <p className="text-[11.5px] text-muted flex items-center gap-1.5 px-1">
        <Trophy className="h-3.5 w-3.5" aria-hidden />
        Click a row for parameters, lineage, the P&L curve, last trades and pause / resume / retire actions. Metrics use all closed trades; CIs are 95%.
      </p>
      <VariantDrawer id={selected} onClose={() => setSelected(null)} onChanged={() => void variants.refresh()} />
    </div>
  )
}
