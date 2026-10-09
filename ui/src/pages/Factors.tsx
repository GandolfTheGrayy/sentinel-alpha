import clsx from 'clsx'
import { Layers } from 'lucide-react'
import { useMemo } from 'react'
import { api } from '../api/client'
import { usePoll } from '../api/hooks'
import type { AttributionReport, ExitReasonStat, FamilyStat, FilterCandidate, RegimeStat } from '../api/types'
import { BucketChart, HBarChart } from '../components/charts/BarCharts'
import { Heatmap } from '../components/charts/Heatmap'
import { Badge, exitReasonTone } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { DataTable, type Column } from '../components/ui/DataTable'
import { EmptyState, ErrorState } from '../components/ui/EmptyState'
import { Meter } from '../components/ui/Gauge'
import { Skeleton, SkeletonChart } from '../components/ui/Skeleton'
import { fmtFrac, fmtNum, fmtR, fmtUsd, signClass } from '../lib/format'
import { useTheme } from '../lib/theme'
import { fmtDateTime, fmtRelative } from '../lib/time'
import { useApp } from '../state/AppContext'

const MIN_TRADES = 30

export function FactorsPage() {
  const { changeTick } = useApp()
  const report = usePoll(() => api.getAttribution(), 120_000, [changeTick])
  const r = report.data

  if (report.error && !r) return <ErrorState error={report.error} onRetry={() => void report.refresh()} />
  if (!r) return <FactorsSkeleton />
  if (!r.features || r.features.length === 0) return <NotEnough report={r} />

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted px-1">
        <span>
          Report generated <span className="text-text">{fmtRelative(r.generated_at)}</span> <span className="text-faint">({fmtDateTime(r.generated_at)})</span>
        </span>
        <span>
          <span className="text-text font-medium">{fmtNum(r.n_trades)}</span> closed trades in the last <span className="text-text">{r.window_days}</span> days
        </span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Feature importance" subtitle="Gradient-boosting permutation importance (share of total)">
          <HBarChart data={r.features.slice(0, 14).map((f) => ({ name: f.name, value: f.importance }))} valueFormatter={(v) => fmtFrac(v, { digits: 1 })} labelWidth={120} />
        </Card>
        <Card title="Logistic coefficients" subtitle="L2 logistic regression on win/loss; sign shows direction">
          {r.logistic && r.logistic.length > 0 ? (
            <HBarChart data={r.logistic.map((c) => ({ name: c.name, value: c.coef }))} diverging valueFormatter={(v) => v.toFixed(2)} labelWidth={120} />
          ) : (
            <EmptyState title="No coefficients" compact />
          )}
        </Card>
      </div>

      <Card title="Expectancy by feature bucket" subtitle="Mean R per bucket with 95% CI; bar opacity scales with the bucket's trade count" flush>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-4 gap-y-2 px-4 pb-4">
          {r.features.map((f) => (
            <div key={f.name} className="min-w-0">
              <div className="flex items-baseline justify-between px-1 mb-1">
                <span className="font-mono text-[12px] font-semibold text-text">{f.name}</span>
                <span className="text-[11px] text-muted">importance {fmtFrac(f.importance, { digits: 1 })}</span>
              </div>
              <BucketChart buckets={f.buckets} />
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 px-1 text-[10.5px] text-faint">
                {f.buckets.map((b) => (
                  <span key={b.label}>
                    {b.label}: n={b.n}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="Filter candidates" subtitle="Rules validated out-of-sample; 'candidate' rules may be proposed to the analyst" flush>
        <FiltersTable filters={r.filters ?? []} />
      </Card>

      <Card title="Hour x weekday" subtitle="Mean expectancy (R) by entry hour (ET) and weekday; hover for counts">
        {r.heatmap ? <Heatmap data={r.heatmap} /> : <EmptyState title="No heatmap" compact />}
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        <Card title="By family" flush>
          <FamilyTable rows={r.by_family ?? []} />
        </Card>
        <Card title="By regime" flush>
          <RegimeTable rows={r.by_regime ?? []} />
        </Card>
        <Card title="By exit reason" flush>
          <ExitTable rows={r.by_exit_reason ?? []} />
        </Card>
      </div>
    </div>
  )
}

function NotEnough({ report }: { report: AttributionReport }) {
  const { colors } = useTheme()
  const n = report.n_trades
  return (
    <Card>
      <EmptyState
        icon={Layers}
        title="Not enough trades yet"
        description={
          <>
            The factor report needs at least {MIN_TRADES} closed trades in the last {report.window_days} days. There are {fmtNum(n)} so far. The attribution job runs daily; this page fills in automatically once the threshold is met.
          </>
        }
        action={
          <div className="w-64">
            <Meter fraction={n / MIN_TRADES} color={colors.info} />
            <div className="text-[11px] text-muted mt-1">
              {fmtNum(n)} / {MIN_TRADES} trades
            </div>
          </div>
        }
      />
    </Card>
  )
}

function FactorsSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-4 w-80" />
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Feature importance">
          <SkeletonChart height={300} />
        </Card>
        <Card title="Logistic coefficients">
          <SkeletonChart height={300} />
        </Card>
      </div>
      <Card title="Expectancy by feature bucket">
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonChart key={i} height={150} />
          ))}
        </div>
      </Card>
    </div>
  )
}

function FiltersTable({ filters }: { filters: FilterCandidate[] }) {
  const columns = useMemo<Column<FilterCandidate>[]>(
    () => [
      { key: 'rule', header: 'Rule', render: (f) => <span className="font-mono text-[11.5px] text-text">{f.rule}</span>, sortValue: (f) => f.rule },
      { key: 'feature', header: 'Feature', render: (f) => <span className="text-muted">{f.feature}</span>, sortValue: (f) => f.feature, hideBelow: 'md' },
      { key: 'removed', header: 'Removed', align: 'right', render: (f) => fmtNum(f.n_removed), sortValue: (f) => f.n_removed },
      { key: 'before', header: 'Before', align: 'right', render: (f) => <span className={signClass(f.expectancy_before)}>{fmtR(f.expectancy_before)}</span>, sortValue: (f) => f.expectancy_before },
      { key: 'after', header: 'After', align: 'right', render: (f) => <span className={signClass(f.expectancy_after)}>{fmtR(f.expectancy_after)}</span>, sortValue: (f) => f.expectancy_after },
      {
        key: 'oos',
        header: 'OOS delta',
        align: 'right',
        render: (f) => <span className={clsx('font-semibold', signClass(f.oos_delta))}>{fmtR(f.oos_delta)}</span>,
        sortValue: (f) => f.oos_delta,
      },
      {
        key: 'verdict',
        header: 'Verdict',
        render: (f) => (
          <Badge tone={f.verdict === 'candidate' ? 'profit' : 'dim'} dot size="xs">
            {f.verdict}
          </Badge>
        ),
        sortValue: (f) => f.verdict,
      },
    ],
    [],
  )
  return <DataTable columns={columns} rows={filters} rowKey={(f) => f.rule} defaultSort={{ key: 'oos', dir: 'desc' }} empty="No filter candidates" dense />
}

function FamilyTable({ rows }: { rows: FamilyStat[] }) {
  const columns = useMemo<Column<FamilyStat>[]>(
    () => [
      { key: 'family', header: 'Family', render: (f) => <span className="font-mono text-[11.5px]">{f.family}</span>, sortValue: (f) => f.family },
      { key: 'n', header: 'n', align: 'right', render: (f) => fmtNum(f.n), sortValue: (f) => f.n },
      { key: 'exp', header: 'Exp R', align: 'right', render: (f) => <span className={clsx('font-semibold', signClass(f.expectancy_r))}>{fmtR(f.expectancy_r)}</span>, sortValue: (f) => f.expectancy_r },
      { key: 'win', header: 'Win', align: 'right', render: (f) => fmtFrac(f.win_rate), sortValue: (f) => f.win_rate },
      { key: 'pnl', header: 'P&L', align: 'right', render: (f) => <span className={signClass(f.pnl)}>{fmtUsd(f.pnl, { sign: true })}</span>, sortValue: (f) => f.pnl },
    ],
    [],
  )
  return <DataTable columns={columns} rows={rows} rowKey={(f) => f.family} defaultSort={{ key: 'exp', dir: 'desc' }} empty="No family stats" dense />
}

function RegimeTable({ rows }: { rows: RegimeStat[] }) {
  const columns = useMemo<Column<RegimeStat>[]>(
    () => [
      { key: 'regime', header: 'Regime', render: (f) => <span className="font-mono text-[11.5px]">{f.regime}</span>, sortValue: (f) => f.regime },
      { key: 'n', header: 'n', align: 'right', render: (f) => fmtNum(f.n), sortValue: (f) => f.n },
      { key: 'exp', header: 'Exp R', align: 'right', render: (f) => <span className={clsx('font-semibold', signClass(f.expectancy_r))}>{fmtR(f.expectancy_r)}</span>, sortValue: (f) => f.expectancy_r },
      { key: 'win', header: 'Win', align: 'right', render: (f) => fmtFrac(f.win_rate), sortValue: (f) => f.win_rate },
    ],
    [],
  )
  return <DataTable columns={columns} rows={rows} rowKey={(f) => f.regime} defaultSort={{ key: 'exp', dir: 'desc' }} empty="No regime stats" dense />
}

function ExitTable({ rows }: { rows: ExitReasonStat[] }) {
  const columns = useMemo<Column<ExitReasonStat>[]>(
    () => [
      { key: 'reason', header: 'Exit reason', render: (f) => <Badge tone={exitReasonTone(f.exit_reason)} size="xs">{f.exit_reason}</Badge>, sortValue: (f) => f.exit_reason },
      { key: 'n', header: 'n', align: 'right', render: (f) => fmtNum(f.n), sortValue: (f) => f.n },
      { key: 'exp', header: 'Exp R', align: 'right', render: (f) => <span className={clsx('font-semibold', signClass(f.expectancy_r))}>{fmtR(f.expectancy_r)}</span>, sortValue: (f) => f.expectancy_r },
    ],
    [],
  )
  return <DataTable columns={columns} rows={rows} rowKey={(f) => f.exit_reason} defaultSort={{ key: 'n', dir: 'desc' }} empty="No exit stats" dense />
}
