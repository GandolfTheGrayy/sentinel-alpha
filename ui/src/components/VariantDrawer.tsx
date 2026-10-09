import clsx from 'clsx'
import { Archive, Pause, Play } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { api } from '../api/client'
import { useAction, usePoll } from '../api/hooks'
import type { Trade, VariantDetail, VariantStatusChange } from '../api/types'
import { fmtDuration, fmtFrac, fmtNum, fmtPct, fmtR, fmtUsd, fmtX, signClass } from '../lib/format'
import { fmtDate, fmtDateTime } from '../lib/time'
import { PnlCurve } from './charts/PnlCurve'
import { Badge, OriginBadge, StatusBadge, exitReasonTone } from './ui/Badge'
import { Button } from './ui/Button'
import { SectionTitle } from './ui/Card'
import { useConfirm } from './ui/Confirm'
import { DataTable, type Column } from './ui/DataTable'
import { Drawer } from './ui/Drawer'
import { ErrorState } from './ui/EmptyState'
import { KeyValue } from './ui/KeyValue'
import { Skeleton, SkeletonChart, SkeletonText } from './ui/Skeleton'
import { useToast } from './ui/Toast'

interface VariantDrawerProps {
  id: string | null
  onClose: () => void
  onChanged?: () => void
}

export function VariantDrawer({ id, onClose, onChanged }: VariantDrawerProps) {
  const detail = usePoll<VariantDetail | null>(() => (id ? api.getVariant(id) : Promise.resolve(null)), null, [id])
  const v = detail.data
  const confirm = useConfirm()
  const toast = useToast()
  const change = useAction((target: string, status: VariantStatusChange) => api.setVariantStatus(target, status))

  useEffect(() => {
    if (change.error) toast({ tone: 'error', title: 'Status change failed', message: change.error.message })
  }, [change.error, toast])

  const act = async (status: VariantStatusChange) => {
    if (!v) return
    const verb = status === 'active' ? 'Resume' : status === 'paused' ? 'Pause' : 'Retire'
    const ok = await confirm({
      title: `${verb} ${v.id}?`,
      message:
        status === 'retired'
          ? 'The variant stops trading permanently and its sleeve is released. Open positions are closed by the engine on the next tick.'
          : status === 'paused'
            ? 'No new entries will be taken until it is resumed. Open positions continue to be managed.'
            : 'The variant rejoins the tournament and receives the exploration floor allocation.',
      confirmLabel: verb,
      tone: status === 'retired' ? 'danger' : status === 'paused' ? 'warn' : 'primary',
    })
    if (!ok) return
    const updated = await change.run(v.id, status)
    if (updated) {
      toast({ tone: 'success', title: `${v.id} is now ${updated.status}` })
      detail.setData((prev) => (prev ? { ...prev, ...updated } : prev))
      onChanged?.()
    }
  }

  const tradeColumns = useMemo<Column<Trade>[]>(
    () => [
      { key: 'ts', header: 'Closed', render: (t) => <span className="text-muted">{fmtDateTime(t.exit_ts)}</span> },
      {
        key: 'sym',
        header: 'Symbol',
        render: (t) => (
          <span className="inline-flex items-center gap-1.5">
            <span className="font-medium">{t.symbol}</span>
            <Badge tone={t.side === 'long' ? 'profit' : 'loss'} size="xs">
              {t.side}
            </Badge>
          </span>
        ),
      },
      { key: 'r', header: 'R', align: 'right', render: (t) => <span className={clsx('font-semibold', signClass(t.pnl_r))}>{fmtR(t.pnl_r)}</span> },
      { key: 'pnl', header: 'P&L', align: 'right', render: (t) => <span className={signClass(t.pnl)}>{fmtUsd(t.pnl, { sign: true, decimals: 2 })}</span> },
      { key: 'exit', header: 'Exit', render: (t) => <Badge tone={exitReasonTone(t.exit_reason)} size="xs">{t.exit_reason}</Badge> },
    ],
    [],
  )

  const open = id !== null
  const m = v?.metrics

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={v ? v.id : id ?? ''}
      subtitle={v ? v.name : undefined}
      actions={
        v && (
          <div className="flex items-center gap-1.5">
            {v.status === 'paused' ? (
              <Button size="sm" variant="primary" icon={<Play className="h-3.5 w-3.5" />} loading={change.pending} onClick={() => act('active')}>
                Resume
              </Button>
            ) : v.status !== 'retired' ? (
              <Button size="sm" variant="warn" icon={<Pause className="h-3.5 w-3.5" />} loading={change.pending} onClick={() => act('paused')}>
                Pause
              </Button>
            ) : null}
            {v.status !== 'retired' && (
              <Button size="sm" variant="danger" icon={<Archive className="h-3.5 w-3.5" />} loading={change.pending} onClick={() => act('retired')}>
                Retire
              </Button>
            )}
          </div>
        )
      }
    >
      {detail.error && !v ? (
        <ErrorState error={detail.error} onRetry={() => void detail.refresh()} />
      ) : !v ? (
        <div className="space-y-4">
          <div className="flex gap-2">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-5 w-24" />
          </div>
          <SkeletonText lines={4} />
          <SkeletonChart height={180} />
          <SkeletonText lines={6} />
        </div>
      ) : (
        <div className="space-y-6">
          <div className="flex flex-wrap items-center gap-1.5">
            <StatusBadge status={v.status} />
            <OriginBadge origin={v.origin} />
            {v.is_control && <Badge tone="warn">control</Badge>}
            <Badge tone="neutral">{v.family}</Badge>
            <Badge tone="dim">{v.timeframe}</Badge>
            {v.markets.map((mk) => (
              <Badge key={mk} tone="dim">
                {mk}
              </Badge>
            ))}
            <span className="text-[11.5px] text-muted ml-auto">created {fmtDate(v.created_at, true)}</span>
          </div>

          {m && (
            <section>
              <SectionTitle className="mb-2">Metrics (closed trades)</SectionTitle>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <Stat label="Expectancy" value={fmtR(m.expectancy_r)} cls={signClass(m.expectancy_r)} sub={`CI ${fmtR(m.expectancy_ci[0])} … ${fmtR(m.expectancy_ci[1])}`} />
                <Stat label="Win rate" value={fmtFrac(m.win_rate)} sub={`CI ${fmtFrac(m.win_rate_ci[0])} … ${fmtFrac(m.win_rate_ci[1])}`} />
                <Stat label="P&L" value={fmtUsd(m.pnl, { sign: true })} cls={signClass(m.pnl)} sub={`${fmtNum(m.n)} trades`} />
                <Stat label="Allocation" value={fmtFrac(v.allocation, { digits: 1 })} sub={`${v.open_positions} open`} />
              </div>
              <KeyValue
                className="mt-3"
                cols={2}
                dense
                items={[
                  { k: 'Profit factor', v: fmtX(m.profit_factor) },
                  { k: 'Sharpe', v: fmtNum(m.sharpe, 2) },
                  { k: 'Max drawdown', v: <span className="text-loss">{fmtPct(m.max_dd_pct)}</span> },
                  { k: 'Avg hold', v: fmtDuration(m.avg_hold_minutes) },
                  { k: 'Last 30: expectancy', v: <span className={signClass(m.last_30.expectancy_r)}>{fmtR(m.last_30.expectancy_r)}</span> },
                  { k: 'Last 30: win rate', v: fmtFrac(m.last_30.win_rate) },
                  { k: 'Last 30: P&L', v: <span className={signClass(m.last_30.pnl)}>{fmtUsd(m.last_30.pnl, { sign: true })}</span> },
                  { k: 'Last 30: n', v: fmtNum(m.last_30.n) },
                ]}
              />
            </section>
          )}

          <section>
            <SectionTitle className="mb-2">Cumulative P&L</SectionTitle>
            <PnlCurve data={v.equity_curve} id={`v-${v.id.replace(/[^a-z0-9]/gi, '')}`} />
          </section>

          <section>
            <SectionTitle className="mb-2">Parameters</SectionTitle>
            {Object.keys(v.params).length === 0 ? (
              <p className="text-[12px] text-muted">No parameters.</p>
            ) : (
              <KeyValue dense cols={2} items={Object.entries(v.params).map(([k, val]) => ({ k, v: String(val), mono: true }))} />
            )}
          </section>

          <section>
            <SectionTitle className="mb-2">Lineage</SectionTitle>
            <ol className="relative border-l border-border ml-1.5 space-y-2">
              {v.lineage.map((l) => (
                <li key={l.id} className="pl-4 relative">
                  <span className={clsx('absolute -left-[5px] top-1.5 h-2 w-2 rounded-full', l.id === v.id ? 'bg-accent' : 'bg-faint')} />
                  <div className="flex flex-wrap items-center gap-1.5 text-[12px]">
                    <span className={clsx('font-mono', l.id === v.id ? 'text-text font-semibold' : 'text-muted')}>{l.id}</span>
                    <OriginBadge origin={l.origin} size="xs" />
                    <StatusBadge status={l.status} size="xs" />
                    <span className="text-faint text-[11px]">{fmtDate(l.created_at)}</span>
                  </div>
                </li>
              ))}
            </ol>
          </section>

          {v.notes && (
            <section>
              <SectionTitle className="mb-2">Notes</SectionTitle>
              <p className="text-[12.5px] text-muted leading-relaxed">{v.notes}</p>
            </section>
          )}

          <section>
            <SectionTitle className="mb-2">Last trades</SectionTitle>
            <div className="card">
              <DataTable columns={tradeColumns} rows={v.trades.slice(0, 12)} rowKey={(t) => t.id} dense empty="No closed trades yet" />
            </div>
          </section>
        </div>
      )}
    </Drawer>
  )
}

function Stat({ label, value, sub, cls }: { label: string; value: string; sub?: string; cls?: string }) {
  return (
    <div className="rounded-lg border border-border bg-panel-2 px-3 py-2 min-w-0">
      <div className="text-[10.5px] uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx('text-[15px] font-semibold mt-0.5 truncate', cls ?? 'text-text')}>{value}</div>
      {sub && <div className="text-[10.5px] text-faint truncate">{sub}</div>}
    </div>
  )
}
