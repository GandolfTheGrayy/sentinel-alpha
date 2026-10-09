import clsx from 'clsx'
import { useMemo, useState } from 'react'
import { api } from '../api/client'
import { usePoll } from '../api/hooks'
import { EXIT_REASONS, type ExitReason, type Trade } from '../api/types'
import { TradeDrawer } from '../components/TradeDrawer'
import { Badge, exitReasonTone } from '../components/ui/Badge'
import { Card } from '../components/ui/Card'
import { Chips } from '../components/ui/Chips'
import { DataTable, type Column } from '../components/ui/DataTable'
import { ErrorState } from '../components/ui/EmptyState'
import { fmtDuration, fmtFrac, fmtNum, fmtPrice, fmtQty, fmtR, fmtUsd, signClass } from '../lib/format'
import { fmtDateTime } from '../lib/time'
import { useApp } from '../state/AppContext'

export function TradesPage() {
  const { changeTick } = useApp()
  const [variant, setVariant] = useState('')
  const [symbol, setSymbol] = useState('')
  const [exit, setExit] = useState<ExitReason | null>(null)
  const [selected, setSelected] = useState<Trade | null>(null)

  const trades = usePoll(() => api.getTrades({ limit: 500, variant: variant || undefined, symbol: symbol || undefined }), 30_000, [variant, symbol, changeTick])
  const variants = usePoll(() => api.getVariants(), 120_000)

  const symbols = useMemo(() => {
    const s = new Set<string>()
    for (const t of trades.data ?? []) s.add(t.symbol)
    if (symbol) s.add(symbol)
    return [...s].sort()
  }, [trades.data, symbol])

  const rows = useMemo(() => {
    const r = trades.data ?? []
    return exit ? r.filter((t) => t.exit_reason === exit) : r
  }, [trades.data, exit])

  const exitCounts = useMemo(() => {
    const c = new Map<ExitReason, number>()
    for (const t of trades.data ?? []) c.set(t.exit_reason, (c.get(t.exit_reason) ?? 0) + 1)
    return c
  }, [trades.data])

  const summary = useMemo(() => {
    const n = rows.length
    if (!n) return null
    const wins = rows.filter((t) => t.pnl > 0).length
    const pnl = rows.reduce((a, t) => a + t.pnl, 0)
    const expR = rows.reduce((a, t) => a + t.pnl_r, 0) / n
    return { n, winRate: wins / n, pnl, expR }
  }, [rows])

  const columns = useMemo<Column<Trade>[]>(
    () => [
      { key: 'exit_ts', header: 'Closed', render: (t) => <span className="text-muted whitespace-nowrap">{fmtDateTime(t.exit_ts)}</span>, sortValue: (t) => t.exit_ts },
      { key: 'variant', header: 'Variant', render: (t) => <span className="font-mono text-[11.5px]">{t.variant_id}</span>, sortValue: (t) => t.variant_id },
      {
        key: 'symbol',
        header: 'Symbol',
        render: (t) => (
          <span className="inline-flex items-center gap-1.5">
            <span className="font-semibold">{t.symbol}</span>
            <Badge tone={t.side === 'long' ? 'profit' : 'loss'} size="xs">
              {t.side}
            </Badge>
          </span>
        ),
        sortValue: (t) => t.symbol,
      },
      { key: 'qty', header: 'Qty', align: 'right', render: (t) => fmtQty(t.qty), sortValue: (t) => t.qty, hideBelow: 'xl' },
      {
        key: 'prices',
        header: 'Entry → Exit',
        align: 'right',
        render: (t) => (
          <span className="text-muted whitespace-nowrap">
            {fmtPrice(t.entry_price)} <span className="text-faint">→</span> <span className="text-text">{fmtPrice(t.exit_price)}</span>
          </span>
        ),
        hideBelow: 'xl',
      },
      {
        key: 'pnl',
        header: 'P&L',
        align: 'right',
        render: (t) => <span className={clsx('font-semibold', signClass(t.pnl))}>{fmtUsd(t.pnl, { sign: true, decimals: 2 })}</span>,
        sortValue: (t) => t.pnl,
      },
      { key: 'r', header: 'R', align: 'right', render: (t) => <span className={clsx('font-semibold', signClass(t.pnl_r))}>{fmtR(t.pnl_r)}</span>, sortValue: (t) => t.pnl_r },
      { key: 'hold', header: 'Hold', align: 'right', render: (t) => <span className="text-muted">{fmtDuration(t.hold_minutes)}</span>, sortValue: (t) => t.hold_minutes, hideBelow: 'lg' },
      { key: 'exit', header: 'Exit', render: (t) => <Badge tone={exitReasonTone(t.exit_reason)} size="xs">{t.exit_reason}</Badge>, sortValue: (t) => t.exit_reason },
    ],
    [],
  )

  return (
    <div className="space-y-4">
      <Card
        title="Trade tape"
        subtitle={
          summary
            ? `${fmtNum(summary.n)} trades · win rate ${fmtFrac(summary.winRate)} · expectancy ${fmtR(summary.expR)} · ${fmtUsd(summary.pnl, { sign: true })}`
            : trades.loading
              ? 'Loading trades…'
              : 'No trades match the current filters'
        }
        flush
      >
        <div className="px-4 pb-3 flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <select className="input min-w-[180px]" value={variant} onChange={(e) => setVariant(e.target.value)} aria-label="Filter by variant">
              <option value="">All variants</option>
              {(variants.data ?? []).map((v) => (
                <option key={v.id} value={v.id}>
                  {v.id} · {v.status}
                </option>
              ))}
            </select>
            <select className="input min-w-[140px]" value={symbol} onChange={(e) => setSymbol(e.target.value)} aria-label="Filter by symbol">
              <option value="">All symbols</option>
              {symbols.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            {(variant || symbol || exit) && (
              <button
                type="button"
                className="text-[11.5px] text-muted hover:text-text underline-offset-2 hover:underline"
                onClick={() => {
                  setVariant('')
                  setSymbol('')
                  setExit(null)
                }}
              >
                Clear filters
              </button>
            )}
          </div>
          <Chips
            options={EXIT_REASONS.map((r) => ({ value: r, label: r, count: exitCounts.get(r) ?? 0 }))}
            value={exit}
            onChange={setExit}
            allLabel="Any exit"
            size="xs"
          />
        </div>
        {trades.error && !trades.data ? (
          <ErrorState error={trades.error} onRetry={() => void trades.refresh()} />
        ) : (
          <DataTable
            columns={columns}
            rows={rows}
            rowKey={(t) => t.id}
            loading={trades.loading}
            onRowClick={setSelected}
            activeKey={selected?.id ?? null}
            defaultSort={{ key: 'exit_ts', dir: 'desc' }}
            empty="No trades yet"
            emptyDescription="Closed trades appear here with their full feature snapshot."
            maxHeight="calc(100vh - 290px)"
            skeletonRows={12}
            dense
          />
        )}
      </Card>
      <TradeDrawer trade={selected} onClose={() => setSelected(null)} />
    </div>
  )
}
