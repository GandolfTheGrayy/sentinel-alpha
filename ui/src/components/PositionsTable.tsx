import clsx from 'clsx'
import { useMemo } from 'react'
import type { Position } from '../api/types'
import { fmtPrice, fmtQty, fmtR, fmtUsd, signClass } from '../lib/format'
import { fmtRelative } from '../lib/time'
import { Badge } from './ui/Badge'
import { DataTable, type Column } from './ui/DataTable'

interface PositionsTableProps {
  positions: Position[] | null
  loading?: boolean
  compact?: boolean
}

export function PositionsTable({ positions, loading, compact }: PositionsTableProps) {
  const columns = useMemo<Column<Position>[]>(
    () => [
      {
        key: 'symbol',
        header: 'Symbol',
        render: (p) => (
          <span className="inline-flex items-center gap-2">
            <span className="font-semibold text-text">{p.symbol}</span>
            <Badge tone={p.side === 'long' ? 'profit' : 'loss'} size="xs">
              {p.side}
            </Badge>
          </span>
        ),
        sortValue: (p) => p.symbol,
      },
      { key: 'variant', header: 'Variant', render: (p) => <span className="font-mono text-[11.5px] text-muted">{p.variant_id}</span>, sortValue: (p) => p.variant_id },
      { key: 'qty', header: 'Qty', align: 'right', render: (p) => fmtQty(p.qty), sortValue: (p) => p.qty, hideBelow: 'sm' },
      { key: 'entry', header: 'Entry', align: 'right', render: (p) => fmtPrice(p.entry_price), sortValue: (p) => p.entry_price, hideBelow: 'md' },
      { key: 'last', header: 'Last', align: 'right', render: (p) => fmtPrice(p.current_price), sortValue: (p) => p.current_price },
      {
        key: 'upnl',
        header: 'Unrealized',
        align: 'right',
        render: (p) => <span className={clsx('font-semibold', signClass(p.unrealized_pnl))}>{fmtUsd(p.unrealized_pnl, { sign: true, decimals: 2 })}</span>,
        sortValue: (p) => p.unrealized_pnl,
      },
      { key: 'r', header: 'R', align: 'right', render: (p) => <span className={signClass(p.unrealized_r)}>{fmtR(p.unrealized_r)}</span>, sortValue: (p) => p.unrealized_r },
      {
        key: 'levels',
        header: 'Stop / Target',
        align: 'right',
        render: (p) => (
          <span className="text-muted">
            <span className="text-loss">{fmtPrice(p.stop)}</span> / <span className="text-profit">{fmtPrice(p.target)}</span>
          </span>
        ),
        hideBelow: 'lg',
      },
      { key: 'age', header: 'Age', align: 'right', render: (p) => <span className="text-muted">{fmtRelative(p.entry_ts).replace(' ago', '')}</span>, sortValue: (p) => p.entry_ts, hideBelow: 'md' },
      ...(compact
        ? []
        : [
            {
              key: 'reason',
              header: 'Reason',
              render: (p: Position) => <span className="text-muted truncate block max-w-[260px]">{p.reason}</span>,
              hideBelow: 'xl' as const,
            },
          ]),
    ],
    [compact],
  )

  return (
    <DataTable
      columns={columns}
      rows={positions}
      rowKey={(p) => p.lot_id}
      loading={loading}
      defaultSort={{ key: 'upnl', dir: 'desc' }}
      empty="No open positions"
      emptyDescription="The engine is flat. Positions appear here when a variant's signal is filled."
      dense
    />
  )
}
