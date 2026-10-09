import clsx from 'clsx'
import { Check, X } from 'lucide-react'
import { useState } from 'react'
import type { Decision, Proposal } from '../api/types'
import { fmtNum, fmtPct, fmtR, fmtX, signClass } from '../lib/format'
import { fmtRelative } from '../lib/time'
import { Badge, proposalStatusTone, proposalTypeLabel } from './ui/Badge'
import { Button } from './ui/Button'

interface ProposalCardProps {
  proposal: Proposal
  onDecide?: (id: number, decision: Decision) => Promise<void> | void
  pending?: boolean
  compact?: boolean
}

export function ProposalCard({ proposal: p, onDecide, pending, compact }: ProposalCardProps) {
  const [showPayload, setShowPayload] = useState(false)
  const bt = p.backtest
  return (
    <article className="rounded-lg border border-border bg-panel-2 px-3.5 py-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-[11.5px] text-faint font-mono">#{p.id}</span>
        <Badge tone="claude">{proposalTypeLabel(p.type)}</Badge>
        {p.target && <span className="font-mono text-[12px] text-text">{p.target}</span>}
        <Badge tone={proposalStatusTone(p.status)} dot pulse={p.status === 'testing'}>
          {p.status}
        </Badge>
        <span className="ml-auto text-[11px] text-faint whitespace-nowrap">
          run #{p.run_id} · {fmtRelative(p.created_at)}
        </span>
      </div>
      <p className="text-[12.5px] text-text leading-relaxed mt-2">{p.rationale}</p>
      {p.decision_reason && (
        <p className={clsx('text-[12px] mt-1.5 leading-relaxed', p.status === 'accepted' ? 'text-profit' : p.status === 'rejected' ? 'text-muted' : 'text-warn')}>
          <span className="font-medium">Decision:</span> {p.decision_reason}
        </p>
      )}
      {p.created_variant_id && (
        <p className="text-[12px] text-muted mt-1">
          Spawned <span className="font-mono text-text">{p.created_variant_id}</span>
        </p>
      )}
      {bt && (
        <div className={clsx('grid gap-x-4 gap-y-1 mt-2.5 text-[11.5px]', compact ? 'grid-cols-3' : 'grid-cols-3 sm:grid-cols-6')}>
          <Mini label="n" value={fmtNum(bt.n)} />
          <Mini label="exp R" value={fmtR(bt.expectancy_r)} cls={signClass(bt.expectancy_r)} />
          <Mini label="PF" value={fmtX(bt.profit_factor)} />
          <Mini label="max DD" value={fmtPct(bt.max_dd_pct, { digits: 1 })} cls="text-loss" />
          <Mini label="control" value={fmtR(bt.control_expectancy_r)} cls={signClass(bt.control_expectancy_r)} />
          <Mini label="incumbent" value={bt.incumbent_expectancy_r === null ? '—' : fmtR(bt.incumbent_expectancy_r)} cls={signClass(bt.incumbent_expectancy_r)} />
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 mt-3">
        <button type="button" onClick={() => setShowPayload((s) => !s)} className="text-[11.5px] text-muted hover:text-text underline-offset-2 hover:underline">
          {showPayload ? 'Hide payload' : 'Show payload'}
        </button>
        {p.status === 'pending' && onDecide && (
          <div className="ml-auto flex items-center gap-1.5">
            <Button size="xs" variant="primary" icon={<Check className="h-3 w-3" />} loading={pending} onClick={() => void onDecide(p.id, 'accept')}>
              Accept
            </Button>
            <Button size="xs" variant="danger" icon={<X className="h-3 w-3" />} loading={pending} onClick={() => void onDecide(p.id, 'reject')}>
              Reject
            </Button>
          </div>
        )}
      </div>
      {showPayload && <pre className="mt-2 text-[11px] font-mono rounded-md border border-border bg-panel px-3 py-2 overflow-x-auto text-muted">{JSON.stringify(p.payload, null, 2)}</pre>}
    </article>
  )
}

function Mini({ label, value, cls }: { label: string; value: string; cls?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] uppercase tracking-wide text-faint">{label}</div>
      <div className={clsx('font-semibold truncate', cls ?? 'text-text')}>{value}</div>
    </div>
  )
}
