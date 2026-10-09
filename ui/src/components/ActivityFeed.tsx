import clsx from 'clsx'
import { Activity } from 'lucide-react'
import { useMemo } from 'react'
import type { EventKind, EventRow } from '../api/types'
import { fmtTime } from '../lib/time'
import { Badge, eventKindTone } from './ui/Badge'
import { EmptyState } from './ui/EmptyState'
import { SkeletonRows } from './ui/Skeleton'

interface ActivityFeedProps {
  fetched: EventRow[] | null
  live: EventRow[]
  loading?: boolean
  limit?: number
  kinds?: EventKind[] | null
  maxHeight?: number | string
}

/** Merges fetched + streamed events (dedupe by id), newest first. */
export function ActivityFeed({ fetched, live, loading, limit = 80, kinds, maxHeight = 460 }: ActivityFeedProps) {
  const rows = useMemo(() => {
    const map = new Map<number, EventRow>()
    for (const e of fetched ?? []) map.set(e.id, e)
    for (const e of live) map.set(e.id, e)
    let all = [...map.values()].sort((a, b) => b.ts.localeCompare(a.ts) || b.id - a.id)
    if (kinds && kinds.length) all = all.filter((e) => kinds.includes(e.kind))
    return all.slice(0, limit)
  }, [fetched, live, limit, kinds])

  if (loading && rows.length === 0) return <SkeletonRows rows={8} cols={3} />
  if (rows.length === 0) return <EmptyState icon={Activity} title="No activity yet" description="Fills, signals, exits and system events show up here as they happen." compact />

  return (
    <ul className="overflow-y-auto divide-y divide-border/60" style={{ maxHeight }}>
      {rows.map((e) => (
        <li
          key={e.id}
          className={clsx(
            'flex items-start gap-2.5 px-4 py-2 text-[12px] border-l-2',
            e.level === 'error' ? 'border-l-loss bg-loss/5' : e.level === 'warn' ? 'border-l-warn' : 'border-l-transparent',
          )}
        >
          <span className="text-faint font-mono text-[10.5px] pt-0.5 shrink-0 w-[58px]">{fmtTime(e.ts, true)}</span>
          <Badge tone={eventKindTone(e.kind)} size="xs" className="shrink-0 w-[68px] justify-center">
            {e.kind}
          </Badge>
          <span className="text-text leading-[18px] min-w-0 break-words">{e.message}</span>
        </li>
      ))}
    </ul>
  )
}
