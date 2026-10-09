import clsx from 'clsx'
import type { ReactNode } from 'react'

export interface KVItem {
  k: ReactNode
  v: ReactNode
  /** Render the value in monospace. */
  mono?: boolean
  title?: string
}

interface KeyValueProps {
  items: KVItem[]
  cols?: 1 | 2 | 3
  className?: string
  dense?: boolean
}

/** Definition grid: label on the left (muted), value on the right. */
export function KeyValue({ items, cols = 2, className, dense }: KeyValueProps) {
  return (
    <dl
      className={clsx(
        'grid gap-x-6',
        dense ? 'gap-y-1' : 'gap-y-2',
        cols === 1 && 'grid-cols-1',
        cols === 2 && 'grid-cols-1 sm:grid-cols-2',
        cols === 3 && 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
        className,
      )}
    >
      {items.map((it, i) => (
        <div key={i} className="flex items-baseline justify-between gap-3 min-w-0 border-b border-border/60 pb-1" title={it.title}>
          <dt className="text-[11.5px] text-muted shrink-0 max-w-[55%] truncate">{it.k}</dt>
          <dd className={clsx('text-[12.5px] text-text text-right min-w-0 break-words', it.mono && 'font-mono text-[11.5px]')}>{it.v}</dd>
        </div>
      ))}
    </dl>
  )
}
