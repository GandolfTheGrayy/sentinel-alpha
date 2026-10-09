import clsx from 'clsx'
import type { ReactNode } from 'react'

export interface ChipOption<V extends string> {
  value: V
  label: ReactNode
  count?: number
}

interface ChipsProps<V extends string> {
  options: ChipOption<V>[]
  value: V | null
  onChange: (v: V | null) => void
  /** Label for the "all" chip; omit to hide it. */
  allLabel?: string
  className?: string
  size?: 'xs' | 'sm'
}

/** Single-select filter chips with an optional "All" chip. */
export function Chips<V extends string>({ options, value, onChange, allLabel = 'All', className, size = 'sm' }: ChipsProps<V>) {
  const base = clsx(
    'inline-flex items-center gap-1.5 rounded-full border font-medium transition-colors whitespace-nowrap',
    size === 'sm' ? 'h-7 px-2.5 text-[11.5px]' : 'h-6 px-2 text-[11px]',
  )
  const idle = 'border-border text-muted hover:text-text hover:border-border-strong bg-transparent'
  const active = 'border-accent/50 bg-accent/12 text-accent'
  return (
    <div className={clsx('flex flex-wrap gap-1.5', className)} role="group">
      {allLabel && (
        <button type="button" className={clsx(base, value === null ? active : idle)} onClick={() => onChange(null)} aria-pressed={value === null}>
          {allLabel}
        </button>
      )}
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={clsx(base, value === o.value ? active : idle)}
          onClick={() => onChange(value === o.value ? null : o.value)}
          aria-pressed={value === o.value}
        >
          {o.label}
          {o.count !== undefined && <span className="text-[10px] opacity-70">{o.count}</span>}
        </button>
      ))}
    </div>
  )
}

interface SegmentedProps<V extends string> {
  options: { value: V; label: ReactNode }[]
  value: V
  onChange: (v: V) => void
  className?: string
  size?: 'xs' | 'sm'
}

/** Segmented control (range switchers, tabs). */
export function Segmented<V extends string>({ options, value, onChange, className, size = 'sm' }: SegmentedProps<V>) {
  return (
    <div className={clsx('inline-flex items-center rounded-lg border border-border bg-panel-2 p-0.5', className)} role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="tab"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={clsx(
            'rounded-md font-medium transition-colors whitespace-nowrap',
            size === 'sm' ? 'h-6 px-2.5 text-[11.5px]' : 'h-5 px-2 text-[11px]',
            value === o.value ? 'bg-panel text-text shadow-sm border border-border' : 'text-muted hover:text-text border border-transparent',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
