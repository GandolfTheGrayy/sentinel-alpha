import clsx from 'clsx'
import type { ReactNode } from 'react'

export interface TooltipRow {
  label: ReactNode
  value: ReactNode
  color?: string
  muted?: boolean
}

/** Shared tooltip chrome for all charts. Values lead, labels follow. */
export function TooltipBox({ title, rows, className }: { title?: ReactNode; rows: TooltipRow[]; className?: string }) {
  return (
    <div className={clsx('rounded-lg border border-border bg-tooltip px-3 py-2 shadow-xl text-[11.5px] min-w-[150px]', className)}>
      {title && <div className="text-muted mb-1.5">{title}</div>}
      <div className="space-y-1">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted min-w-0">
              {r.color && <span className="inline-block h-0.5 w-3 rounded-full shrink-0" style={{ background: r.color }} />}
              <span className="truncate">{r.label}</span>
            </span>
            <span className={clsx('font-semibold whitespace-nowrap', r.muted ? 'text-muted' : 'text-text')}>{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface RoundedBarProps {
  x?: number
  y?: number
  width?: number
  height?: number
  fill?: string
  fillOpacity?: number
  value?: number | [number, number]
  radius?: number
  /** Bars grow horizontally (recharts layout="vertical"). */
  horizontal?: boolean
}

/**
 * Bar shape with a 4px rounded data-end and a square baseline end. Works for positive and
 * negative values in both orientations. Pass as `shape={<RoundedBar />}`.
 */
export function RoundedBar({ x = 0, y = 0, width = 0, height = 0, fill = 'currentColor', fillOpacity, value, radius = 4, horizontal }: RoundedBarProps) {
  let left = x
  let top = y
  let w = width
  let h = height
  if (w < 0) {
    left = x + w
    w = -w
  }
  if (h < 0) {
    top = y + h
    h = -h
  }
  if (w <= 0 || h <= 0) return <g />
  const cur = Array.isArray(value) ? value[1] - value[0] : (value ?? 0)
  const negative = cur < 0
  let d: string
  if (horizontal) {
    const r = Math.min(radius, h / 2, w)
    if (!negative) {
      // rounded right end
      d = `M${left},${top} H${left + w - r} A${r},${r} 0 0 1 ${left + w},${top + r} V${top + h - r} A${r},${r} 0 0 1 ${left + w - r},${top + h} H${left} Z`
    } else {
      // rounded left end
      d = `M${left + w},${top} H${left + r} A${r},${r} 0 0 0 ${left},${top + r} V${top + h - r} A${r},${r} 0 0 0 ${left + r},${top + h} H${left + w} Z`
    }
  } else {
    const r = Math.min(radius, w / 2, h)
    if (!negative) {
      // rounded top
      d = `M${left},${top + h} V${top + r} A${r},${r} 0 0 1 ${left + r},${top} H${left + w - r} A${r},${r} 0 0 1 ${left + w},${top + r} V${top + h} Z`
    } else {
      // rounded bottom
      d = `M${left},${top} V${top + h - r} A${r},${r} 0 0 0 ${left + r},${top + h} H${left + w - r} A${r},${r} 0 0 0 ${left + w},${top + h - r} V${top} Z`
    }
  }
  return <path d={d} fill={fill} fillOpacity={fillOpacity} />
}
