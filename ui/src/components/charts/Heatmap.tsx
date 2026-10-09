import clsx from 'clsx'
import { useRef, useState } from 'react'
import type { Heatmap as HeatmapData } from '../../api/types'
import { fmtNum, fmtR } from '../../lib/format'

interface HeatmapProps {
  data: HeatmapData
  rowLabel?: string
  colLabel?: string
}

interface Hover {
  r: number
  c: number
  x: number
  y: number
}

/** Hour x weekday heatmap. Diverging teal/rose around zero; counts on hover and as a title. */
export function Heatmap({ data, rowLabel = 'Weekday', colLabel = 'Hour (ET)' }: HeatmapProps) {
  const [hover, setHover] = useState<Hover | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const maxAbs = Math.max(0.05, ...data.values.flat().map((v) => (v === null ? 0 : Math.abs(v))))

  const cellStyle = (v: number | null) => {
    if (v === null) return { background: 'color-mix(in oklab, var(--muted) 6%, transparent)' }
    const strength = Math.min(1, Math.abs(v) / maxAbs)
    const pct = Math.round(10 + 72 * strength)
    const tone = v >= 0 ? 'var(--profit)' : 'var(--loss)'
    return { background: `color-mix(in oklab, ${tone} ${pct}%, transparent)` }
  }

  const hovered = hover ? { v: data.values[hover.r]?.[hover.c] ?? null, n: data.counts[hover.r]?.[hover.c] ?? 0 } : null

  return (
    <div className="relative" ref={containerRef}>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-[3px] w-full min-w-[420px]">
          <thead>
            <tr>
              <th className="text-left text-[10.5px] font-medium text-muted pb-1 pr-2 whitespace-nowrap">{rowLabel} \ {colLabel}</th>
              {data.cols.map((c) => (
                <th key={c} className="text-[10.5px] font-medium text-muted pb-1 text-center">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, r) => (
              <tr key={row}>
                <td className="text-[11px] text-muted pr-2 whitespace-nowrap">{row}</td>
                {data.cols.map((_, c) => {
                  const v = data.values[r]?.[c] ?? null
                  const n = data.counts[r]?.[c] ?? 0
                  return (
                    <td key={c} className="p-0">
                      <div
                        role="img"
                        aria-label={`${row} ${data.cols[c]}: ${v === null ? 'insufficient data' : fmtR(v)} (${n} trades)`}
                        title={`${row} ${data.cols[c]}h: ${v === null ? 'n/a' : fmtR(v)} · n=${n}`}
                        className={clsx(
                          'h-9 rounded-md flex items-center justify-center text-[11px] font-medium cursor-default transition-transform',
                          hover && hover.r === r && hover.c === c && 'ring-2 ring-text/40 scale-[1.04]',
                          v === null ? 'text-faint' : 'text-text',
                        )}
                        style={cellStyle(v)}
                        onMouseEnter={(e) => {
                          const rect = containerRef.current?.getBoundingClientRect()
                          const self = e.currentTarget.getBoundingClientRect()
                          setHover({ r, c, x: self.left - (rect?.left ?? 0) + self.width / 2, y: self.top - (rect?.top ?? 0) })
                        }}
                        onMouseLeave={() => setHover(null)}
                      >
                        {v === null ? '·' : fmtR(v, 2).replace('R', '')}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {hover && hovered && (
        <div
          className="pointer-events-none absolute z-10 -translate-x-1/2 -translate-y-full rounded-lg border border-border bg-tooltip px-2.5 py-1.5 text-[11px] shadow-xl whitespace-nowrap"
          style={{ left: hover.x, top: hover.y - 6 }}
        >
          <div className="text-muted">
            {data.rows[hover.r]} · {data.cols[hover.c]}:00 ET
          </div>
          <div className="flex gap-3 mt-0.5">
            <span className={clsx('font-semibold', hovered.v === null ? 'text-muted' : hovered.v >= 0 ? 'text-profit' : 'text-loss')}>
              {hovered.v === null ? 'n/a' : fmtR(hovered.v)}
            </span>
            <span className="text-muted">n={fmtNum(hovered.n)}</span>
          </div>
        </div>
      )}
      <div className="flex items-center gap-2 mt-2 text-[10.5px] text-muted">
        <span>{fmtR(-maxAbs)}</span>
        <div className="h-2 flex-1 max-w-[200px] rounded-full" style={{ background: 'linear-gradient(90deg, var(--loss), color-mix(in oklab, var(--muted) 15%, transparent) 50%, var(--profit))' }} />
        <span>{fmtR(maxAbs)}</span>
        <span className="ml-2">· cells with fewer than 3 trades are blank</span>
      </div>
    </div>
  )
}
