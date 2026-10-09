import clsx from 'clsx'
import type { ReactNode } from 'react'
import { clamp } from '../../lib/format'

interface ArcGaugeProps {
  /** 0..1 */
  fraction: number
  size?: number
  stroke?: number
  /** CSS color for the filled arc. */
  color: string
  trackColor?: string
  children?: ReactNode
  className?: string
  label?: string
}

/** Semi-circular gauge. Children render centered beneath the arc. */
export function ArcGauge({ fraction, size = 160, stroke = 12, color, trackColor, children, className, label }: ArcGaugeProps) {
  const f = clamp(Number.isFinite(fraction) ? fraction : 0, 0, 1)
  const r = (size - stroke) / 2
  const cx = size / 2
  const cy = size / 2
  const circ = Math.PI * r // half circle length
  const dash = circ * f
  const path = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`
  return (
    <div className={clsx('relative flex flex-col items-center', className)} style={{ width: size }} role="img" aria-label={label}>
      <svg width={size} height={size / 2 + stroke / 2} viewBox={`0 0 ${size} ${size / 2 + stroke / 2}`} aria-hidden>
        <path d={path} fill="none" stroke={trackColor ?? 'color-mix(in oklab, currentColor 12%, transparent)'} strokeWidth={stroke} strokeLinecap="round" className="text-muted" />
        <path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ + 10}`}
          style={{ transition: 'stroke-dasharray 500ms ease' }}
        />
      </svg>
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-center justify-end text-center" style={{ height: size / 2 - stroke }}>
        {children}
      </div>
    </div>
  )
}

interface MeterProps {
  fraction: number
  color?: string
  className?: string
  height?: number
  /** Optional marker positions (0..1). */
  markers?: number[]
}

/** Horizontal meter. The track is a lighter step of the same color. */
export function Meter({ fraction, color = 'var(--accent)', className, height = 6, markers }: MeterProps) {
  const f = clamp(Number.isFinite(fraction) ? fraction : 0, 0, 1)
  return (
    <div className={clsx('relative w-full rounded-full overflow-hidden', className)} style={{ height, background: `color-mix(in oklab, ${color} 18%, transparent)` }}>
      <div className="h-full rounded-full" style={{ width: `${f * 100}%`, background: color, transition: 'width 400ms ease' }} />
      {markers?.map((m, i) => (
        <div key={i} className="absolute top-0 bottom-0 w-px bg-panel" style={{ left: `${clamp(m, 0, 1) * 100}%` }} />
      ))}
    </div>
  )
}

/** Tiny ring used in the top bar. */
export function MiniRing({ fraction, color, size = 18, stroke = 3, className }: { fraction: number; color: string; size?: number; stroke?: number; className?: string }) {
  const f = clamp(Number.isFinite(fraction) ? fraction : 0, 0, 1)
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={className} aria-hidden>
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={`color-mix(in oklab, ${color} 20%, transparent)`} strokeWidth={stroke} />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${c * f} ${c}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    </svg>
  )
}
