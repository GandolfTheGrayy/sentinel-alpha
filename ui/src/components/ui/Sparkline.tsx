import { useTheme } from '../../lib/theme'

interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  /** Color by the sign of the last value (default) or force a color. */
  color?: string
  className?: string
}

/** Tiny inline line chart. Draws a faint zero line when the series crosses zero. */
export function Sparkline({ data, width = 96, height = 26, color, className }: SparklineProps) {
  const { colors } = useTheme()
  if (!data || data.length < 2) {
    return <svg width={width} height={height} className={className} aria-hidden />
  }
  const min = Math.min(0, ...data)
  const max = Math.max(0, ...data)
  const span = max - min || 1
  const pad = 2
  const x = (i: number) => pad + (i / (data.length - 1)) * (width - pad * 2)
  const y = (v: number) => pad + (1 - (v - min) / span) * (height - pad * 2)
  const d = data.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')
  const last = data[data.length - 1] ?? 0
  const stroke = color ?? (last >= 0 ? colors.profit : colors.loss)
  const zeroY = y(0)
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className={className} aria-hidden>
      {min < 0 && max > 0 && <line x1={pad} x2={width - pad} y1={zeroY} y2={zeroY} stroke={colors.grid} strokeWidth={1} />}
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(data.length - 1)} cy={y(last)} r={2} fill={stroke} />
    </svg>
  )
}
