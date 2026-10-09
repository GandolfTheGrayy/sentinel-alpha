import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { EquityPoint } from '../../api/types'
import { fmtUsd } from '../../lib/format'
import { useTheme } from '../../lib/theme'
import { fmtDateTime, makeTickFormatter } from '../../lib/time'
import { SkeletonChart } from '../ui/Skeleton'
import { EmptyState } from '../ui/EmptyState'
import { TooltipBox } from './ChartTooltip'

interface EquityChartProps {
  data: EquityPoint[] | null
  loading?: boolean
  height?: number
  showBenchmark?: boolean
}

export function EquityChart({ data, loading, height = 260, showBenchmark = true }: EquityChartProps) {
  const { colors } = useTheme()

  const { domain, tickFormatter } = useMemo(() => {
    if (!data || data.length === 0) return { domain: ['auto', 'auto'] as [string, string], tickFormatter: (s: string) => s }
    let lo = Infinity
    let hi = -Infinity
    for (const p of data) {
      lo = Math.min(lo, p.equity, showBenchmark ? p.benchmark : p.equity)
      hi = Math.max(hi, p.equity, showBenchmark ? p.benchmark : p.equity)
    }
    const pad = Math.max((hi - lo) * 0.08, 50)
    const span = new Date(data[data.length - 1]!.ts).getTime() - new Date(data[0]!.ts).getTime()
    return { domain: [Math.floor(lo - pad), Math.ceil(hi + pad)] as [number, number], tickFormatter: makeTickFormatter(span) }
  }, [data, showBenchmark])

  if (loading && !data) return <SkeletonChart height={height} />
  if (!data || data.length < 2) return <EmptyState title="No equity history yet" description="Points appear as the engine records equity snapshots." compact />

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colors.accent} stopOpacity={0.28} />
              <stop offset="100%" stopColor={colors.accent} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke={colors.grid} />
          <XAxis dataKey="ts" tickFormatter={tickFormatter} axisLine={false} tickLine={false} minTickGap={48} interval="preserveStartEnd" />
          <YAxis
            domain={domain}
            axisLine={false}
            tickLine={false}
            width={58}
            tickFormatter={(v: number) => fmtUsd(v, { compact: true })}
            tickCount={5}
          />
          <Tooltip
            cursor={{ stroke: colors.faint, strokeWidth: 1 }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null
              const p = payload[0]?.payload as EquityPoint | undefined
              if (!p) return null
              const delta = p.equity - p.benchmark
              return (
                <TooltipBox
                  title={fmtDateTime(p.ts)}
                  rows={[
                    { label: 'Equity', value: fmtUsd(p.equity, { decimals: 2 }), color: colors.accent },
                    ...(showBenchmark
                      ? [
                          { label: 'Benchmark', value: fmtUsd(p.benchmark, { decimals: 2 }), color: colors.faint },
                          { label: 'vs benchmark', value: fmtUsd(delta, { sign: true }), muted: true },
                        ]
                      : []),
                    { label: 'Cash', value: fmtUsd(p.cash), muted: true },
                  ]}
                />
              )
            }}
          />
          {showBenchmark && <Line type="monotone" dataKey="benchmark" stroke={colors.faint} strokeWidth={1.5} dot={false} isAnimationActive={false} />}
          <Area
            type="monotone"
            dataKey="equity"
            stroke={colors.accent}
            strokeWidth={2}
            fill="url(#equityGradient)"
            dot={false}
            activeDot={{ r: 4, stroke: colors.panel, strokeWidth: 2, fill: colors.accent }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export function EquityLegend({ showBenchmark = true }: { showBenchmark?: boolean }) {
  const { colors } = useTheme()
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-muted">
      <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
        <span className="h-0.5 w-4 rounded-full" style={{ background: colors.accent }} />
        Total equity
      </span>
      {showBenchmark && (
        <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
          <span className="h-0.5 w-4 rounded-full" style={{ background: colors.faint }} />
          Buy & hold benchmark
        </span>
      )}
    </div>
  )
}
