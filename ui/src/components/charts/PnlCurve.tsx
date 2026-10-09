import { useMemo } from 'react'
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { PnlCurvePoint } from '../../api/types'
import { fmtUsd } from '../../lib/format'
import { useTheme } from '../../lib/theme'
import { fmtDateTime, makeTickFormatter } from '../../lib/time'
import { EmptyState } from '../ui/EmptyState'
import { TooltipBox } from './ChartTooltip'

interface PnlCurveProps {
  data: PnlCurvePoint[]
  height?: number
  id?: string
}

/** Cumulative P&L area with a zero baseline; colored by the final sign. */
export function PnlCurve({ data, height = 180, id = 'pnl' }: PnlCurveProps) {
  const { colors } = useTheme()
  const last = data[data.length - 1]?.pnl_cum ?? 0
  const color = last >= 0 ? colors.profit : colors.loss
  const tickFormatter = useMemo(() => {
    if (data.length < 2) return (s: string) => s
    const span = new Date(data[data.length - 1]!.ts).getTime() - new Date(data[0]!.ts).getTime()
    return makeTickFormatter(span)
  }, [data])

  if (data.length < 2) return <EmptyState title="No closed trades yet" compact />

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={`${id}Gradient`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.25} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke={colors.grid} />
          <XAxis dataKey="ts" tickFormatter={tickFormatter} axisLine={false} tickLine={false} minTickGap={40} interval="preserveStartEnd" />
          <YAxis axisLine={false} tickLine={false} width={52} tickFormatter={(v: number) => fmtUsd(v, { compact: true })} tickCount={4} />
          <ReferenceLine y={0} stroke={colors.faint} strokeWidth={1} />
          <Tooltip
            cursor={{ stroke: colors.faint, strokeWidth: 1 }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null
              const p = payload[0]?.payload as PnlCurvePoint | undefined
              if (!p) return null
              return <TooltipBox title={fmtDateTime(p.ts)} rows={[{ label: 'Cumulative P&L', value: fmtUsd(p.pnl_cum, { sign: true, decimals: 2 }), color }]} />
            }}
          />
          <Area
            type="monotone"
            dataKey="pnl_cum"
            stroke={color}
            strokeWidth={2}
            fill={`url(#${id}Gradient)`}
            dot={false}
            activeDot={{ r: 4, stroke: colors.panel, strokeWidth: 2, fill: color }}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
