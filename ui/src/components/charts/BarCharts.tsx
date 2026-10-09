import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { BudgetHistoryWeek, FeatureBucket } from '../../api/types'
import { fmtFrac, fmtNum, fmtR, fmtUsd } from '../../lib/format'
import { useTheme } from '../../lib/theme'
import { fmtDate } from '../../lib/time'
import { RoundedBar, TooltipBox } from './ChartTooltip'

// ---------------------------------------------------------------------------
// Horizontal bars (importance / coefficients)
// ---------------------------------------------------------------------------

export interface HBarDatum {
  name: string
  value: number
}

interface HBarChartProps {
  data: HBarDatum[]
  /** Color bars by sign (diverging) instead of a single hue. */
  diverging?: boolean
  color?: string
  valueFormatter?: (v: number) => string
  height?: number
  labelWidth?: number
}

export function HBarChart({ data, diverging, color, valueFormatter = (v) => v.toFixed(3), height, labelWidth = 110 }: HBarChartProps) {
  const { colors } = useTheme()
  const fill = color ?? colors.info
  const h = height ?? Math.max(120, data.length * 22 + 24)
  const maxAbs = Math.max(0.0001, ...data.map((d) => Math.abs(d.value)))
  return (
    <div style={{ height: h }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 0 }} barCategoryGap={6}>
          <CartesianGrid horizontal={false} stroke={colors.grid} />
          <XAxis
            type="number"
            domain={diverging ? [-maxAbs * 1.1, maxAbs * 1.1] : [0, maxAbs * 1.1]}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => valueFormatter(v)}
            tickCount={5}
          />
          <YAxis type="category" dataKey="name" width={labelWidth} axisLine={false} tickLine={false} interval={0} />
          {diverging && <ReferenceLine x={0} stroke={colors.faint} />}
          <Tooltip
            cursor={{ fill: 'color-mix(in oklab, var(--text) 5%, transparent)' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null
              const d = payload[0]?.payload as HBarDatum | undefined
              if (!d) return null
              return <TooltipBox rows={[{ label: d.name, value: valueFormatter(d.value), color: diverging ? (d.value >= 0 ? colors.profit : colors.loss) : fill }]} />
            }}
          />
          <Bar dataKey="value" barSize={14} isAnimationActive={false} shape={<RoundedBar horizontal />}>
            {data.map((d) => (
              <Cell key={d.name} fill={diverging ? (d.value >= 0 ? colors.profit : colors.loss) : fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Bucketed expectancy (small multiple)
// ---------------------------------------------------------------------------

interface BucketChartProps {
  buckets: FeatureBucket[]
  height?: number
}

interface BucketDatum extends FeatureBucket {
  err: [number, number]
  opacity: number
}

export function BucketChart({ buckets, height = 150 }: BucketChartProps) {
  const { colors } = useTheme()
  const maxN = Math.max(1, ...buckets.map((b) => b.n))
  const data: BucketDatum[] = buckets.map((b) => ({
    ...b,
    err: [Math.max(0, b.expectancy_r - b.ci[0]), Math.max(0, b.ci[1] - b.expectancy_r)],
    opacity: b.n === 0 ? 0.15 : 0.45 + 0.55 * Math.sqrt(b.n / maxN),
  }))
  const lo = Math.min(0, ...data.map((d) => d.ci[0]))
  const hi = Math.max(0, ...data.map((d) => d.ci[1]))
  const pad = Math.max(0.05, (hi - lo) * 0.1)
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 0 }} barCategoryGap="28%">
          <CartesianGrid vertical={false} stroke={colors.grid} />
          <XAxis dataKey="label" axisLine={false} tickLine={false} interval={0} />
          <YAxis axisLine={false} tickLine={false} width={40} tickFormatter={(v: number) => `${v.toFixed(1)}R`} domain={[lo - pad, hi + pad]} tickCount={4} />
          <ReferenceLine y={0} stroke={colors.faint} />
          <Tooltip
            cursor={{ fill: 'color-mix(in oklab, var(--text) 5%, transparent)' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null
              const d = payload[0]?.payload as BucketDatum | undefined
              if (!d) return null
              return (
                <TooltipBox
                  title={`Bucket ${d.label}`}
                  rows={[
                    { label: 'Expectancy', value: fmtR(d.expectancy_r), color: d.expectancy_r >= 0 ? colors.profit : colors.loss },
                    { label: '95% CI', value: `${fmtR(d.ci[0])} to ${fmtR(d.ci[1])}`, muted: true },
                    { label: 'Win rate', value: fmtFrac(d.win_rate), muted: true },
                    { label: 'Trades', value: fmtNum(d.n), muted: true },
                  ]}
                />
              )
            }}
          />
          <Bar dataKey="expectancy_r" isAnimationActive={false} shape={<RoundedBar />} maxBarSize={28}>
            {data.map((d) => (
              <Cell key={d.label} fill={d.expectancy_r >= 0 ? colors.profit : colors.loss} fillOpacity={d.opacity} />
            ))}
            <ErrorBar dataKey="err" width={4} strokeWidth={1} stroke={colors.muted} direction="y" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Budget history
// ---------------------------------------------------------------------------

interface BudgetHistoryChartProps {
  history: BudgetHistoryWeek[]
  cap: number
  height?: number
}

export function BudgetHistoryChart({ history, cap, height = 200 }: BudgetHistoryChartProps) {
  const { colors } = useTheme()
  const data = history.map((h) => ({ ...h, label: fmtDate(h.week_start) }))
  const maxV = Math.max(cap, ...history.map((h) => h.spent_usd))
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap="30%">
          <CartesianGrid vertical={false} stroke={colors.grid} />
          <XAxis dataKey="label" axisLine={false} tickLine={false} interval="preserveStartEnd" minTickGap={24} />
          <YAxis axisLine={false} tickLine={false} width={44} tickFormatter={(v: number) => fmtUsd(v)} domain={[0, Math.ceil(maxV * 1.15)]} tickCount={4} />
          <ReferenceLine y={cap} stroke={colors.warn} strokeWidth={1} label={{ value: 'cap', position: 'insideTopRight', fill: colors.warn, fontSize: 10 }} />
          <Tooltip
            cursor={{ fill: 'color-mix(in oklab, var(--text) 5%, transparent)' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null
              const d = payload[0]?.payload as (BudgetHistoryWeek & { label: string }) | undefined
              if (!d) return null
              return (
                <TooltipBox
                  title={`Week of ${d.label}`}
                  rows={[
                    { label: 'Spent', value: fmtUsd(d.spent_usd, { decimals: 2 }), color: colors.claude },
                    { label: 'Runs', value: fmtNum(d.runs), muted: true },
                    { label: 'Cap', value: fmtUsd(cap, { decimals: 2 }), muted: true },
                  ]}
                />
              )
            }}
          />
          <Bar dataKey="spent_usd" fill={colors.claude} isAnimationActive={false} shape={<RoundedBar />} maxBarSize={32}>
            {data.map((d, i) => (
              <Cell key={d.week_start} fill={colors.claude} fillOpacity={i === data.length - 1 ? 1 : 0.6} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
