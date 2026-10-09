import clsx from 'clsx'
import { Ban, OctagonX, Pause, Play, Scale } from 'lucide-react'
import { useEffect, useState, type FormEvent } from 'react'
import { api } from '../api/client'
import { useAction, usePoll } from '../api/hooks'
import type { EngineAction } from '../api/types'
import { BudgetHistoryChart } from '../components/charts/BarCharts'
import { ConfigTree } from '../components/ConfigTree'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, SectionTitle } from '../components/ui/Card'
import { useConfirm } from '../components/ui/Confirm'
import { EmptyState, ErrorState } from '../components/ui/EmptyState'
import { ArcGauge } from '../components/ui/Gauge'
import { KeyValue } from '../components/ui/KeyValue'
import { Skeleton, SkeletonChart, SkeletonText } from '../components/ui/Skeleton'
import { useToast } from '../components/ui/Toast'
import { fmtFrac, fmtNum, fmtUptime, fmtUsd } from '../lib/format'
import { useTheme } from '../lib/theme'
import { fmtDate, fmtDateTime, fmtRelative } from '../lib/time'
import { useApp } from '../state/AppContext'

export function SettingsPage() {
  const { changeTick } = useApp()
  const budget = usePoll(() => api.getBudget(), 60_000, [changeTick])
  const config = usePoll(() => api.getConfig(), null)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <BudgetCard budget={budget} />
        <div className="space-y-4">
          <CalibrationCard budget={budget} />
          <EngineCard />
        </div>
      </div>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <Card title="Configuration" subtitle="Sanitized: secrets are redacted by the API">
          {config.error && !config.data ? (
            <ErrorState error={config.error} onRetry={() => void config.refresh()} />
          ) : !config.data ? (
            <SkeletonText lines={8} />
          ) : (
            <ConfigTree config={config.data} />
          )}
        </Card>
        <UniverseCard />
      </div>
    </div>
  )
}

type BudgetPoll = ReturnType<typeof usePoll<Awaited<ReturnType<typeof api.getBudget>>>>

function BudgetCard({ budget }: { budget: BudgetPoll }) {
  const { colors } = useTheme()
  const b = budget.data
  const frac = b && b.weekly_cap_usd > 0 ? b.spent_usd / b.weekly_cap_usd : 0
  const color = frac >= 0.95 ? colors.loss : frac >= 0.8 ? colors.warn : colors.claude
  return (
    <Card title="Claude budget" subtitle="Weekly cap = plan allowance estimate x weekly share; rolling week">
      {budget.error && !b ? (
        <ErrorState error={budget.error} onRetry={() => void budget.refresh()} />
      ) : !b ? (
        <div className="space-y-4">
          <Skeleton className="h-24 w-full" />
          <SkeletonText lines={5} />
          <SkeletonChart height={200} />
        </div>
      ) : (
        <div className="space-y-5">
          <div className="flex flex-col sm:flex-row items-center gap-5">
            <ArcGauge fraction={frac} color={color} size={170} label={`Spent ${fmtUsd(b.spent_usd, { decimals: 2 })} of ${fmtUsd(b.weekly_cap_usd, { decimals: 2 })}`}>
              <div className="text-[20px] font-semibold text-text leading-6">{fmtUsd(b.spent_usd, { decimals: 2 })}</div>
              <div className="text-[11px] text-muted">of {fmtUsd(b.weekly_cap_usd, { decimals: 2 })} cap</div>
            </ArcGauge>
            <KeyValue
              className="flex-1 w-full"
              cols={1}
              dense
              items={[
                { k: 'Plan', v: <Badge tone="claude">{b.plan}</Badge> },
                { k: 'Weekly share of plan', v: fmtFrac(b.weekly_share, { digits: 0 }) },
                { k: 'Allowance estimate', v: `${fmtUsd(b.weekly_allowance_usd_est, { decimals: 2 })} / week` },
                { k: 'Weekly cap', v: fmtUsd(b.weekly_cap_usd, { decimals: 2 }) },
                { k: 'Remaining', v: <span className={b.remaining_usd <= 0 ? 'text-loss font-semibold' : 'text-profit font-semibold'}>{fmtUsd(b.remaining_usd, { decimals: 2 })}</span> },
                { k: 'Week started', v: `${fmtDate(b.week_start, true)} · ${b.runs_this_week} runs` },
              ]}
            />
          </div>
          <KeyValue
            cols={2}
            dense
            items={[
              { k: 'Analyst model', v: b.models.analyst, mono: true },
              { k: 'Strategist model', v: b.models.strategist, mono: true },
              { k: 'Analyst schedule', v: b.schedule.analyst },
              { k: 'Strategist schedule', v: b.schedule.strategist },
            ]}
          />
          <div>
            <SectionTitle className="mb-2">Spend by week</SectionTitle>
            {b.history.length === 0 ? <EmptyState title="No history yet" compact /> : <BudgetHistoryChart history={b.history} cap={b.weekly_cap_usd} />}
          </div>
        </div>
      )}
    </Card>
  )
}

function CalibrationCard({ budget }: { budget: BudgetPoll }) {
  const toast = useToast()
  const [value, setValue] = useState('')
  const calibrate = useAction((pct: number) => api.calibrateBudget(pct))
  const b = budget.data

  useEffect(() => {
    if (b?.calibration.observed_pct !== null && b?.calibration.observed_pct !== undefined && value === '') {
      setValue(String(b.calibration.observed_pct))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [b?.calibration.observed_pct])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const pct = Number(value)
    if (!Number.isFinite(pct) || pct <= 0 || pct > 100) {
      toast({ tone: 'warn', title: 'Enter a percentage between 0 and 100' })
      return
    }
    const updated = await calibrate.run(pct)
    if (updated !== undefined) {
      // The contract does not pin the response body; use it when it looks like a budget, and refetch regardless.
      const looksLikeBudget = typeof updated === 'object' && updated !== null && 'weekly_cap_usd' in updated
      if (looksLikeBudget) budget.setData(() => updated)
      toast({
        tone: 'success',
        title: 'Budget calibrated',
        message: looksLikeBudget
          ? `Allowance estimate ${fmtUsd(updated.weekly_allowance_usd_est, { decimals: 2 })}/week · cap ${fmtUsd(updated.weekly_cap_usd, { decimals: 2 })}`
          : `Observed ${pct}% recorded; the allowance estimate has been rescaled.`,
      })
      void budget.refresh()
    } else {
      toast({ tone: 'error', title: 'Calibration failed', message: calibrate.error?.message ?? 'The API rejected the calibration.' })
    }
  }

  return (
    <Card title="Calibrate allowance" subtitle="After a week, enter the percentage shown by Claude Code /usage that is attributable to Sentinel">
      <form onSubmit={(e) => void submit(e)} className="flex flex-col sm:flex-row sm:items-end gap-3">
        <label className="flex-1 min-w-0">
          <span className="block text-[11.5px] text-muted mb-1">Observed weekly usage (%)</span>
          <div className="relative">
            <input
              type="number"
              inputMode="decimal"
              min={0.1}
              max={100}
              step={0.1}
              placeholder="e.g. 7.5"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              className="input w-full pr-8"
              aria-label="Observed weekly usage percent"
            />
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[12px] text-faint">%</span>
          </div>
        </label>
        <Button type="submit" variant="primary" size="md" icon={<Scale className="h-3.5 w-3.5" />} loading={calibrate.pending} disabled={!value}>
          Rescale estimate
        </Button>
      </form>
      {b && (
        <p className="text-[11.5px] text-muted mt-3">
          {b.calibration.observed_pct !== null ? (
            <>
              Last calibration: <span className="text-text">{b.calibration.observed_pct}%</span> observed.{' '}
            </>
          ) : null}
          {b.calibration.note}
        </p>
      )}
    </Card>
  )
}

const ACTIONS: { action: EngineAction; label: string; icon: typeof Pause; variant: 'warn' | 'primary' | 'danger'; title: string; message: string }[] = [
  {
    action: 'pause',
    label: 'Pause entries',
    icon: Pause,
    variant: 'warn',
    title: 'Pause the engine?',
    message: 'No new entries will be taken. Open positions keep their stops, targets and time exits.',
  },
  {
    action: 'resume',
    label: 'Resume',
    icon: Play,
    variant: 'primary',
    title: 'Resume the engine?',
    message: 'Entries resume on the next tick for every active variant.',
  },
  {
    action: 'halt',
    label: 'Halt',
    icon: OctagonX,
    variant: 'danger',
    title: 'Halt the engine?',
    message: 'Stops the trading loop entirely until resumed. Use when something looks wrong with data or the broker.',
  },
  {
    action: 'flatten',
    label: 'Flatten all',
    icon: Ban,
    variant: 'danger',
    title: 'Flatten all positions?',
    message: 'Sends market orders to close every open lot across all variants. This cannot be undone.',
  },
]

function EngineCard() {
  const { status, refreshStatus } = useApp()
  const confirm = useConfirm()
  const toast = useToast()
  const act = useAction((a: EngineAction) => api.engineAction(a))
  const e = status?.engine
  const state = !e ? null : e.halted ? 'halted' : e.paused ? 'paused' : e.running ? 'running' : 'stopped'

  const run = async (cfg: (typeof ACTIONS)[number]) => {
    const ok = await confirm({ title: cfg.title, message: cfg.message, confirmLabel: cfg.label, tone: cfg.variant === 'primary' ? 'primary' : cfg.variant === 'warn' ? 'warn' : 'danger' })
    if (!ok) return
    const res = await act.run(cfg.action)
    if (res?.ok) {
      toast({ tone: 'success', title: `Engine: ${cfg.action} acknowledged` })
      void refreshStatus()
    } else {
      toast({ tone: 'error', title: `Engine ${cfg.action} failed`, message: act.error?.message ?? 'The API did not acknowledge the action.' })
    }
  }

  const disabled = (a: EngineAction) => {
    if (!e) return true
    if (a === 'pause') return e.paused || e.halted
    if (a === 'resume') return !e.paused && !e.halted && e.running
    if (a === 'halt') return e.halted
    return false
  }

  return (
    <Card
      title="Engine controls"
      subtitle="Each action asks for confirmation"
      actions={
        state && (
          <Badge tone={state === 'running' ? 'profit' : state === 'paused' ? 'warn' : 'loss'} dot pulse={state === 'running'}>
            {state}
          </Badge>
        )
      }
    >
      {!e ? (
        <SkeletonText lines={3} />
      ) : (
        <div className="space-y-4">
          <KeyValue
            cols={2}
            dense
            items={[
              { k: 'Last tick', v: `${fmtRelative(e.last_tick)} · ${fmtDateTime(e.last_tick)}` },
              { k: 'Uptime', v: fmtUptime(e.uptime_s) },
              { k: 'Ticks', v: fmtNum(e.tick_count) },
              { k: 'Errors (1h)', v: <span className={clsx(e.errors_1h > 0 && 'text-loss font-semibold')}>{fmtNum(e.errors_1h)}</span> },
            ]}
          />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {ACTIONS.map((cfg) => (
              <Button key={cfg.action} variant={cfg.variant} size="md" icon={<cfg.icon className="h-3.5 w-3.5" />} disabled={disabled(cfg.action)} loading={act.pending} onClick={() => void run(cfg)}>
                {cfg.label}
              </Button>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}

function UniverseCard() {
  const { status } = useApp()
  const u = status?.universe
  return (
    <Card title="Universe" subtitle={u ? `${u.equities.length} equities · ${u.crypto.length} crypto pairs` : undefined}>
      {!u ? (
        <SkeletonText lines={3} />
      ) : (
        <div className="space-y-4">
          <div>
            <SectionTitle className="mb-2">Equities (regular session)</SectionTitle>
            <div className="flex flex-wrap gap-1.5">
              {u.equities.map((s) => (
                <Badge key={s} tone="neutral" outline size="md" className="font-mono">
                  {s}
                </Badge>
              ))}
              {u.equities.length === 0 && <span className="text-[12px] text-muted">None configured.</span>}
            </div>
          </div>
          <div>
            <SectionTitle className="mb-2">Crypto (24/7)</SectionTitle>
            <div className="flex flex-wrap gap-1.5">
              {u.crypto.map((s) => (
                <Badge key={s} tone="neutral" outline size="md" className="font-mono">
                  {s}
                </Badge>
              ))}
              {u.crypto.length === 0 && <span className="text-[12px] text-muted">None configured.</span>}
            </div>
          </div>
        </div>
      )}
    </Card>
  )
}
