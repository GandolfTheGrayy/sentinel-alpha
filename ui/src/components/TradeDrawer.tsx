import clsx from 'clsx'
import type { Trade } from '../api/types'
import { fmtDuration, fmtFeature, fmtPrice, fmtQty, fmtR, fmtUsd, signClass } from '../lib/format'
import { fmtDateTime } from '../lib/time'
import { Badge, exitReasonTone } from './ui/Badge'
import { SectionTitle } from './ui/Card'
import { Drawer } from './ui/Drawer'
import { KeyValue } from './ui/KeyValue'

interface TradeDrawerProps {
  trade: Trade | null
  onClose: () => void
}

export function TradeDrawer({ trade: t, onClose }: TradeDrawerProps) {
  return (
    <Drawer
      open={t !== null}
      onClose={onClose}
      title={
        t && (
          <span className="inline-flex items-center gap-2">
            <span>
              {t.symbol} <span className="text-muted font-normal">#{t.id}</span>
            </span>
            <Badge tone={t.side === 'long' ? 'profit' : 'loss'} size="xs">
              {t.side}
            </Badge>
            <Badge tone={exitReasonTone(t.exit_reason)} size="xs">
              {t.exit_reason}
            </Badge>
          </span>
        )
      }
      subtitle={t && <span className="font-mono">{t.variant_id}</span>}
    >
      {t && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <Stat label="P&L" value={fmtUsd(t.pnl, { sign: true, decimals: 2 })} cls={signClass(t.pnl)} />
            <Stat label="R-multiple" value={fmtR(t.pnl_r)} cls={signClass(t.pnl_r)} />
            <Stat label="Hold" value={fmtDuration(t.hold_minutes)} />
            <Stat label="Fees" value={fmtUsd(t.fees, { decimals: 2 })} />
          </div>

          <section>
            <SectionTitle className="mb-2">Execution</SectionTitle>
            <KeyValue
              dense
              items={[
                { k: 'Entry', v: `${fmtPrice(t.entry_price)} · ${fmtDateTime(t.entry_ts)}` },
                { k: 'Exit', v: `${fmtPrice(t.exit_price)} · ${fmtDateTime(t.exit_ts)}` },
                { k: 'Quantity', v: fmtQty(t.qty) },
                { k: 'Family', v: t.family },
              ]}
            />
          </section>

          <section>
            <SectionTitle className="mb-2">Excursion (MAE / MFE)</SectionTitle>
            <ExcursionBar mae={t.mae_r} mfe={t.mfe_r} result={t.pnl_r} />
          </section>

          <section>
            <SectionTitle className="mb-2">Entry reason</SectionTitle>
            <p className="text-[12.5px] text-text leading-relaxed rounded-lg border border-border bg-panel-2 px-3 py-2">{t.reason || '—'}</p>
          </section>

          <section>
            <SectionTitle className="mb-2">Feature snapshot ({Object.keys(t.features).length})</SectionTitle>
            {Object.keys(t.features).length === 0 ? (
              <p className="text-[12px] text-muted">No features recorded for this trade.</p>
            ) : (
              <KeyValue
                dense
                cols={2}
                items={Object.entries(t.features)
                  .sort(([a], [b]) => a.localeCompare(b))
                  .map(([k, v]) => ({ k: <span className="font-mono text-[11px]">{k}</span>, v: fmtFeature(k, v), mono: true }))}
              />
            )}
          </section>
        </div>
      )}
    </Drawer>
  )
}

function Stat({ label, value, cls }: { label: string; value: string; cls?: string }) {
  return (
    <div className="rounded-lg border border-border bg-panel-2 px-3 py-2 min-w-0">
      <div className="text-[10.5px] uppercase tracking-wide text-muted">{label}</div>
      <div className={clsx('text-[15px] font-semibold mt-0.5 truncate', cls ?? 'text-text')}>{value}</div>
    </div>
  )
}

/** Horizontal excursion bar: MAE to the left of zero (rose), MFE to the right (teal), result marker. */
function ExcursionBar({ mae, mfe, result }: { mae: number; mfe: number; result: number }) {
  const lo = Math.min(mae, result, -0.5, 0)
  const hi = Math.max(mfe, result, 0.5, 0)
  const span = hi - lo || 1
  const pct = (v: number) => ((v - lo) / span) * 100
  return (
    <div>
      <div className="relative h-7 rounded-md bg-panel-2 border border-border overflow-hidden">
        <div className="absolute top-0 bottom-0 bg-loss/35" style={{ left: `${pct(mae)}%`, width: `${pct(0) - pct(mae)}%` }} />
        <div className="absolute top-0 bottom-0 bg-profit/35" style={{ left: `${pct(0)}%`, width: `${pct(mfe) - pct(0)}%` }} />
        <div className="absolute top-0 bottom-0 w-px bg-text/50" style={{ left: `${pct(0)}%` }} />
        <div
          className={clsx('absolute top-1 bottom-1 w-1 rounded-full', result >= 0 ? 'bg-profit' : 'bg-loss')}
          style={{ left: `calc(${pct(result)}% - 2px)` }}
          title={`Result ${fmtR(result)}`}
        />
      </div>
      <div className="flex justify-between text-[11px] mt-1">
        <span className="text-loss">MAE {fmtR(mae)}</span>
        <span className={clsx('font-medium', signClass(result))}>result {fmtR(result)}</span>
        <span className="text-profit">MFE {fmtR(mfe)}</span>
      </div>
    </div>
  )
}
