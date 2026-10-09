/** Time helpers. All market-facing displays use America/New_York. */
import { nowMs } from './clock'

export const NY_TZ = 'America/New_York'

const partsFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: NY_TZ,
  hourCycle: 'h23',
  weekday: 'short',
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
})

export interface NYParts {
  year: number
  month: number
  day: number
  hour: number
  minute: number
  second: number
  /** 0 = Sunday ... 6 = Saturday */
  weekday: number
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export function nyParts(date: Date): NYParts {
  const out: Record<string, string> = {}
  for (const p of partsFmt.formatToParts(date)) out[p.type] = p.value
  return {
    year: Number(out.year),
    month: Number(out.month),
    day: Number(out.day),
    hour: Number(out.hour) % 24,
    minute: Number(out.minute),
    second: Number(out.second),
    weekday: WEEKDAYS.indexOf(out.weekday ?? 'Sun'),
  }
}

/** Offset (ms) between NY wall-clock and UTC at the given instant. */
export function nyOffsetMs(date: Date): number {
  const p = nyParts(date)
  const asUtc = Date.UTC(p.year, p.month - 1, p.day, p.hour, p.minute, p.second)
  return asUtc - Math.floor(date.getTime() / 1000) * 1000
}

/** Build an instant from a NY wall-clock time (same-day offset assumption). */
export function nyWallToInstant(base: Date, dayOffset: number, hour: number, minute: number): Date {
  const p = nyParts(base)
  const offset = nyOffsetMs(base)
  const wallUtc = Date.UTC(p.year, p.month - 1, p.day + dayOffset, hour, minute, 0)
  return new Date(wallUtc - offset)
}

const clockFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: NY_TZ,
  hourCycle: 'h23',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
})

export function nyClock(date: Date): string {
  return clockFmt.format(date)
}

const timeFmt = new Intl.DateTimeFormat('en-US', { timeZone: NY_TZ, hourCycle: 'h23', hour: '2-digit', minute: '2-digit' })
const timeSecFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: NY_TZ,
  hourCycle: 'h23',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
})
const dateFmt = new Intl.DateTimeFormat('en-US', { timeZone: NY_TZ, month: 'short', day: 'numeric' })
const dateYearFmt = new Intl.DateTimeFormat('en-US', { timeZone: NY_TZ, month: 'short', day: 'numeric', year: 'numeric' })
const dateTimeFmt = new Intl.DateTimeFormat('en-US', {
  timeZone: NY_TZ,
  hourCycle: 'h23',
  month: 'short',
  day: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
})
const weekdayFmt = new Intl.DateTimeFormat('en-US', { timeZone: NY_TZ, weekday: 'short', month: 'short', day: 'numeric' })

export function parseTs(iso: string | null | undefined): Date | null {
  if (!iso) return null
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d
}

export function fmtTime(iso: string | null | undefined, seconds = false): string {
  const d = parseTs(iso)
  if (!d) return '—'
  return (seconds ? timeSecFmt : timeFmt).format(d)
}

export function fmtDate(iso: string | null | undefined, withYear = false): string {
  const d = parseTs(iso)
  if (!d) return '—'
  return (withYear ? dateYearFmt : dateFmt).format(d)
}

export function fmtDateTime(iso: string | null | undefined): string {
  const d = parseTs(iso)
  if (!d) return '—'
  return dateTimeFmt.format(d)
}

export function fmtWeekday(iso: string | null | undefined): string {
  const d = parseTs(iso)
  if (!d) return '—'
  return weekdayFmt.format(d)
}

/**
 * "just now", "4m ago", "2h ago", "3d ago" — or "in 4m" for future instants.
 * `now` defaults to the app clock (simulated when the engine runs in sim mode).
 */
export function fmtRelative(iso: string | null | undefined, now: number = nowMs()): string {
  const d = parseTs(iso)
  if (!d) return '—'
  const diff = d.getTime() - now
  const abs = Math.abs(diff)
  const future = diff > 0
  let s: string
  if (abs < 45_000) s = future ? 'in a moment' : 'just now'
  else if (abs < 3_600_000) s = `${Math.round(abs / 60_000)}m`
  else if (abs < 86_400_000) s = `${Math.round(abs / 3_600_000)}h`
  else s = `${Math.round(abs / 86_400_000)}d`
  if (abs < 45_000) return s
  return future ? `in ${s}` : `${s} ago`
}

/** Countdown to an instant: "2h 14m 05s" / "03m 12s". `now` defaults to the app clock. */
export function fmtCountdown(iso: string | null | undefined, now: number = nowMs()): string {
  const d = parseTs(iso)
  if (!d) return '—'
  const diff = Math.max(0, d.getTime() - now)
  const totalS = Math.floor(diff / 1000)
  const days = Math.floor(totalS / 86400)
  const h = Math.floor((totalS % 86400) / 3600)
  const m = Math.floor((totalS % 3600) / 60)
  const s = totalS % 60
  const pad = (n: number) => n.toString().padStart(2, '0')
  if (days > 0) return `${days}d ${pad(h)}h ${pad(m)}m`
  if (h > 0) return `${h}h ${pad(m)}m ${pad(s)}s`
  return `${pad(m)}m ${pad(s)}s`
}

/** Tick formatter for chart x-axes depending on the span of the data. */
export function makeTickFormatter(spanMs: number): (iso: string) => string {
  if (spanMs <= 36 * 3_600_000) return (iso) => fmtTime(iso)
  if (spanMs <= 10 * 86_400_000) return (iso) => weekdayFmt.format(new Date(iso)).replace(',', '')
  return (iso) => fmtDate(iso)
}
