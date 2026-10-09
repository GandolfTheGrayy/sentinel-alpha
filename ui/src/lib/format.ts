/** Number formatting helpers. All inputs may be null/undefined and render as an em dash. */

const usd0 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 })
const usd2 = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 })
const num0 = new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 })
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

export const DASH = '—'

export function isNum(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v)
}

export interface SignOpts {
  /** Prefix a "+" for positive values. */
  sign?: boolean
}

function withSign(s: string, n: number, sign: boolean | undefined) {
  if (!sign) return s
  if (n > 0) return `+${s}`
  return s
}

/** $12,345 (0 decimals) or $12,345.67 when `decimals` is 2. */
export function fmtUsd(n: number | null | undefined, opts: SignOpts & { decimals?: 0 | 2; compact?: boolean } = {}): string {
  if (!isNum(n)) return DASH
  if (opts.compact && Math.abs(n) >= 10000) {
    const s = `$${compact.format(Math.abs(n))}`
    return n < 0 ? `-${s}` : withSign(s, n, opts.sign)
  }
  const f = opts.decimals === 2 ? usd2 : usd0
  const s = f.format(Math.abs(n))
  return n < 0 ? `-${s}` : withSign(s, n, opts.sign)
}

/** Percent where the input is already in percent units (0.23 => "0.23%"). */
export function fmtPct(n: number | null | undefined, opts: SignOpts & { digits?: number } = {}): string {
  if (!isNum(n)) return DASH
  const d = opts.digits ?? 2
  const s = `${Math.abs(n).toFixed(d)}%`
  return n < 0 ? `-${s}` : withSign(s, n, opts.sign)
}

/** Percent where the input is a fraction (0.52 => "52%"). */
export function fmtFrac(n: number | null | undefined, opts: SignOpts & { digits?: number } = {}): string {
  if (!isNum(n)) return DASH
  return fmtPct(n * 100, { ...opts, digits: opts.digits ?? 0 })
}

/** R-multiple with sign: "+0.42R". */
export function fmtR(n: number | null | undefined, digits = 2): string {
  if (!isNum(n)) return DASH
  const s = `${Math.abs(n).toFixed(digits)}R`
  if (n < 0) return `-${s}`
  if (n > 0) return `+${s}`
  return s
}

export function fmtNum(n: number | null | undefined, digits = 0): string {
  if (!isNum(n)) return DASH
  if (digits === 0) return num0.format(n)
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function fmtCompact(n: number | null | undefined): string {
  if (!isNum(n)) return DASH
  return compact.format(n)
}

/** Price formatting: 2 decimals for > 10, 4 decimals for small prices. */
export function fmtPrice(n: number | null | undefined): string {
  if (!isNum(n)) return DASH
  const digits = Math.abs(n) >= 1000 ? 2 : Math.abs(n) >= 10 ? 2 : 4
  return n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function fmtQty(n: number | null | undefined): string {
  if (!isNum(n)) return DASH
  if (Number.isInteger(n)) return num0.format(n)
  return n.toLocaleString('en-US', { maximumFractionDigits: 4 })
}

export function fmtX(n: number | null | undefined, digits = 2): string {
  if (!isNum(n)) return DASH
  return `${n.toFixed(digits)}x`
}

/** Minutes -> "45m", "3h 10m", "2d 4h". */
export function fmtDuration(minutes: number | null | undefined): string {
  if (!isNum(minutes)) return DASH
  const m = Math.round(minutes)
  if (m < 60) return `${m}m`
  const h = Math.floor(m / 60)
  const rem = m % 60
  if (h < 24) return rem ? `${h}h ${rem}m` : `${h}h`
  const d = Math.floor(h / 24)
  const hr = h % 24
  return hr ? `${d}d ${hr}h` : `${d}d`
}

/** Seconds -> "1d 02:14:05" style uptime. */
export function fmtUptime(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return DASH
  const s = Math.max(0, Math.floor(seconds))
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  if (d > 0) return `${d}d ${h}h ${m}m`
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export function fmtTokens(n: number | null | undefined): string {
  if (!isNum(n)) return DASH
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 100_000 ? 0 : 1)}k`
  return num0.format(n)
}

/** CSS class for a signed value. Zero/undefined -> muted. */
export function signClass(n: number | null | undefined, opts: { zeroMuted?: boolean } = {}): string {
  if (!isNum(n)) return 'text-muted'
  if (n > 0) return 'text-profit'
  if (n < 0) return 'text-loss'
  return opts.zeroMuted === false ? 'text-text' : 'text-muted'
}

/** Format a feature-snapshot value for display. */
export function fmtFeature(name: string, v: unknown): string {
  if (v === null || v === undefined) return DASH
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'string') return v
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return DASH
    if (Number.isInteger(v)) return num0.format(v)
    if (/pct|ret_|dist|gap|vol|dd|frac|share/.test(name)) return v.toFixed(3)
    return Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2)
  }
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
}

export function clamp(n: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, n))
}

export function titleCase(s: string) {
  return s.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}
