import clsx from 'clsx'
import type { ReactNode } from 'react'
import type { EventKind, EventLevel, ExitReason, ProposalStatus, ProposalType, RunKind, RunStatus, VariantOrigin, VariantStatus } from '../../api/types'

export type Tone = 'accent' | 'profit' | 'loss' | 'warn' | 'claude' | 'info' | 'neutral' | 'dim'

const TONE_CLASSES: Record<Tone, string> = {
  accent: 'bg-accent/15 text-accent',
  profit: 'bg-profit/15 text-profit',
  loss: 'bg-loss/15 text-loss',
  warn: 'bg-warn/15 text-warn',
  claude: 'bg-claude/15 text-claude',
  info: 'bg-info/15 text-info',
  neutral: 'bg-muted/15 text-text',
  dim: 'bg-muted/10 text-muted',
}

const DOT_CLASSES: Record<Tone, string> = {
  accent: 'bg-accent',
  profit: 'bg-profit',
  loss: 'bg-loss',
  warn: 'bg-warn',
  claude: 'bg-claude',
  info: 'bg-info',
  neutral: 'bg-muted',
  dim: 'bg-faint',
}

interface BadgeProps {
  tone?: Tone
  children: ReactNode
  dot?: boolean
  pulse?: boolean
  size?: 'xs' | 'sm' | 'md'
  className?: string
  title?: string
  outline?: boolean
}

export function Badge({ tone = 'neutral', children, dot, pulse, size = 'sm', className, title, outline }: BadgeProps) {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-md font-medium whitespace-nowrap leading-none',
        size === 'xs' && 'px-1.5 py-[3px] text-[10px]',
        size === 'sm' && 'px-1.5 py-1 text-[11px]',
        size === 'md' && 'px-2 py-1.5 text-[12px]',
        outline ? 'border border-border bg-transparent text-text' : TONE_CLASSES[tone],
        className,
      )}
    >
      {dot && <span className={clsx('h-1.5 w-1.5 rounded-full shrink-0', DOT_CLASSES[tone], pulse && 'animate-pulse-dot')} />}
      {children}
    </span>
  )
}

export function Dot({ tone = 'neutral', pulse, className }: { tone?: Tone; pulse?: boolean; className?: string }) {
  return <span className={clsx('inline-block h-2 w-2 rounded-full', DOT_CLASSES[tone], pulse && 'animate-pulse-dot', className)} />
}

// ---------------------------------------------------------------------------
// Domain tone mappings
// ---------------------------------------------------------------------------

export function variantStatusTone(s: VariantStatus): Tone {
  switch (s) {
    case 'active':
      return 'profit'
    case 'incubating':
      return 'info'
    case 'probation':
      return 'warn'
    case 'paused':
      return 'neutral'
    case 'retired':
      return 'dim'
  }
}

export function originTone(o: VariantOrigin): Tone {
  switch (o) {
    case 'seed':
      return 'dim'
    case 'optimizer':
      return 'info'
    case 'claude':
      return 'claude'
  }
}

export function eventKindTone(k: EventKind): Tone {
  switch (k) {
    case 'fill':
      return 'profit'
    case 'signal':
      return 'info'
    case 'exit':
      return 'accent'
    case 'risk':
      return 'loss'
    case 'research':
      return 'claude'
    case 'tournament':
      return 'warn'
    case 'system':
      return 'neutral'
    case 'data':
      return 'dim'
  }
}

export function eventLevelTone(l: EventLevel): Tone {
  return l === 'error' ? 'loss' : l === 'warn' ? 'warn' : 'dim'
}

export function exitReasonTone(r: ExitReason): Tone {
  switch (r) {
    case 'target':
      return 'profit'
    case 'stop':
      return 'loss'
    case 'kill':
      return 'loss'
    case 'time':
      return 'neutral'
    case 'session_end':
      return 'dim'
    case 'signal':
      return 'info'
    case 'rebalance':
      return 'dim'
  }
}

export function runStatusTone(s: RunStatus): Tone {
  switch (s) {
    case 'ok':
      return 'profit'
    case 'error':
      return 'loss'
    case 'skipped':
      return 'dim'
    case 'running':
      return 'claude'
  }
}

export function runKindTone(k: RunKind): Tone {
  return k === 'strategist' ? 'claude' : 'info'
}

export function proposalStatusTone(s: ProposalStatus): Tone {
  switch (s) {
    case 'pending':
      return 'warn'
    case 'testing':
      return 'info'
    case 'accepted':
      return 'profit'
    case 'rejected':
      return 'dim'
  }
}

export function proposalTypeLabel(t: ProposalType): string {
  switch (t) {
    case 'param_change':
      return 'param change'
    case 'new_variant':
      return 'new variant'
    case 'filter':
      return 'filter'
    case 'retire':
      return 'retire'
    case 'new_strategy_code':
      return 'new strategy'
  }
}

export function StatusBadge({ status, size }: { status: VariantStatus; size?: BadgeProps['size'] }) {
  return (
    <Badge tone={variantStatusTone(status)} dot size={size}>
      {status}
    </Badge>
  )
}

export function OriginBadge({ origin, size }: { origin: VariantOrigin; size?: BadgeProps['size'] }) {
  return (
    <Badge tone={originTone(origin)} size={size}>
      {origin}
    </Badge>
  )
}
