import clsx from 'clsx'
import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { Skeleton } from './Skeleton'

interface KpiTileProps {
  label: string
  value: ReactNode
  /** Secondary line under the value (e.g. a % delta). */
  sub?: ReactNode
  subClassName?: string
  valueClassName?: string
  icon?: LucideIcon
  loading?: boolean
  className?: string
  /** Optional element rendered at the bottom (meter, sparkline). */
  footer?: ReactNode
}

export function KpiTile({ label, value, sub, subClassName, valueClassName, icon: Icon, loading, className, footer }: KpiTileProps) {
  return (
    <div className={clsx('card px-4 py-3 min-w-0', className)}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-medium uppercase tracking-wide text-muted truncate">{label}</div>
        {Icon && <Icon className="h-3.5 w-3.5 text-faint shrink-0 hidden sm:block" aria-hidden />}
      </div>
      {loading ? (
        <div className="mt-2 space-y-1.5">
          <Skeleton className="h-6 w-28" />
          <Skeleton className="h-3 w-16" />
        </div>
      ) : (
        <>
          <div className={clsx('mt-1 text-[20px] font-semibold leading-7 text-text truncate', valueClassName)}>{value}</div>
          {sub !== undefined && <div className={clsx('text-[11.5px] leading-4 mt-0.5 truncate', subClassName ?? 'text-muted')}>{sub}</div>}
        </>
      )}
      {footer && <div className="mt-2">{footer}</div>}
    </div>
  )
}
