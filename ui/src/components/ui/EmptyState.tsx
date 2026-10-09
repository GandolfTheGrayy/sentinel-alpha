import clsx from 'clsx'
import { Inbox, type LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'

interface EmptyStateProps {
  icon?: LucideIcon
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
  compact?: boolean
}

export function EmptyState({ icon: Icon = Inbox, title, description, action, className, compact }: EmptyStateProps) {
  return (
    <div className={clsx('flex flex-col items-center justify-center text-center', compact ? 'py-6 px-4' : 'py-12 px-6', className)}>
      <div className="h-9 w-9 rounded-full bg-muted/10 flex items-center justify-center text-muted mb-3">
        <Icon className="h-4 w-4" aria-hidden />
      </div>
      <div className="text-[13px] font-medium text-text">{title}</div>
      {description && <div className="text-[12px] text-muted mt-1 max-w-sm leading-relaxed">{description}</div>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export function ErrorState({ error, onRetry, className }: { error: Error | string; onRetry?: () => void; className?: string }) {
  const message = typeof error === 'string' ? error : error.message
  return (
    <div className={clsx('flex flex-col items-center justify-center text-center py-8 px-6', className)}>
      <div className="text-[13px] font-medium text-loss">Failed to load</div>
      <div className="text-[12px] text-muted mt-1 max-w-sm break-words">{message}</div>
      {onRetry && (
        <button type="button" onClick={onRetry} className="mt-3 text-[12px] text-accent hover:underline">
          Retry
        </button>
      )}
    </div>
  )
}
