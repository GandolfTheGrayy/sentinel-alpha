import clsx from 'clsx'
import type { ReactNode } from 'react'

interface CardProps {
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  className?: string
  bodyClassName?: string
  /** Remove body padding (for tables and charts that manage their own). */
  flush?: boolean
}

export function Card({ title, subtitle, actions, children, className, bodyClassName, flush }: CardProps) {
  const hasHeader = title || subtitle || actions
  return (
    <section className={clsx('card flex flex-col min-w-0', className)}>
      {hasHeader && (
        <header className="flex items-start justify-between gap-3 px-4 pt-3.5 pb-2.5">
          <div className="min-w-0">
            {title && <h2 className="text-[13px] font-semibold text-text leading-5 truncate">{title}</h2>}
            {subtitle && <div className="text-[11.5px] text-muted leading-4 mt-0.5">{subtitle}</div>}
          </div>
          {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </header>
      )}
      <div className={clsx('min-w-0 flex-1', flush ? '' : hasHeader ? 'px-4 pb-4' : 'p-4', bodyClassName)}>{children}</div>
    </section>
  )
}

export function SectionTitle({ children, className }: { children: ReactNode; className?: string }) {
  return <h3 className={clsx('text-[11px] font-medium uppercase tracking-wide text-muted', className)}>{children}</h3>
}
