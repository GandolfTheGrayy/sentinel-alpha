import clsx from 'clsx'
import { X } from 'lucide-react'
import { useEffect, type ReactNode } from 'react'
import { IconButton } from './Button'

interface DrawerProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  subtitle?: ReactNode
  actions?: ReactNode
  children?: ReactNode
  /** Tailwind max-width class for the panel (desktop). */
  widthClass?: string
}

export function Drawer({ open, onClose, title, subtitle, actions, children, widthClass = 'sm:max-w-[640px]' }: DrawerProps) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/50 animate-fade-in" onClick={onClose} aria-hidden />
      <aside className={clsx('relative h-full w-full bg-panel border-l border-border shadow-2xl flex flex-col animate-slide-in', widthClass)}>
        <header className="flex items-start justify-between gap-3 px-5 py-4 border-b border-border shrink-0">
          <div className="min-w-0">
            {title && <h2 className="text-[14px] font-semibold text-text leading-5 truncate">{title}</h2>}
            {subtitle && <div className="text-[12px] text-muted mt-0.5">{subtitle}</div>}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {actions}
            <IconButton label="Close" onClick={onClose}>
              <X className="h-4 w-4" />
            </IconButton>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto px-5 py-4">{children}</div>
      </aside>
    </div>
  )
}
