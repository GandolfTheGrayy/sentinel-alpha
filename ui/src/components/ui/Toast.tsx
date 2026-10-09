import clsx from 'clsx'
import { CircleAlert, CircleCheck, Info, X } from 'lucide-react'
import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from 'react'

export interface ToastOptions {
  tone?: 'success' | 'error' | 'info' | 'warn'
  title: string
  message?: string
  ttlMs?: number
}

interface ToastItem extends ToastOptions {
  id: number
}

type ToastFn = (opts: ToastOptions) => void

const ToastContext = createContext<ToastFn | null>(null)

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([])
  const nextId = useRef(1)

  const dismiss = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback<ToastFn>(
    (opts) => {
      const id = nextId.current++
      setItems((prev) => [...prev.slice(-4), { ...opts, id }])
      const ttl = opts.ttlMs ?? (opts.tone === 'error' ? 8000 : 4500)
      setTimeout(() => dismiss(id), ttl)
    },
    [dismiss],
  )

  const value = useMemo(() => toast, [toast])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-[calc(100%-2rem)] max-w-sm pointer-events-none" aria-live="polite">
        {items.map((t) => {
          const tone = t.tone ?? 'info'
          const Icon = tone === 'success' ? CircleCheck : tone === 'error' || tone === 'warn' ? CircleAlert : Info
          return (
            <div key={t.id} className="card pointer-events-auto flex items-start gap-3 px-3.5 py-3 shadow-xl animate-slide-in">
              <Icon
                className={clsx(
                  'h-4 w-4 mt-0.5 shrink-0',
                  tone === 'success' && 'text-profit',
                  tone === 'error' && 'text-loss',
                  tone === 'warn' && 'text-warn',
                  tone === 'info' && 'text-info',
                )}
                aria-hidden
              />
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] font-medium text-text">{t.title}</div>
                {t.message && <div className="text-[12px] text-muted mt-0.5 break-words">{t.message}</div>}
              </div>
              <button type="button" aria-label="Dismiss" onClick={() => dismiss(t.id)} className="text-muted hover:text-text shrink-0">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): ToastFn {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}
