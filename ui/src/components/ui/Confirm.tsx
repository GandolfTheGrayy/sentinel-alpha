import { AlertTriangle } from 'lucide-react'
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Button } from './Button'

export interface ConfirmOptions {
  title: string
  message?: ReactNode
  confirmLabel?: string
  cancelLabel?: string
  tone?: 'primary' | 'danger' | 'warn'
}

type ConfirmFn = (opts: ConfirmOptions) => Promise<boolean>

const ConfirmContext = createContext<ConfirmFn | null>(null)

interface Pending {
  opts: ConfirmOptions
  resolve: (v: boolean) => void
}

export function ConfirmProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null)
  const confirmBtn = useRef<HTMLButtonElement>(null)

  const confirm = useCallback<ConfirmFn>((opts) => {
    return new Promise<boolean>((resolve) => {
      setPending({ opts, resolve })
    })
  }, [])

  const close = useCallback(
    (v: boolean) => {
      pending?.resolve(v)
      setPending(null)
    },
    [pending],
  )

  useEffect(() => {
    if (!pending) return
    confirmBtn.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close(false)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [pending, close])

  const value = useMemo(() => confirm, [confirm])

  const tone = pending?.opts.tone ?? 'primary'

  return (
    <ConfirmContext.Provider value={value}>
      {children}
      {pending && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
          <div className="absolute inset-0 bg-black/55 animate-fade-in" onClick={() => close(false)} aria-hidden />
          <div className="relative card w-full max-w-md p-5 shadow-2xl animate-fade-in">
            <div className="flex gap-3">
              {tone !== 'primary' && (
                <div className={`h-8 w-8 rounded-full flex items-center justify-center shrink-0 ${tone === 'danger' ? 'bg-loss/15 text-loss' : 'bg-warn/15 text-warn'}`}>
                  <AlertTriangle className="h-4 w-4" aria-hidden />
                </div>
              )}
              <div className="min-w-0">
                <h2 id="confirm-title" className="text-[14px] font-semibold text-text">
                  {pending.opts.title}
                </h2>
                {pending.opts.message && <div className="text-[12.5px] text-muted mt-1.5 leading-relaxed">{pending.opts.message}</div>}
              </div>
            </div>
            <div className="flex justify-end gap-2 mt-5">
              <Button variant="ghost" onClick={() => close(false)}>
                {pending.opts.cancelLabel ?? 'Cancel'}
              </Button>
              <Button ref={confirmBtn} variant={tone === 'danger' ? 'danger' : tone === 'warn' ? 'warn' : 'primary'} onClick={() => close(true)}>
                {pending.opts.confirmLabel ?? 'Confirm'}
              </Button>
            </div>
          </div>
        </div>
      )}
    </ConfirmContext.Provider>
  )
}

export function useConfirm(): ConfirmFn {
  const ctx = useContext(ConfirmContext)
  if (!ctx) throw new Error('useConfirm must be used within ConfirmProvider')
  return ctx
}
