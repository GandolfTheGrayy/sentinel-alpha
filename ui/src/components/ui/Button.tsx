import clsx from 'clsx'
import { LoaderCircle } from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode, Ref } from 'react'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'warn' | 'claude'
type Size = 'xs' | 'sm' | 'md'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
  icon?: ReactNode
  iconRight?: ReactNode
  ref?: Ref<HTMLButtonElement>
}

const VARIANT: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:brightness-110 border border-transparent',
  secondary: 'bg-panel-2 text-text border border-border hover:border-border-strong hover:bg-panel',
  ghost: 'bg-transparent text-muted border border-transparent hover:text-text hover:bg-muted/10',
  danger: 'bg-loss/15 text-loss border border-loss/30 hover:bg-loss/25',
  warn: 'bg-warn/15 text-warn border border-warn/30 hover:bg-warn/25',
  claude: 'bg-claude/15 text-claude border border-claude/30 hover:bg-claude/25',
}

const SIZE: Record<Size, string> = {
  xs: 'h-6 px-2 text-[11px] gap-1 rounded-md',
  sm: 'h-7 px-2.5 text-[12px] gap-1.5 rounded-md',
  md: 'h-8 px-3 text-[12.5px] gap-2 rounded-lg',
}

export function Button({ variant = 'secondary', size = 'sm', loading, icon, iconRight, className, children, disabled, type = 'button', ref, ...rest }: ButtonProps) {
  return (
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      className={clsx(
        'inline-flex items-center justify-center font-medium whitespace-nowrap transition-colors select-none disabled:opacity-50',
        VARIANT[variant],
        SIZE[size],
        className,
      )}
      {...rest}
    >
      {loading ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" aria-hidden /> : icon}
      {children}
      {iconRight}
    </button>
  )
}

export function IconButton({ label, className, size = 'sm', children, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; size?: Size }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={clsx(
        'inline-flex items-center justify-center rounded-md text-muted hover:text-text hover:bg-muted/10 transition-colors disabled:opacity-50',
        size === 'xs' && 'h-6 w-6',
        size === 'sm' && 'h-7 w-7',
        size === 'md' && 'h-8 w-8',
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  )
}
