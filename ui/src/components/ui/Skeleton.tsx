import clsx from 'clsx'

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx('skeleton', className)} aria-hidden />
}

export function SkeletonText({ lines = 3, className }: { lines?: number; className?: string }) {
  return (
    <div className={clsx('space-y-2', className)} aria-hidden>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton key={i} className={clsx('h-3', i === lines - 1 ? 'w-2/3' : 'w-full')} />
      ))}
    </div>
  )
}

export function SkeletonRows({ rows = 6, cols = 5, className }: { rows?: number; cols?: number; className?: string }) {
  return (
    <div className={clsx('px-4 py-2 space-y-2.5', className)} aria-hidden>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={c} className={clsx('h-3.5', c === 0 ? 'w-32' : 'flex-1')} />
          ))}
        </div>
      ))}
    </div>
  )
}

export function SkeletonChart({ height = 220, className }: { height?: number; className?: string }) {
  return (
    <div className={clsx('relative w-full', className)} style={{ height }} aria-hidden>
      <Skeleton className="absolute inset-0 rounded-lg" />
    </div>
  )
}
