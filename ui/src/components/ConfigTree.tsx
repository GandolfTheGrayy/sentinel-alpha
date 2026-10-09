import clsx from 'clsx'
import { ChevronRight } from 'lucide-react'
import { useState } from 'react'
import type { ConfigValue } from '../api/types'

function isObject(v: ConfigValue): v is { [key: string]: ConfigValue } {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

function Leaf({ value }: { value: ConfigValue }) {
  if (value === null) return <span className="text-faint italic">null</span>
  if (typeof value === 'boolean') return <span className="text-warn">{String(value)}</span>
  if (typeof value === 'number') return <span className="text-info">{String(value)}</span>
  if (typeof value === 'string') return <span className="text-profit">"{value}"</span>
  return <span className="text-muted">{JSON.stringify(value)}</span>
}

function Node({ name, value, depth, defaultOpen }: { name: string; value: ConfigValue; depth: number; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen)
  const isBranch = isObject(value) || (Array.isArray(value) && value.some((v) => isObject(v)))
  const isList = Array.isArray(value) && !isBranch

  if (!isBranch) {
    return (
      <div className="flex items-baseline gap-2 py-[3px] font-mono text-[11.5px]" style={{ paddingLeft: depth * 16 + 20 }}>
        <span className="text-text">{name}</span>
        <span className="text-faint">:</span>
        {isList ? (
          <span className="flex flex-wrap gap-1">
            {(value as ConfigValue[]).map((v, i) => (
              <span key={i} className="rounded bg-muted/10 px-1 text-muted">
                <Leaf value={v} />
              </span>
            ))}
            {(value as ConfigValue[]).length === 0 && <span className="text-faint">[]</span>}
          </span>
        ) : (
          <Leaf value={value} />
        )}
      </div>
    )
  }

  const entries: [string, ConfigValue][] = isObject(value) ? Object.entries(value) : (value as ConfigValue[]).map((v, i) => [String(i), v])

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 py-[3px] font-mono text-[11.5px] w-full text-left hover:bg-muted/10 rounded"
        style={{ paddingLeft: depth * 16 + 4 }}
        aria-expanded={open}
      >
        <ChevronRight className={clsx('h-3.5 w-3.5 text-faint transition-transform', open && 'rotate-90')} />
        <span className="text-text font-semibold">{name}</span>
        <span className="text-faint">{isObject(value) ? `{${entries.length}}` : `[${entries.length}]`}</span>
      </button>
      {open && (
        <div>
          {entries.map(([k, v]) => (
            <Node key={k} name={k} value={v} depth={depth + 1} defaultOpen={depth < 0} />
          ))}
        </div>
      )}
    </div>
  )
}

export function ConfigTree({ config }: { config: { [key: string]: ConfigValue } }) {
  const entries = Object.entries(config)
  if (entries.length === 0) return <p className="text-[12px] text-muted">Empty config.</p>
  return (
    <div className="rounded-lg border border-border bg-panel-2 py-1.5 px-1 overflow-x-auto">
      {entries.map(([k, v]) => (
        <Node key={k} name={k} value={v} depth={0} defaultOpen />
      ))}
    </div>
  )
}
