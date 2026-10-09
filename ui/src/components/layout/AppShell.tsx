import { X } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { useMediaQuery } from '../../api/hooks'
import { useApp } from '../../state/AppContext'
import { IconButton } from '../ui/Button'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'

const SIDEBAR_KEY = 'sentinel.sidebar'

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_KEY) === 'collapsed'
  } catch {
    return false
  }
}

export function AppShell({ children }: { children: ReactNode }) {
  const isWide = useMediaQuery('(min-width: 900px)')
  const [userCollapsed, setUserCollapsed] = useState(readCollapsed)
  const collapsed = !isWide || userCollapsed

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_KEY, userCollapsed ? 'collapsed' : 'expanded')
    } catch {
      // ignore
    }
  }, [userCollapsed])

  return (
    <div className="h-full flex bg-bg text-text">
      <Sidebar collapsed={collapsed} canExpand={isWide} onToggle={() => setUserCollapsed((v) => !v)} />
      <div className="flex-1 min-w-0 flex flex-col h-full">
        <TopBar />
        <MockBanner />
        <main className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-3 sm:p-4 lg:p-5 max-w-[1680px] mx-auto">{children}</div>
        </main>
      </div>
    </div>
  )
}

function MockBanner() {
  const { apiMode, mockReason, bannerDismissed, dismissBanner } = useApp()
  if (apiMode !== 'mock' || mockReason !== 'fallback' || bannerDismissed) return null
  return (
    <div className="shrink-0 flex items-center gap-3 px-4 py-2 bg-warn/12 border-b border-warn/30 text-[12px] text-text" role="status">
      <span className="h-2 w-2 rounded-full bg-warn shrink-0" />
      <span className="min-w-0">
        <strong className="font-semibold">Backend unreachable</strong> — showing demo data. Start the API on <code className="font-mono text-[11px]">127.0.0.1:8787</code> and reload to see live data.
      </span>
      <IconButton label="Dismiss" size="xs" className="ml-auto" onClick={dismissBanner}>
        <X className="h-3.5 w-3.5" />
      </IconButton>
    </div>
  )
}
