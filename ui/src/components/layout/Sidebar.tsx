import clsx from 'clsx'
import { PanelLeftClose, PanelLeftOpen } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { NAV } from '../../lib/nav'

interface SidebarProps {
  collapsed: boolean
  canExpand: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, canExpand, onToggle }: SidebarProps) {
  return (
    <aside
      className={clsx(
        'h-full shrink-0 flex flex-col border-r border-border bg-panel transition-[width] duration-200',
        collapsed ? 'w-[56px]' : 'w-[216px]',
      )}
    >
      <div className={clsx('flex items-center h-14 border-b border-border', collapsed ? 'justify-center' : 'px-4 gap-2.5')}>
        <Logo />
        {!collapsed && (
          <div className="leading-tight min-w-0">
            <div className="text-[13.5px] font-semibold text-text tracking-tight">Sentinel</div>
            <div className="text-[10.5px] text-muted">v2 · paper-trading lab</div>
          </div>
        )}
      </div>
      <nav className={clsx('flex-1 py-3', collapsed ? 'px-2' : 'px-2.5')} aria-label="Primary">
        <ul className="space-y-0.5">
          {NAV.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                end={item.path === '/'}
                title={collapsed ? item.label : undefined}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center rounded-lg text-[12.5px] font-medium transition-colors',
                    collapsed ? 'justify-center h-9 w-9 mx-auto' : 'gap-2.5 px-2.5 h-9',
                    isActive ? 'bg-accent/12 text-accent' : 'text-muted hover:text-text hover:bg-muted/10',
                  )
                }
              >
                <item.icon className="h-4 w-4 shrink-0" aria-hidden />
                {!collapsed && <span className="truncate">{item.label}</span>}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      {canExpand && (
        <div className={clsx('border-t border-border p-2', collapsed ? 'flex justify-center' : '')}>
          <button
            type="button"
            onClick={onToggle}
            className={clsx(
              'flex items-center rounded-lg text-[12px] text-muted hover:text-text hover:bg-muted/10 transition-colors',
              collapsed ? 'justify-center h-9 w-9' : 'gap-2.5 px-2.5 h-9 w-full',
            )}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      )}
    </aside>
  )
}

function Logo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <rect width="32" height="32" rx="8" fill="var(--bg)" stroke="var(--border)" />
      <polyline points="6,22 11.5,16 15.5,19 21,10 26,13" fill="none" stroke="var(--accent)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="26" cy="13" r="2.4" fill="var(--accent)" />
    </svg>
  )
}
