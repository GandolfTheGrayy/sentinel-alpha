import { ArrowLeftRight, FlaskConical, Layers, LayoutDashboard, Settings, Trophy, type LucideIcon } from 'lucide-react'

export interface NavItem {
  path: string
  label: string
  icon: LucideIcon
  description: string
}

export const NAV: NavItem[] = [
  { path: '/', label: 'Overview', icon: LayoutDashboard, description: 'Equity, market, population, budget and live activity' },
  { path: '/tournament', label: 'Tournament', icon: Trophy, description: 'Variant leaderboard and lifecycle' },
  { path: '/trades', label: 'Trades', icon: ArrowLeftRight, description: 'Closed trade tape with feature snapshots' },
  { path: '/factors', label: 'Factors', icon: Layers, description: 'Attribution report: what is working and why' },
  { path: '/research', label: 'Research', icon: FlaskConical, description: 'Claude sessions, proposals and lab memory' },
  { path: '/settings', label: 'Settings', icon: Settings, description: 'Budget, engine controls, config and universe' },
]

export function navFor(pathname: string): NavItem {
  return NAV.find((n) => (n.path === '/' ? pathname === '/' : pathname.startsWith(n.path))) ?? NAV[0]!
}
