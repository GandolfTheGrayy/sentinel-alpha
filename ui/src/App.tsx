import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ConfirmProvider } from './components/ui/Confirm'
import { ToastProvider } from './components/ui/Toast'
import { FactorsPage } from './pages/Factors'
import { OverviewPage } from './pages/Overview'
import { ResearchPage } from './pages/Research'
import { SettingsPage } from './pages/Settings'
import { TournamentPage } from './pages/Tournament'
import { TradesPage } from './pages/Trades'
import { AppProvider } from './state/AppContext'

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <ConfirmProvider>
          <AppProvider>
            <AppShell>
              <Routes>
                <Route path="/" element={<OverviewPage />} />
                <Route path="/tournament" element={<TournamentPage />} />
                <Route path="/trades" element={<TradesPage />} />
                <Route path="/factors" element={<FactorsPage />} />
                <Route path="/research" element={<ResearchPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </AppShell>
          </AppProvider>
        </ConfirmProvider>
      </ToastProvider>
    </BrowserRouter>
  )
}
