import React, { useState, useEffect } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Layers,
  Zap,
  FileText,
  GitGraph,
  Swords,
  Activity
} from 'lucide-react'

// Original Pages
import TracesPage from './pages/TracesPage'
import TraceDetailPage from './pages/TraceDetailPage'
import AnalyticsPage from './pages/AnalyticsPage'

// New Pages (Make sure these files exist in src/pages/)
import PromptsPage from './pages/PromptsPage'
import FlowPage from './pages/FlowPage'
import ArenaPage from './pages/ArenaPage'

// API base URL
export const API_URL = import.meta.env.DEV ? 'http://localhost:8787' : '/api'

function App() {
  const [isConnected, setIsConnected] = useState(false)

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_URL}/health`)
        setIsConnected(res.ok)
      } catch {
        setIsConnected(false)
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen flex bg-[var(--bg-primary)]">
      {/* Sidebar */}
      <aside className="w-[260px] flex flex-col border-r border-[var(--border-subtle)]"
        style={{
          background: 'linear-gradient(180deg, #0C0C10 0%, #08080A 100%)',
        }}
      >
        {/* Logo */}
        <div className="p-6 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center relative"
            style={{
              background: 'linear-gradient(135deg, #3B82F6, #8B5CF6)',
              boxShadow: '0 4px 16px rgba(59, 130, 246, 0.3)',
            }}
          >
            <Zap className="w-5 h-5 text-white fill-current" />
          </div>
          <div>
            <span className="font-bold text-[15px] tracking-tight text-white">AgentTracer</span>
            <p className="text-[10px] text-[#52525B] font-medium tracking-wider uppercase">Observability</p>
          </div>
        </div>

        <nav className="flex-1 px-3 mt-2">
          <p className="px-3 text-[10px] font-bold text-[#3F3F46] uppercase tracking-[0.12em] mb-3 mt-2">
            Monitor
          </p>
          <div className="space-y-0.5">
            <NavItem to="/" icon={<Layers className="w-[18px] h-[18px]" />} label="Traces" />
            <NavItem to="/analytics" icon={<LayoutDashboard className="w-[18px] h-[18px]" />} label="Analytics" />
          </div>

          <p className="px-3 text-[10px] font-bold text-[#3F3F46] uppercase tracking-[0.12em] mb-3 mt-8">
            Intelligence
          </p>
          <div className="space-y-0.5">
            <NavItem to="/prompts" icon={<FileText className="w-[18px] h-[18px]" />} label="Prompt Registry" />
            <NavItem to="/flow" icon={<GitGraph className="w-[18px] h-[18px]" />} label="Flow Graph" />
            <NavItem to="/arena" icon={<Swords className="w-[18px] h-[18px]" />} label="Model Arena" />
          </div>
        </nav>

        {/* Connection Status */}
        <div className="p-4">
          <div className="flex items-center gap-3 px-4 py-3 rounded-xl"
            style={{
              background: 'linear-gradient(145deg, rgba(19, 19, 24, 0.8), rgba(14, 14, 18, 0.9))',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div className="relative">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-400' : 'bg-red-500'}`} />
              {isConnected && (
                <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-60" />
              )}
            </div>
            <div className="flex-1">
              <p className="text-xs font-medium text-[#A1A1AA]">
                {isConnected ? 'System Online' : 'Offline'}
              </p>
            </div>
            <Activity className={`w-3.5 h-3.5 ${isConnected ? 'text-emerald-500' : 'text-red-500'}`} />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        <div className="flex-1 overflow-auto bg-[var(--bg-primary)]">
          <Routes>
            {/* Existing Routes */}
            <Route path="/" element={<TracesPage />} />
            <Route path="/trace/:traceId" element={<TraceDetailPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />

            {/* New Routes */}
            <Route path="/prompts" element={<PromptsPage />} />
            <Route path="/flow" element={<FlowPage />} />
            <Route path="/arena" element={<ArenaPage />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}

function NavItem({ to, icon, label }) {
  return (
    <NavLink to={to} className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
      {icon}
      <span>{label}</span>
    </NavLink>
  )
}

export default App