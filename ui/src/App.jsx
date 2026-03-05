import React, { useState, useEffect } from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Layers,
  Zap,
  FileText,
  GitGraph,
  Swords,
  Activity,
  ChevronDown,
  ChevronRight
} from 'lucide-react'

import RunsPage from './pages/TracesPage'
import RunDetailPage from './pages/TraceDetailPage'
import AnalyticsPage from './pages/AnalyticsPage'
import PromptsPage from './pages/PromptsPage'
import FlowPage from './pages/FlowPage'
import ArenaPage from './pages/ArenaPage'

export const API_URL = import.meta.env.DEV ? 'http://localhost:8787' : '/api'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [monitorOpen, setMonitorOpen] = useState(true)
  const [intelligenceOpen, setIntelligenceOpen] = useState(true)

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
    <div style={{ minHeight: '100vh', display: 'flex', background: 'var(--bg-primary)' }}>
      {/* Sidebar */}
      <aside style={{
        width: 220,
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg-surface)',
        borderRight: '1px solid var(--border-faint)',
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div style={{ padding: '16px 14px 12px', borderBottom: '1px solid var(--border-faint)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div style={{
              width: 28,
              height: 28,
              borderRadius: 7,
              background: 'var(--accent)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}>
              <Zap style={{ width: 14, height: 14, color: '#fff', fill: '#fff' }} />
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>
                AgentTracer
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 1 }}>
                Observability
              </div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '10px 8px', overflowY: 'auto' }}>

          {/* Monitor section */}
          <button
            onClick={() => setMonitorOpen(v => !v)}
            className="sidebar-section-toggle"
            style={{ width: '100%', marginBottom: 4 }}
          >
            {monitorOpen
              ? <ChevronDown style={{ width: 10, height: 10 }} />
              : <ChevronRight style={{ width: 10, height: 10 }} />
            }
            Monitor
          </button>

          {monitorOpen && (
            <div style={{ marginBottom: 6 }}>
              <NavItem to="/" icon={<Layers style={{ width: 14, height: 14 }} />} label="Runs" />
              <NavItem to="/analytics" icon={<LayoutDashboard style={{ width: 14, height: 14 }} />} label="Analytics" />
            </div>
          )}

          {/* Intelligence section */}
          <button
            onClick={() => setIntelligenceOpen(v => !v)}
            className="sidebar-section-toggle"
            style={{ width: '100%', marginTop: 8, marginBottom: 4 }}
          >
            {intelligenceOpen
              ? <ChevronDown style={{ width: 10, height: 10 }} />
              : <ChevronRight style={{ width: 10, height: 10 }} />
            }
            Intelligence
          </button>

          {intelligenceOpen && (
            <div>
              <NavItem to="/prompts" icon={<FileText style={{ width: 14, height: 14 }} />} label="Prompt Registry" />
              <NavItem to="/flow" icon={<GitGraph style={{ width: 14, height: 14 }} />} label="Flow Graph" />
              <NavItem to="/arena" icon={<Swords style={{ width: 14, height: 14 }} />} label="Model Arena" />
            </div>
          )}
        </nav>

        {/* Connection status */}
        <div style={{ padding: '10px 8px', borderTop: '1px solid var(--border-faint)' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '7px 10px',
            borderRadius: 7,
            background: 'var(--bg-raised)',
            border: '1px solid var(--border-faint)',
          }}>
            <div style={{ position: 'relative', flexShrink: 0 }}>
              <div style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: isConnected ? 'var(--green)' : 'var(--red)',
              }} />
            </div>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1 }}>
              {isConnected ? 'Connected' : 'Offline'}
            </span>
            <Activity style={{
              width: 12,
              height: 12,
              color: isConnected ? 'var(--green)' : 'var(--red)',
            }} />
          </div>
        </div>
      </aside>

      {/* Main */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'hidden' }}>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          <Routes>
            <Route path="/" element={<RunsPage />} />
            <Route path="/runs/:runId" element={<RunDetailPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
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
    <NavLink
      to={to}
      end={to === '/'}
      className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
    >
      {icon}
      <span>{label}</span>
    </NavLink>
  )
}

export default App
