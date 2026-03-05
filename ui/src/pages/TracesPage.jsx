import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Clock,
  AlertCircle,
  CheckCircle,
  Loader2,
  Bot,
  ChevronRight,
  ArrowLeft,
  LayoutGrid,
  Hash,
  Timer,
  Layers,
  RefreshCw
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { API_URL } from '../App'

export default function RunsPage() {
  const [view, setView] = useState('agents')
  const [selectedService, setSelectedService] = useState(null)
  const [services, setServices] = useState([])
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => { fetchServices() }, [])

  const fetchServices = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/v1/services`)
      const data = await res.json()
      setServices(data || [])
    } catch (err) {
      console.error('Failed to fetch services:', err)
    } finally {
      setLoading(false)
    }
  }

  const selectAgent = async (agentName) => {
    setSelectedService(agentName)
    setView('runs')
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/v1/runs?limit=100&service_name=${agentName}`)
      const data = await res.json()
      setRuns(data.items || [])
    } catch (err) {
      console.error('Failed to fetch runs:', err)
    } finally {
      setLoading(false)
    }
  }

  const goBackToAgents = () => {
    setView('agents')
    setSelectedService(null)
    setRuns([])
    fetchServices()
  }

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>

      {/* Page Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {view === 'runs' && (
            <button
              onClick={goBackToAgents}
              className="btn btn-ghost"
              style={{ padding: '0 8px', height: 28 }}
            >
              <ArrowLeft style={{ width: 14, height: 14 }} />
            </button>
          )}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              {view === 'agents'
                ? <LayoutGrid style={{ width: 14, height: 14, color: 'var(--accent)' }} />
                : <Bot style={{ width: 14, height: 14, color: 'var(--accent)' }} />
              }
              <span className="page-title">
                {view === 'agents' ? 'Agents' : selectedService}
              </span>
            </div>
            <div className="page-subtitle">
              {view === 'agents'
                ? 'Select an agent to inspect its execution history'
                : `${runs.length} run${runs.length !== 1 ? 's' : ''} recorded`}
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 200 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <Loader2 style={{ width: 20, height: 20, color: 'var(--accent)' }} className="animate-spin" />
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Loading...</span>
            </div>
          </div>
        ) : view === 'agents' ? (

          /* Agent Grid */
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
            gap: 10,
          }} className="stagger">
            {services.length === 0 ? (
              <div style={{
                gridColumn: '1 / -1',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                padding: '60px 0',
                gap: 10,
              }} className="animate-fadeIn">
                <div style={{
                  width: 44,
                  height: 44,
                  borderRadius: 10,
                  background: 'var(--bg-sunken)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}>
                  <Bot style={{ width: 20, height: 20, color: 'var(--text-disabled)' }} />
                </div>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>
                  No agents detected
                </span>
                <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>
                  Run your agent code to start tracing
                </span>
              </div>
            ) : (
              services.map((agent) => (
                <button
                  key={agent}
                  onClick={() => selectAgent(agent)}
                  className="animate-fadeInUp"
                  style={{
                    opacity: 0,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'flex-start',
                    padding: '14px 16px',
                    background: 'var(--bg-surface)',
                    border: '1px solid var(--border-subtle)',
                    borderRadius: 10,
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'border-color 0.15s, box-shadow 0.15s',
                    boxShadow: 'var(--shadow-sm)',
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = 'var(--border-strong)'
                    e.currentTarget.style.boxShadow = 'var(--shadow-md)'
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = 'var(--border-subtle)'
                    e.currentTarget.style.boxShadow = 'var(--shadow-sm)'
                  }}
                >
                  <div style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    background: 'var(--accent-muted)',
                    border: '1px solid var(--accent-ring)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: 10,
                  }}>
                    <Bot style={{ width: 14, height: 14, color: 'var(--accent)' }} />
                  </div>
                  <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', marginBottom: 3 }}>
                    {agent}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                    View runs
                  </span>
                </button>
              ))
            )}
          </div>

        ) : (

          /* Run List */
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }} className="stagger">
            {runs.length === 0 ? (
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                padding: '60px 0',
                gap: 10,
              }} className="animate-fadeIn">
                <Layers style={{ width: 24, height: 24, color: 'var(--text-disabled)' }} />
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                  No runs found for {selectedService}
                </span>
              </div>
            ) : (
              runs.map((run) => <RunRow key={run.run_id} run={run} />)
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function RunRow({ run }) {
  const isError = run.status === 'error'
  const isRunning = run.status === 'running'

  const statusBg = isError ? 'var(--red-bg)' : isRunning ? 'rgba(37,99,235,0.08)' : 'var(--green-bg)'
  const statusBorder = isError ? 'rgba(217,48,37,0.15)' : isRunning ? 'rgba(37,99,235,0.2)' : 'rgba(26,138,90,0.15)'
  const statusIcon = isError
    ? <AlertCircle style={{ width: 13, height: 13, color: 'var(--red)' }} />
    : isRunning
      ? <RefreshCw style={{ width: 13, height: 13, color: '#2563EB' }} className="animate-spin" />
      : <CheckCircle style={{ width: 13, height: 13, color: 'var(--green)' }} />

  return (
    <Link
      to={`/runs/${run.run_id}`}
      className="animate-fadeInUp"
      style={{
        opacity: 0,
        display: 'block',
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-faint)',
        borderRadius: 8,
        padding: '10px 14px',
        textDecoration: 'none',
        transition: 'border-color 0.12s, box-shadow 0.12s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = 'var(--border-default)'
        e.currentTarget.style.boxShadow = 'var(--shadow-sm)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = 'var(--border-faint)'
        e.currentTarget.style.boxShadow = 'none'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>

        {/* Status */}
        <div style={{
          width: 28,
          height: 28,
          borderRadius: 7,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexShrink: 0,
          background: statusBg,
          border: `1px solid ${statusBorder}`,
        }}>
          {statusIcon}
        </div>

        {/* Name + ID */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 13,
            fontWeight: 500,
            color: 'var(--text-primary)',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}>
            {run.input_summary || run.service_name || 'Unnamed Run'}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginTop: 2 }}>
            <Hash style={{ width: 10, height: 10, color: 'var(--text-tertiary)' }} />
            <span style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
              {run.run_id.slice(0, 12)}
            </span>
          </div>
        </div>

        {/* Time */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-secondary)', flexShrink: 0 }}>
          <Clock style={{ width: 11, height: 11 }} />
          <span style={{ fontSize: 12 }}>
            {run.started_at
              ? formatDistanceToNow(new Date(run.started_at * 1000), { addSuffix: true })
              : 'Unknown'}
          </span>
        </div>

        {/* Duration */}
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, justifyContent: 'flex-end' }}>
            <Timer style={{ width: 11, height: 11, color: 'var(--text-tertiary)' }} />
            <span style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-primary)' }}>
              {Math.round(run.duration_ms || 0)}ms
            </span>
          </div>
          {run.span_count && (
            <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>
              {run.span_count} spans
            </div>
          )}
        </div>

        <ChevronRight style={{ width: 14, height: 14, color: 'var(--text-disabled)', flexShrink: 0 }} />
      </div>
    </Link>
  )
}
