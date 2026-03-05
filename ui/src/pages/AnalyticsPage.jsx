import React, { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import {
  Activity, AlertCircle, Coins, Clock, ChevronDown, Layers, Zap
} from 'lucide-react'
import { API_URL } from '../App'

const THEME = {
  primary: '#E8501A',
  secondary: '#F97316',
  danger: '#D93025',
  success: '#1A8A5A',
  grid: '#EBEBEA',
  text: '#1A1917',
  textDim: '#9B9997',
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [timeRange, setTimeRange] = useState(24)
  const [services, setServices] = useState([])
  const [selectedService, setSelectedService] = useState('')

  useEffect(() => {
    fetch(`${API_URL}/v1/services`)
      .then(res => res.json())
      .then(data => setServices(data || []))
      .catch(err => console.error('Failed to load services', err))
  }, [])

  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true)
      setError('')
      try {
        let url = `${API_URL}/v1/analytics/overview?hours=${timeRange}`
        if (selectedService) url += `&service_name=${encodeURIComponent(selectedService)}`
        const res = await fetch(url)
        if (!res.ok) throw new Error('Failed to load analytics')
        const data = await res.json()
        if (data?.error) throw new Error(data.error)
        setAnalytics(data)
      } catch (err) {
        setAnalytics(null)
        setError(err.message || 'Failed to load analytics')
      } finally {
        setLoading(false)
      }
    }
    fetchAnalytics()
    const interval = setInterval(fetchAnalytics, 30000)
    return () => clearInterval(interval)
  }, [timeRange, selectedService])

  const chartData = analytics?.timeseries || []
  const hasData = chartData.length > 0
  const totalCost = analytics?.overview?.total_cost || 0
  const totalRuns = analytics?.overview?.run_count || analytics?.overview?.trace_count || 0
  const avgLatency = analytics?.overview?.avg_duration_ms || 0
  const errorRate = analytics?.overview?.error_rate || 0

  return (
    <div style={{ minHeight: '100%', background: 'var(--bg-primary)' }}>

      {/* Page header */}
      <div className="page-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Activity style={{ width: 14, height: 14, color: 'var(--accent)' }} />
            <span className="page-title">Analytics</span>
          </div>
          <div className="page-subtitle">System performance overview</div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Service filter */}
          <div style={{ position: 'relative' }}>
            <select
              value={selectedService}
              onChange={(e) => setSelectedService(e.target.value)}
              className="select-premium"
              style={{ minWidth: 140 }}
            >
              <option value="">All Services</option>
              {services.map(s => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* Time range */}
          <div style={{
            display: 'flex',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-default)',
            borderRadius: 7,
            padding: 2,
            gap: 2,
          }}>
            {[1, 24, 168].map((h) => (
              <button
                key={h}
                onClick={() => setTimeRange(h)}
                style={{
                  padding: '3px 10px',
                  borderRadius: 5,
                  fontSize: 12,
                  fontWeight: 500,
                  cursor: 'pointer',
                  border: 'none',
                  transition: 'all 0.12s',
                  background: timeRange === h ? 'var(--bg-raised)' : 'transparent',
                  color: timeRange === h ? 'var(--text-primary)' : 'var(--text-tertiary)',
                  boxShadow: timeRange === h ? 'var(--shadow-sm)' : 'none',
                }}
              >
                {h === 1 ? '1H' : h === 24 ? '24H' : '7D'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 14 }}>

        {error && (
          <div style={{
            padding: '8px 12px',
            background: 'var(--red-bg)',
            border: '1px solid rgba(217,48,37,0.15)',
            borderRadius: 8,
            fontSize: 12,
            color: 'var(--red)',
          }}>
            {error}
          </div>
        )}

        {/* KPI Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
          <StatCard
            label="Total Cost"
            value={`$${totalCost.toFixed(4)}`}
            icon={<Coins style={{ width: 14, height: 14, color: 'var(--green)' }} />}
            sub="Estimated spend"
          />
          <StatCard
            label="Run Volume"
            value={totalRuns.toLocaleString()}
            icon={<Layers style={{ width: 14, height: 14, color: '#2563EB' }} />}
            sub={`${analytics?.overview?.span_count?.toLocaleString() || 0} spans`}
          />
          <StatCard
            label="Avg Latency"
            value={`${avgLatency}ms`}
            icon={<Clock style={{ width: 14, height: 14, color: '#7C3AED' }} />}
            sub={`P95: ${analytics?.overview?.p95_duration_ms || 0}ms`}
          />
          <StatCard
            label="Error Rate"
            value={`${errorRate}%`}
            icon={<AlertCircle style={{ width: 14, height: 14, color: errorRate > 0 ? 'var(--red)' : 'var(--text-tertiary)' }} />}
            sub={`${analytics?.overview?.error_count || 0} errors`}
            alert={errorRate > 5}
          />
        </div>

        {/* Charts */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 10 }}>

          {/* Traffic chart */}
          <div className="glass-card" style={{ padding: '16px 18px', height: 320 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Traffic & Health</div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>Requests vs. Errors over time</div>
              </div>
              <div style={{ display: 'flex', gap: 14 }}>
                <LegendDot color={THEME.primary} label="Requests" />
                <LegendDot color={THEME.danger} label="Errors" />
              </div>
            </div>
            <div style={{ height: 240 }}>
              {hasData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gSpans" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.primary} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={THEME.primary} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="gErrors" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.danger} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={THEME.danger} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={THEME.grid} strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="hour"
                      stroke={THEME.textDim}
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(str) => {
                        const d = new Date(str)
                        return timeRange > 24
                          ? d.toLocaleDateString([], { month: 'short', day: 'numeric' })
                          : d.getHours() + ':00'
                      }}
                      dy={8}
                    />
                    <YAxis stroke={THEME.textDim} fontSize={10} tickLine={false} axisLine={false} />
                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border-default)', strokeWidth: 1 }} />
                    <Area type="monotone" dataKey="spans" stroke={THEME.primary} fill="url(#gSpans)" strokeWidth={1.5} dot={false} activeDot={{ r: 3, fill: THEME.primary }} />
                    <Area type="monotone" dataKey="errors" stroke={THEME.danger} fill="url(#gErrors)" strokeWidth={1.5} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart label="No traffic data available" />
              )}
            </div>
          </div>

          {/* Token usage */}
          <div className="glass-card" style={{ padding: '16px 18px', height: 320 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>Token Usage</div>
                <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>Consumption volume</div>
              </div>
              <LegendDot color={THEME.secondary} label="Tokens" />
            </div>
            <div style={{ height: 240 }}>
              {hasData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -22, bottom: 0 }}>
                    <defs>
                      <linearGradient id="gTokens" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.secondary} stopOpacity={0.2} />
                        <stop offset="95%" stopColor={THEME.secondary} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={THEME.grid} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="hour" hide />
                    <YAxis stroke={THEME.textDim} fontSize={10} tickLine={false} axisLine={false} tickFormatter={v => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v} />
                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'var(--border-default)', strokeWidth: 1 }} />
                    <Area type="monotone" dataKey="tokens" stroke={THEME.secondary} fill="url(#gTokens)" strokeWidth={1.5} dot={false} activeDot={{ r: 3 }} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart label="No token data" />
              )}
            </div>
          </div>
        </div>

        {/* Tables */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>

          <div className="glass-card" style={{ overflow: 'hidden' }}>
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border-faint)',
              background: 'var(--bg-raised)',
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--text-primary)',
            }}>
              Model Performance
            </div>
            <table className="table-premium">
              <thead>
                <tr>
                  <th>Model</th>
                  <th style={{ textAlign: 'right' }}>Calls</th>
                  <th style={{ textAlign: 'right' }}>Cost</th>
                  <th style={{ textAlign: 'right' }}>Latency</th>
                </tr>
              </thead>
              <tbody>
                {analytics?.by_model?.map((m, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 500 }}>{m.model}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{m.count.toLocaleString()}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace' }}>${m.cost?.toFixed(4)}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{m.avg_duration_ms}ms</td>
                  </tr>
                ))}
                {!analytics?.by_model?.length && (
                  <tr><td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-tertiary)', padding: '24px 0' }}>No model data recorded</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div className="glass-card" style={{ overflow: 'hidden' }}>
            <div style={{
              padding: '12px 16px',
              borderBottom: '1px solid var(--border-faint)',
              background: 'var(--bg-raised)',
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--text-primary)',
            }}>
              Operation Types
            </div>
            <table className="table-premium">
              <thead>
                <tr>
                  <th>Type</th>
                  <th style={{ textAlign: 'right' }}>Count</th>
                  <th style={{ textAlign: 'right' }}>Avg Latency</th>
                  <th style={{ textAlign: 'right' }}>Errors</th>
                </tr>
              </thead>
              <tbody>
                {analytics?.by_type?.map((t, i) => (
                  <tr key={i}>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <div className={`status-dot ${t.errors > 0 ? 'status-dot-red' : 'status-dot-green'}`} />
                        <span style={{ fontWeight: 500, textTransform: 'capitalize' }}>{t.span_type}</span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{t.count.toLocaleString()}</td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{t.avg_duration_ms}ms</td>
                    <td style={{ textAlign: 'right', fontFamily: 'monospace', color: t.errors > 0 ? 'var(--red)' : 'var(--text-tertiary)' }}>
                      {t.errors || '-'}
                    </td>
                  </tr>
                ))}
                {!analytics?.by_type?.length && (
                  <tr><td colSpan={4} style={{ textAlign: 'center', color: 'var(--text-tertiary)', padding: '24px 0' }}>No operation data recorded</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, sub, icon, alert }) {
  return (
    <div className="metric-card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          {label}
        </span>
        {icon}
      </div>
      <div style={{
        fontSize: 22,
        fontWeight: 700,
        color: alert ? 'var(--red)' : 'var(--text-primary)',
        fontFamily: 'monospace',
        lineHeight: 1.1,
      }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>{sub}</div>
    </div>
  )
}

function LegendDot({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--text-secondary)' }}>
      <div style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
      {label}
    </div>
  )
}

function EmptyChart({ label }) {
  return (
    <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>{label}</span>
    </div>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border-default)',
      borderRadius: 8,
      padding: '8px 10px',
      fontSize: 11,
      boxShadow: 'var(--shadow-lg)',
    }}>
      <div style={{ color: 'var(--text-tertiary)', marginBottom: 5, paddingBottom: 5, borderBottom: '1px solid var(--border-faint)' }}>
        {new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', gap: 20, marginBottom: 2 }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-secondary)' }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: p.stroke || p.fill, display: 'inline-block' }} />
            {p.name}
          </span>
          <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>
            {p.value.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  )
}
