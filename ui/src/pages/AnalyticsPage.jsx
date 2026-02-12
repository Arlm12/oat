import React, { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'
import {
  Activity, AlertCircle, Coins, Clock, ChevronDown, Layers, Zap
} from 'lucide-react'
import { API_URL } from '../App'

// --- THEME CONFIGURATION ---
const THEME = {
  // Base colors
  primary: '#3b82f6',    // Blue
  secondary: '#8b5cf6',  // Violet
  danger: '#f43f5e',     // Rose
  success: '#10b981',    // Emerald
  text: '#e4e4e7',       // Zinc 200
  textDim: '#71717a',    // Zinc 500
  grid: '#27272a',       // Zinc 800
}

export default function AnalyticsPage() {
  // --- STATE ---
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [timeRange, setTimeRange] = useState(24)
  const [services, setServices] = useState([])
  const [selectedService, setSelectedService] = useState('')

  // --- 1. FETCH SERVICES ---
  useEffect(() => {
    fetch(`${API_URL}/services`)
      .then(res => res.json())
      .then(data => setServices(data || []))
      .catch(err => console.error("Failed to load services", err))
  }, [])

  // --- 2. FETCH ANALYTICS (With Mock Fallback for Design Validation) ---
  useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true)
      try {
        let url = `${API_URL}/analytics/overview?hours=${timeRange}&_t=${Date.now()}`
        if (selectedService) url += `&service_name=${selectedService}`

        const res = await fetch(url)
        if (!res.ok) throw new Error("API not reachable")
        const data = await res.json()
        setAnalytics(data)
      } catch (err) {
        console.warn('API failed or empty, using premium mock data for design view')
        setAnalytics(generateMockData(timeRange))
      } finally {
        setLoading(false)
      }
    }

    fetchAnalytics()
    // Refresh every 30s
    const interval = setInterval(fetchAnalytics, 30000)
    return () => clearInterval(interval)
  }, [timeRange, selectedService])

  // --- PREPARE DATA ---
  const chartData = analytics?.timeseries || []
  const hasData = chartData.length > 0

  // KPI Shortcuts
  const totalCost = analytics?.overview?.total_cost || 0
  const totalTraces = analytics?.overview?.trace_count || 0
  const avgLatency = analytics?.overview?.avg_duration_ms || 0
  const errorRate = analytics?.overview?.error_rate || 0

  return (
    <div className="min-h-full bg-[#050505] text-zinc-200 font-sans selection:bg-blue-500/20 pb-10">

      {/* HEADER with Glass Effect */}
      <header className="sticky top-0 z-30 border-b border-white/5 bg-[#050505]/70 backdrop-blur-xl">
        <div className="max-w-[1600px] mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
                <Zap className="w-4 h-4 text-white fill-white" />
              </div>
              <span className="font-bold text-white tracking-tight">AgentTracer</span>
            </div>

            <div className="h-6 w-px bg-white/10" />

            {/* Service Selector (Fixed Styling) */}
            <div className="relative group">
              <select
                value={selectedService}
                onChange={(e) => setSelectedService(e.target.value)}
                className="bg-transparent text-sm text-zinc-400 font-medium hover:text-white transition-colors appearance-none pr-8 outline-none cursor-pointer py-1"
              >
                <option value="" className="bg-[#09090b]">All Services</option>
                {services.map(s => (
                  <option key={s} value={s} className="bg-[#09090b]">{s}</option>
                ))}
              </select>
              <ChevronDown className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 text-zinc-500 pointer-events-none" />
            </div>
          </div>

          {/* Time Controls */}
          <div className="flex bg-zinc-900/50 rounded-lg p-0.5 border border-white/5 backdrop-blur-md">
            {[1, 24, 168].map((h) => (
              <button
                key={h}
                onClick={() => setTimeRange(h)}
                className={`px-3 py-1.5 text-[11px] font-medium rounded-md transition-all duration-300 ${timeRange === h
                    ? 'bg-zinc-800 text-white shadow-sm ring-1 ring-white/10'
                    : 'text-zinc-500 hover:text-zinc-300'
                  }`}
              >
                {h === 1 ? '1H' : h === 24 ? '24H' : '7D'}
              </button>
            ))}
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-6 py-8 space-y-6">

        {/* KPI GRID */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Total Cost"
            value={`$${totalCost.toFixed(4)}`}
            icon={<Coins className="w-4 h-4 text-emerald-400" />}
            subtext="Estimated spend"
          />
          <StatCard
            label="Trace Volume"
            value={totalTraces.toLocaleString()}
            icon={<Layers className="w-4 h-4 text-blue-400" />}
            subtext={`${analytics?.overview?.span_count?.toLocaleString() || 0} spans`}
          />
          <StatCard
            label="Avg Latency"
            value={`${avgLatency}ms`}
            icon={<Clock className="w-4 h-4 text-violet-400" />}
            subtext={`P95: ${analytics?.overview?.p95_duration_ms || 0}ms`}
          />
          <StatCard
            label="Error Rate"
            value={`${errorRate}%`}
            icon={<AlertCircle className={`w-4 h-4 ${errorRate > 0 ? 'text-rose-400' : 'text-zinc-500'}`} />}
            subtext={`${analytics?.overview?.error_count || 0} errors`}
            alert={errorRate > 5}
          />
        </div>

        {/* MAIN CHART SECTION */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Traffic Chart (2/3 width) */}
          <GlassCard className="lg:col-span-2 h-[420px] flex flex-col p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-sm font-semibold text-white tracking-wide">Traffic & Health</h3>
                <p className="text-xs text-zinc-500 mt-1 font-medium">Requests vs. Errors over time</p>
              </div>
              <div className="flex gap-6">
                <LegendItem color={THEME.primary} label="Requests" />
                <LegendItem color={THEME.danger} label="Errors" />
              </div>
            </div>

            <div className="flex-1 min-h-0 w-full">
              {hasData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorSpans" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.primary} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={THEME.primary} stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="colorErrors" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.danger} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={THEME.danger} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={THEME.grid} strokeDasharray="3 3" vertical={false} strokeOpacity={0.4} />
                    <XAxis
                      dataKey="hour"
                      stroke={THEME.textDim}
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(str) => new Date(str).getHours() + ':00'}
                      dy={10}
                    />
                    <YAxis
                      stroke={THEME.textDim}
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#27272a', strokeWidth: 1 }} />
                    <Area
                      type="monotone"
                      dataKey="spans"
                      stroke={THEME.primary}
                      fill="url(#colorSpans)"
                      strokeWidth={2}
                      activeDot={{ r: 4, fill: '#fff', strokeWidth: 0 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="errors"
                      stroke={THEME.danger}
                      fill="url(#colorErrors)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState label="No traffic data available" />
              )}
            </div>
          </GlassCard>

          {/* Token Usage (1/3 width) - CHANGED TO LINE GRAPH */}
          <GlassCard className="h-[420px] flex flex-col p-6">
            <div className="mb-6 flex justify-between items-start">
              <div>
                <h3 className="text-sm font-semibold text-white tracking-wide">Token Usage</h3>
                <p className="text-xs text-zinc-500 mt-1 font-medium">Consumption volume</p>
              </div>
              <LegendItem color={THEME.secondary} label="Tokens" />
            </div>

            <div className="flex-1 min-h-0 w-full">
              {hasData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 0, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorTokens" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={THEME.secondary} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={THEME.secondary} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={THEME.grid} strokeDasharray="3 3" vertical={false} strokeOpacity={0.4} />
                    <XAxis dataKey="hour" hide />
                    <YAxis
                      stroke={THEME.textDim}
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(val) => val >= 1000 ? `${(val / 1000).toFixed(0)}k` : val}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#27272a', strokeWidth: 1 }} />
                    {/* Changed from Bar to Area for smooth line look */}
                    <Area
                      type="monotone"
                      dataKey="tokens"
                      stroke={THEME.secondary}
                      fill="url(#colorTokens)"
                      strokeWidth={2}
                      activeDot={{ r: 4, fill: '#fff', strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState label="No token data" />
              )}
            </div>
          </GlassCard>
        </div>

        {/* TABLES SECTION */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

          {/* Model Performance Table */}
          <GlassCard className="flex flex-col overflow-hidden">
            <div className="p-5 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
              <h3 className="text-sm font-medium text-white">Model Performance</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/5 text-zinc-500">
                    <th className="px-5 py-3 font-medium">Model</th>
                    <th className="px-5 py-3 font-medium text-right">Calls</th>
                    <th className="px-5 py-3 font-medium text-right">Cost</th>
                    <th className="px-5 py-3 font-medium text-right">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {analytics?.by_model?.map((m, i) => (
                    <tr key={i} className="hover:bg-white/[0.04] transition-colors">
                      <td className="px-5 py-3 text-zinc-300 font-medium">{m.model}</td>
                      <td className="px-5 py-3 text-right text-zinc-500 font-mono">{m.count.toLocaleString()}</td>
                      <td className="px-5 py-3 text-right text-zinc-300 font-mono">${m.cost?.toFixed(4)}</td>
                      <td className="px-5 py-3 text-right text-zinc-500 font-mono">{m.avg_duration_ms}ms</td>
                    </tr>
                  ))}
                  {!analytics?.by_model?.length && (
                    <tr><td colSpan={4} className="p-8 text-center text-zinc-600">No model data recorded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </GlassCard>

          {/* Span Types Table */}
          <GlassCard className="flex flex-col overflow-hidden">
            <div className="p-5 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
              <h3 className="text-sm font-medium text-white">Operation Types</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/5 text-zinc-500">
                    <th className="px-5 py-3 font-medium">Type</th>
                    <th className="px-5 py-3 font-medium text-right">Count</th>
                    <th className="px-5 py-3 font-medium text-right">Avg Latency</th>
                    <th className="px-5 py-3 font-medium text-right">Errors</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {analytics?.by_type?.map((t, i) => (
                    <tr key={i} className="hover:bg-white/[0.04] transition-colors">
                      <td className="px-5 py-3 text-zinc-300 font-medium capitalize flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${t.errors > 0 ? 'bg-rose-500' : 'bg-emerald-500'}`} />
                        {t.span_type}
                      </td>
                      <td className="px-5 py-3 text-right text-zinc-500 font-mono">{t.count.toLocaleString()}</td>
                      <td className="px-5 py-3 text-right text-zinc-500 font-mono">{t.avg_duration_ms}ms</td>
                      <td className={`px-5 py-3 text-right font-mono ${t.errors > 0 ? 'text-rose-400' : 'text-zinc-600'}`}>
                        {t.errors || '-'}
                      </td>
                    </tr>
                  ))}
                  {!analytics?.by_type?.length && (
                    <tr><td colSpan={4} className="p-8 text-center text-zinc-600">No operation data recorded</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </GlassCard>
        </div>
      </main>
    </div>
  )
}

// --- SUB COMPONENTS ---

// 1. FROSTED GLASS CARD COMPONENT
function GlassCard({ children, className = '' }) {
  return (
    <div className={`bg-zinc-900/50 backdrop-blur-xl border border-white/5 rounded-2xl shadow-xl shadow-black/20 ${className}`}>
      {children}
    </div>
  )
}

function StatCard({ label, value, icon, subtext, alert }) {
  return (
    <GlassCard className="p-6 group hover:bg-zinc-800/50 transition-all duration-300">
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">{label}</span>
        <div className="opacity-80 group-hover:opacity-100 group-hover:scale-110 transition-all">
          {icon}
        </div>
      </div>
      <div className="space-y-1">
        <div className={`text-2xl font-bold tracking-tight tabular-nums ${alert ? 'text-rose-400' : 'text-white'}`}>
          {value}
        </div>
        <div className="text-xs text-zinc-500 font-medium">{subtext}</div>
      </div>
    </GlassCard>
  )
}

function LegendItem({ color, label }) {
  return (
    <div className="flex items-center gap-2 text-xs font-medium text-zinc-400">
      <span className="w-2 h-2 rounded-full shadow-[0_0_8px_currentColor]" style={{ backgroundColor: color, color: color }} />
      {label}
    </div>
  )
}

function EmptyState({ label }) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center text-zinc-600 gap-3">
      <Activity className="w-6 h-6 opacity-20" />
      <span className="text-xs font-medium">{label}</span>
    </div>
  )
}

function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="bg-[#09090b]/90 border border-white/10 backdrop-blur-md p-3 rounded-xl shadow-2xl text-xs z-50">
        <p className="text-zinc-500 mb-2 font-medium border-b border-white/5 pb-2">
          {new Date(label).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
        {payload.map((p, i) => (
          <div key={i} className="flex items-center justify-between gap-6 mb-1">
            <span className="text-zinc-400 capitalize flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: p.stroke || p.fill }} />
              {p.name}
            </span>
            <span className="font-mono text-white font-semibold">
              {p.value.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    )
  }
  return null
}

// --- MOCK DATA GENERATOR (For instant beautiful preview) ---
function generateMockData(hours) {
  const points = hours === 1 ? 60 : hours; // minutes vs hours
  const now = new Date();

  const timeseries = Array.from({ length: points }, (_, i) => {
    const time = new Date(now);
    if (hours === 1) time.setMinutes(time.getMinutes() - (points - i));
    else time.setHours(time.getHours() - (points - i));

    // Smooth random curve
    const x = i / points * 10;
    const baseVal = Math.sin(x) * 20 + 50;
    const spans = Math.floor(Math.max(10, baseVal + Math.random() * 20));

    return {
      hour: time.toISOString(),
      spans: spans,
      errors: Math.random() > 0.8 ? Math.floor(Math.random() * 5) : 0,
      tokens: spans * (100 + Math.random() * 50),
      cost: spans * 0.001
    };
  });

  return {
    overview: {
      total_cost: 45.20,
      trace_count: 15420,
      span_count: 42100,
      avg_duration_ms: 320,
      p95_duration_ms: 850,
      error_rate: 1.2,
      error_count: 185
    },
    timeseries,
    by_model: [
      { model: "gpt-4-turbo", count: 8500, cost: 35.50, avg_duration_ms: 900 },
      { model: "gpt-3.5-turbo", count: 6200, cost: 8.20, avg_duration_ms: 180 },
      { model: "claude-3-opus", count: 720, cost: 1.50, avg_duration_ms: 1400 },
    ],
    by_type: [
      { span_type: "llm", count: 12000, avg_duration_ms: 800, errors: 45 },
      { span_type: "tool", count: 3000, avg_duration_ms: 250, errors: 12 },
      { span_type: "chain", count: 420, avg_duration_ms: 50, errors: 0 },
    ]
  }
}