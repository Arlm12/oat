import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  Clock,
  AlertCircle,
  CheckCircle,
  Loader2,
  Trash2,
  Cpu,
  Bot,
  ChevronRight,
  ArrowLeft,
  LayoutGrid,
  List,
  Hash,
  Timer,
  Layers
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { API_URL } from '../App'

export default function TracesPage() {
  // State for Hierarchy
  const [view, setView] = useState('agents') // 'agents' | 'traces'
  const [selectedService, setSelectedService] = useState(null)

  // Data State
  const [services, setServices] = useState([])
  const [traces, setTraces] = useState([])
  const [loading, setLoading] = useState(true)

  // 1. Fetch Agents on Mount
  useEffect(() => {
    fetchServices()
  }, [])

  const fetchServices = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${API_URL}/services`)
      const data = await res.json()
      setServices(data || [])
    } catch (err) {
      console.error('Failed to fetch services:', err)
    } finally {
      setLoading(false)
    }
  }

  // 2. Fetch Traces when Agent is Selected
  const selectAgent = async (agentName) => {
    setSelectedService(agentName)
    setView('traces')
    setLoading(true)
    try {
      // Filter traces by the selected service name
      const res = await fetch(`${API_URL}/traces?limit=100&service_name=${agentName}`)
      const data = await res.json()
      setTraces(data)
    } catch (err) {
      console.error('Failed to fetch traces:', err)
    } finally {
      setLoading(false)
    }
  }

  const goBackToAgents = () => {
    setView('agents')
    setSelectedService(null)
    setTraces([])
    fetchServices()
  }

  return (
    <div className="h-full flex flex-col bg-[var(--bg-primary)]">

      {/* Header Area */}
      <div className="p-6 pb-0">
        <div className="flex items-center gap-4 mb-1">
          {view === 'traces' && (
            <button
              onClick={goBackToAgents}
              className="p-2 rounded-xl transition-all duration-200 text-[#71717A] hover:text-white hover:bg-[var(--bg-elevated)]"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
          )}

          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              {view === 'agents' ? (
                <>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1))' }}
                  >
                    <LayoutGrid className="w-4 h-4 text-blue-400" />
                  </div>
                  <span className="gradient-text">Agents</span>
                </>
              ) : (
                <>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                    style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1))' }}
                  >
                    <Bot className="w-4 h-4 text-blue-400" />
                  </div>
                  {selectedService}
                </>
              )}
            </h1>
            <p className="text-sm text-[#52525B] mt-1 ml-11">
              {view === 'agents'
                ? "Select an agent to inspect its execution history"
                : `${traces.length} traces recorded`}
            </p>
          </div>
        </div>
      </div>

      {/* Divider with gradient */}
      <div className="mx-6 mt-4 mb-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, var(--border-subtle), transparent)' }}
      />

      {/* Main Content Area */}
      <div className="flex-1 overflow-auto p-6">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
              <span className="text-sm text-[#52525B]">Loading...</span>
            </div>
          </div>
        ) : view === 'agents' ? (

          /* === VIEW 1: AGENT GRID === */
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 stagger">
            {services.length === 0 ? (
              <div className="col-span-full flex flex-col items-center justify-center py-24 animate-fadeIn">
                <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                  style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(139, 92, 246, 0.05))' }}
                >
                  <Bot className="w-8 h-8 text-[#27272A]" />
                </div>
                <p className="text-[#52525B] font-medium">No agents detected yet</p>
                <p className="text-sm text-[#3F3F46] mt-1">Run your agent code to generate traces</p>
              </div>
            ) : (
              services.map((agent) => (
                <button
                  key={agent}
                  onClick={() => selectAgent(agent)}
                  className="group flex flex-col items-start p-6 glass-card text-left animate-fadeInUp"
                  style={{ opacity: 0 }}
                >
                  <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-4 transition-all duration-300 group-hover:scale-110"
                    style={{
                      background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(139, 92, 246, 0.08))',
                      boxShadow: '0 0 0 1px rgba(59, 130, 246, 0.1)',
                    }}
                  >
                    <Bot className="w-5 h-5 text-blue-400 group-hover:text-blue-300 transition-colors" />
                  </div>
                  <h3 className="text-[15px] font-semibold text-white mb-1 group-hover:text-blue-400 transition-colors duration-200">
                    {agent}
                  </h3>
                  <p className="text-xs text-[#52525B] group-hover:text-[#71717A] transition-colors">
                    Click to inspect traces →
                  </p>
                </button>
              ))
            )}
          </div>

        ) : (

          /* === VIEW 2: TRACE LIST === */
          <div className="space-y-2 stagger">
            {traces.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 animate-fadeIn">
                <Layers className="w-10 h-10 text-[#27272A] mb-4" />
                <p className="text-[#52525B] font-medium">No traces found for {selectedService}</p>
              </div>
            ) : (
              traces.map((trace) => (
                <TraceRow key={trace.trace_id} trace={trace} />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function TraceRow({ trace }) {
  const isError = trace.error_count > 0 || trace.status === 'error'

  return (
    <Link
      to={`/trace/${trace.trace_id}`}
      className="group block glass-card p-4 animate-fadeInUp"
      style={{ opacity: 0 }}
    >
      <div className="grid grid-cols-12 items-center gap-4">

        {/* Status Icon */}
        <div className="col-span-1 flex justify-center">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isError
              ? 'bg-rose-500/10 border border-rose-500/20'
              : 'bg-emerald-500/10 border border-emerald-500/20'
            }`}>
            {isError ? (
              <AlertCircle className="w-4 h-4 text-rose-400" />
            ) : (
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            )}
          </div>
        </div>

        {/* Trace Name & ID */}
        <div className="col-span-4">
          <div className="font-medium text-[15px] text-white truncate pr-4 group-hover:text-blue-400 transition-colors duration-200">
            {trace.name || 'Unnamed Trace'}
          </div>
          <div className="flex items-center gap-1.5 mt-1 text-xs text-[#3F3F46] font-mono">
            <Hash className="w-3 h-3" />
            {trace.trace_id.slice(0, 12)}
          </div>
        </div>

        {/* Timestamp */}
        <div className="col-span-3 flex items-center gap-2 text-sm text-[#71717A]">
          <Clock className="w-3.5 h-3.5" />
          {trace.start_time
            ? formatDistanceToNow(new Date(trace.start_time * 1000), { addSuffix: true })
            : 'Unknown time'}
        </div>

        {/* Metrics */}
        <div className="col-span-3 text-right">
          <div className="flex items-center justify-end gap-1.5 text-sm text-white font-mono">
            <Timer className="w-3.5 h-3.5 text-[#52525B]" />
            {Math.round(trace.duration_ms || 0)}ms
          </div>
          {trace.span_count && (
            <div className="text-xs text-[#3F3F46] mt-0.5">
              {trace.span_count} spans
            </div>
          )}
        </div>

        {/* Chevron */}
        <div className="col-span-1 flex justify-end">
          <ChevronRight className="w-5 h-5 text-[#1E1E28] group-hover:text-[#52525B] transition-colors group-hover:translate-x-0.5 transform duration-200" />
        </div>
      </div>
    </Link>
  )
}