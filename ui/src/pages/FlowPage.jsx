import React, { useState, useEffect, useCallback } from 'react'
import ReactFlow, { Background, Controls, useNodesState, useEdgesState, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import { API_URL } from '../App'
import { Loader2, AlertTriangle, Bot, GitBranch } from 'lucide-react'

const TYPE_COLORS = {
    agent: { bg: '#3B82F6', border: '#60A5FA', glow: 'rgba(59, 130, 246, 0.15)' },
    llm: { bg: '#8B5CF6', border: '#A78BFA', glow: 'rgba(139, 92, 246, 0.15)' },
    tool: { bg: '#10B981', border: '#34D399', glow: 'rgba(16, 185, 129, 0.15)' },
    database: { bg: '#F59E0B', border: '#FBBF24', glow: 'rgba(245, 158, 11, 0.15)' },
    retrieval: { bg: '#EC4899', border: '#F472B6', glow: 'rgba(236, 72, 153, 0.15)' },
    default: { bg: '#6B7280', border: '#9CA3AF', glow: 'rgba(107, 114, 128, 0.1)' },
}

const getLayoutedElements = (nodes, edges) => {
    if (!nodes.length) return { nodes: [], edges: [] }
    const g = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
    g.setGraph({ rankdir: 'LR', nodesep: 80, ranksep: 200 })
    nodes.forEach(node => g.setNode(node.id, { width: 260, height: 84 }))
    edges.forEach(edge => g.setEdge(edge.source, edge.target))
    dagre.layout(g)

    return {
        nodes: nodes.map(node => {
            const { x, y } = g.node(node.id)
            return { ...node, position: { x, y } }
        }),
        edges
    }
}

export default function FlowPage() {
    // Selection State
    const [services, setServices] = useState([])
    const [selectedService, setSelectedService] = useState('')

    const [traces, setTraces] = useState([])
    const [selectedTraceId, setSelectedTraceId] = useState('')

    // Graph State
    const [nodes, setNodes, onNodesChange] = useNodesState([])
    const [edges, setEdges, onEdgesChange] = useEdgesState([])
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    // 1. Fetch Available Agents on Mount
    useEffect(() => {
        fetch(`${API_URL}/services`)
            .then(res => res.json())
            .then(data => {
                setServices(data || [])
                if (data && data.length === 1) setSelectedService(data[0])
            })
            .catch(err => console.error("Failed to fetch services:", err))
    }, [])

    // 2. Fetch Traces when Agent is Selected
    useEffect(() => {
        if (!selectedService) {
            setTraces([])
            return
        }

        setLoading(true)
        fetch(`${API_URL}/traces?service_name=${selectedService}&limit=50`)
            .then(res => res.json())
            .then(data => {
                const traceList = Array.isArray(data) ? data : []
                setTraces(traceList)
                if (traceList.length > 0) {
                    setSelectedTraceId(traceList[0].trace_id)
                }
            })
            .catch(err => setError("Failed to load traces"))
            .finally(() => setLoading(false))
    }, [selectedService])

    // 3. Fetch Graph Data (Spans) when Trace is Selected
    useEffect(() => {
        if (!selectedTraceId) return

        setLoading(true)
        setError(null)
        fetch(`${API_URL}/traces/${selectedTraceId}`)
            .then(res => res.json())
            .then(data => {
                const spans = Array.isArray(data) ? data : (data.spans || [])
                if (spans.length === 0) {
                    setError("No spans found for this trace")
                    return
                }

                // Transform spans to nodes
                const newNodes = spans.map(span => {
                    const colors = TYPE_COLORS[span.span_type] || TYPE_COLORS.default
                    return {
                        id: span.span_id,
                        type: 'default',
                        data: {
                            label: (
                                <div className="flex flex-col gap-0.5">
                                    <span className="font-bold text-[10px] uppercase tracking-wider"
                                        style={{ color: colors.border, opacity: 0.9 }}
                                    >
                                        {span.span_type}
                                    </span>
                                    <span className="font-semibold text-sm truncate text-white">{span.name}</span>
                                    <span className="text-[11px] text-[#71717A] font-mono">{Math.round(span.duration_ms || 0)}ms</span>
                                </div>
                            )
                        },
                        style: {
                            background: 'linear-gradient(145deg, #16161B, #111115)',
                            color: '#fff',
                            border: `1px solid ${colors.border}30`,
                            borderLeft: `3px solid ${colors.bg}`,
                            borderRadius: '12px',
                            padding: '12px 14px',
                            width: 260,
                            boxShadow: `0 4px 16px ${colors.glow}, 0 0 0 1px ${colors.border}15`,
                        }
                    }
                })

                // Transform parent_ids to edges
                const newEdges = spans
                    .filter(s => s.parent_span_id)
                    .map(s => ({
                        id: `${s.parent_span_id}-${s.span_id}`,
                        source: s.parent_span_id,
                        target: s.span_id,
                        type: 'smoothstep',
                        animated: true,
                        style: { stroke: '#27272A', strokeWidth: 2 },
                        markerEnd: { type: MarkerType.ArrowClosed, color: '#3F3F46', width: 16, height: 16 },
                    }))

                const layout = getLayoutedElements(newNodes, newEdges)
                setNodes(layout.nodes)
                setEdges(layout.edges)
            })
            .catch(err => setError("Failed to load trace details"))
            .finally(() => setLoading(false))

    }, [selectedTraceId, setNodes, setEdges])

    return (
        <div className="h-full flex flex-col bg-[var(--bg-primary)]">
            {/* Header / Filter Bar */}
            <div className="p-4 flex gap-4 items-center" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1))' }}
                    >
                        <GitBranch className="w-4 h-4 text-blue-400" />
                    </div>
                    <span className="text-[15px] font-semibold text-white">Agent Flow</span>
                </div>

                <div className="h-6 w-px bg-[var(--border-subtle)]" />

                {/* Dropdown 1: Select Agent */}
                <div className="relative">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Bot className="h-3.5 w-3.5 text-[#52525B]" />
                    </div>
                    <select
                        className="select-premium pl-9 min-w-[160px]"
                        value={selectedService}
                        onChange={(e) => {
                            setSelectedService(e.target.value)
                            setSelectedTraceId('')
                        }}
                    >
                        <option value="" disabled>Select Agent...</option>
                        {services.map(s => (
                            <option key={s} value={s}>{s}</option>
                        ))}
                    </select>
                </div>

                {/* Dropdown 2: Select Trace */}
                <select
                    className="select-premium min-w-[320px]"
                    value={selectedTraceId}
                    onChange={(e) => setSelectedTraceId(e.target.value)}
                    disabled={!selectedService}
                >
                    <option value="" disabled>
                        {selectedService ? "Select Trace..." : "Select an agent first..."}
                    </option>
                    {traces.map(t => (
                        <option key={t.trace_id} value={t.trace_id}>
                            {new Date(t.start_time * 1000).toLocaleTimeString()} — {t.name} ({t.span_count} spans)
                        </option>
                    ))}
                </select>
            </div>

            {/* Graph Canvas */}
            <div className="flex-1 relative">
                {loading && (
                    <div className="absolute inset-0 z-10 flex items-center justify-center"
                        style={{ background: 'rgba(8, 8, 10, 0.85)', backdropFilter: 'blur(8px)' }}
                    >
                        <div className="flex flex-col items-center gap-3">
                            <Loader2 className="w-8 h-8 text-blue-500 animate-spin" />
                            <span className="text-xs text-[#52525B]">Loading flow...</span>
                        </div>
                    </div>
                )}

                {error ? (
                    <div className="h-full flex items-center justify-center flex-col gap-3">
                        <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
                            style={{ background: 'rgba(245, 158, 11, 0.08)' }}
                        >
                            <AlertTriangle className="w-6 h-6 text-amber-500/70" />
                        </div>
                        <p className="text-[#52525B] text-sm">{error}</p>
                    </div>
                ) : !selectedTraceId ? (
                    <div className="h-full flex flex-col items-center justify-center">
                        <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
                            style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(139, 92, 246, 0.04))' }}
                        >
                            <GitBranch className="w-7 h-7 text-[#1E1E28]" />
                        </div>
                        <p className="text-[#3F3F46] text-sm">Select an Agent and Trace to visualize the execution flow</p>
                    </div>
                ) : (
                    <ReactFlow
                        nodes={nodes}
                        edges={edges}
                        onNodesChange={onNodesChange}
                        onEdgesChange={onEdgesChange}
                        fitView
                        minZoom={0.1}
                        maxZoom={2}
                    >
                        <Background color="#1E1E28" gap={24} size={1} />
                        <Controls
                            className="!bg-[#131318] !border !border-[var(--border-subtle)] !rounded-xl !shadow-lg"
                            style={{ button: { backgroundColor: '#131318', color: '#71717A', borderColor: 'var(--border-subtle)' } }}
                        />
                    </ReactFlow>
                )}
            </div>
        </div>
    )
}