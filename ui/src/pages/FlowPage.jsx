import React, { useState, useEffect } from 'react'
import ReactFlow, { Background, Controls, useNodesState, useEdgesState, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import { API_URL } from '../App'
import { Loader2, AlertTriangle, Bot, GitBranch } from 'lucide-react'

const TYPE_COLORS = {
  agent:     { bg: '#E8501A', border: '#F97316' },
  llm:       { bg: '#7C3AED', border: '#8B5CF6' },
  tool:      { bg: '#1A8A5A', border: '#10B981' },
  database:  { bg: '#C07A00', border: '#F59E0B' },
  retrieval: { bg: '#2563EB', border: '#3B82F6' },
  default:   { bg: '#6B7280', border: '#9CA3AF' },
}

const getLayoutedElements = (nodes, edges) => {
  if (!nodes.length) return { nodes: [], edges: [] }
  const g = new dagre.graphlib.Graph().setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 70, ranksep: 180 })
  nodes.forEach(node => g.setNode(node.id, { width: 240, height: 72 }))
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
  const [services, setServices] = useState([])
  const [selectedService, setSelectedService] = useState('')
  const [runs, setRuns] = useState([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/v1/services`)
      .then(res => res.json())
      .then(data => {
        setServices(data || [])
        if (data?.length === 1) setSelectedService(data[0])
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedService) { setRuns([]); return }
    setLoading(true)
    fetch(`${API_URL}/v1/runs?service_name=${selectedService}&limit=50`)
      .then(res => res.json())
      .then(data => {
        const runList = Array.isArray(data) ? data : (data.items || [])
        setRuns(runList)
        if (runList.length > 0) setSelectedRunId(runList[0].run_id)
      })
      .catch(() => setError('Failed to load runs'))
      .finally(() => setLoading(false))
  }, [selectedService])

  useEffect(() => {
    if (!selectedRunId) return
    setLoading(true)
    setError(null)
    fetch(`${API_URL}/v1/runs/${selectedRunId}/graph`)
      .then(res => res.json())
      .then(data => {
        const graph = data.graph || { nodes: [], edges: [] }
        if (!graph.nodes.length) { setError('No spans found for this run'); return }

        const newNodes = graph.nodes.map(node => {
          const spanType = node.type || 'default'
          const colors = TYPE_COLORS[spanType] || TYPE_COLORS.default
          return {
            id: node.id,
            type: 'default',
            data: {
              label: (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <span style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: colors.border }}>
                    {spanType}
                  </span>
                  <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {node.data?.label}
                  </span>
                  <span style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
                    {Math.round(node.data?.duration_ms || 0)}ms
                  </span>
                </div>
              )
            },
            style: {
              background: 'var(--bg-surface)',
              border: `1px solid var(--border-subtle)`,
              borderLeft: `3px solid ${colors.bg}`,
              borderRadius: 8,
              padding: '10px 12px',
              width: 240,
              boxShadow: 'var(--shadow-sm)',
              color: 'var(--text-primary)',
            }
          }
        })

        const newEdges = graph.edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          type: 'smoothstep',
          animated: false,
          style: { stroke: 'var(--border-default)', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#D4D2CF', width: 14, height: 14 },
        }))

        const layout = getLayoutedElements(newNodes, newEdges)
        setNodes(layout.nodes)
        setEdges(layout.edges)
      })
      .catch(() => setError('Failed to load run graph'))
      .finally(() => setLoading(false))
  }, [selectedRunId, setNodes, setEdges])

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>

      {/* Header / filter bar */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <GitBranch style={{ width: 14, height: 14, color: 'var(--accent)' }} />
          <span className="page-title">Flow Graph</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* Agent select */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Bot style={{
              position: 'absolute', left: 8,
              width: 12, height: 12,
              color: 'var(--text-tertiary)',
              pointerEvents: 'none',
            }} />
            <select
              className="select-premium"
              value={selectedService}
              onChange={(e) => { setSelectedService(e.target.value); setSelectedRunId('') }}
              style={{ paddingLeft: 26, minWidth: 160 }}
            >
              <option value="" disabled>Select agent...</option>
              {services.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          {/* Run select */}
          <select
            className="select-premium"
            value={selectedRunId}
            onChange={(e) => setSelectedRunId(e.target.value)}
            disabled={!selectedService}
            style={{ minWidth: 300 }}
          >
            <option value="" disabled>
              {selectedService ? 'Select run...' : 'Select an agent first...'}
            </option>
            {runs.map(run => (
              <option key={run.run_id} value={run.run_id}>
                {new Date(run.started_at * 1000).toLocaleTimeString()} — {run.input_summary || run.service_name || run.run_id.slice(0, 8)} ({run.span_count} spans)
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Canvas */}
      <div style={{ flex: 1, position: 'relative', background: 'var(--bg-primary)' }}>
        {loading && (
          <div style={{
            position: 'absolute', inset: 0, zIndex: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(247, 245, 243, 0.85)',
            backdropFilter: 'blur(4px)',
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
              <Loader2 style={{ width: 20, height: 20, color: 'var(--accent)' }} className="animate-spin" />
              <span style={{ fontSize: 12, color: 'var(--text-tertiary)' }}>Loading flow...</span>
            </div>
          </div>
        )}

        {error ? (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
            <div style={{
              width: 40, height: 40, borderRadius: 10,
              background: 'var(--amber-bg)', border: '1px solid rgba(192,122,0,0.15)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <AlertTriangle style={{ width: 18, height: 18, color: 'var(--amber)' }} />
            </div>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{error}</span>
          </div>
        ) : !selectedRunId ? (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
            <div style={{
              width: 44, height: 44, borderRadius: 10,
              background: 'var(--bg-sunken)', border: '1px solid var(--border-subtle)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <GitBranch style={{ width: 20, height: 20, color: 'var(--text-disabled)' }} />
            </div>
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Select an agent and run to visualize the execution flow
            </span>
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
            <Background
              color="var(--border-faint)"
              gap={20}
              size={1}
              style={{ background: 'var(--bg-primary)' }}
            />
            <Controls style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 8,
              boxShadow: 'var(--shadow-sm)',
            }} />
          </ReactFlow>
        )}
      </div>
    </div>
  )
}
