import React, { useState, useEffect, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactFlow, {
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  MarkerType
} from 'reactflow'
import 'reactflow/dist/style.css'
import {
  ArrowLeft,
  Cpu,
  Wrench,
  Database,
  Shield,
  Bot,
  Zap,
  CheckCircle,
  AlertCircle,
  Loader2,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Hash,
  Link as LinkIcon,
  HardDrive,
  Globe,
  Archive,
  ArrowRightLeft,
  ListOrdered,
  FileText,
  Send,
  FileCode,
  CheckSquare,
  Puzzle,
} from 'lucide-react'
import { API_URL } from '../App'

const SPAN_TYPE_CONFIG = {
  llm:              { icon: Cpu,           color: '#7C3AED', label: 'LLM' },
  tool:             { icon: Wrench,        color: '#1A8A5A', label: 'Tool' },
  retrieval:        { icon: Database,      color: '#2563EB', label: 'Retrieval' },
  guardrail:        { icon: Shield,        color: '#C07A00', label: 'Guardrail' },
  agent:            { icon: Bot,           color: '#E8501A', label: 'Agent' },
  function:         { icon: Zap,           color: '#6B7280', label: 'Function' },
  embedding:        { icon: Hash,          color: '#0891B2', label: 'Embedding' },
  chain:            { icon: LinkIcon,      color: '#EA580C', label: 'Chain' },
  memory:           { icon: HardDrive,     color: '#0D9488', label: 'Memory' },
  http:             { icon: Globe,         color: '#4F46E5', label: 'HTTP' },
  database:         { icon: Database,      color: '#0369A1', label: 'Database' },
  cache:            { icon: Archive,       color: '#65A30D', label: 'Cache' },
  handoff:          { icon: ArrowRightLeft,color: '#C026D3', label: 'Handoff' },
  rerank:           { icon: ListOrdered,   color: '#CA8A04', label: 'Rerank' },
  file_io:          { icon: FileText,      color: '#64748B', label: 'File I/O' },
  web_request:      { icon: Send,          color: '#7C3AED', label: 'Web Request' },
  prompt_template:  { icon: FileCode,      color: '#16A34A', label: 'Template' },
  validation:       { icon: CheckSquare,   color: '#EAB308', label: 'Validation' },
  logging:          { icon: FileText,      color: '#94A3B8', label: 'Logging' },
  custom:           { icon: Puzzle,        color: '#9CA3AF', label: 'Custom' },
}

function SpanNode({ data, selected }) {
  const config = SPAN_TYPE_CONFIG[data.span_type] || SPAN_TYPE_CONFIG.function
  const Icon = config.icon
  const isError = data.status === 'error'

  return (
    <div style={{
      padding: '8px 12px',
      borderRadius: 8,
      border: `1px solid ${selected ? 'var(--border-strong)' : 'var(--border-subtle)'}`,
      borderLeft: `3px solid ${config.color}`,
      background: selected ? 'var(--bg-raised)' : 'var(--bg-surface)',
      minWidth: 180,
      cursor: 'pointer',
      boxShadow: selected ? 'var(--shadow-md)' : 'var(--shadow-sm)',
      transition: 'border-color 0.12s, box-shadow 0.12s',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <Icon style={{ width: 11, height: 11, color: config.color, flexShrink: 0 }} />
        <span style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-tertiary)' }}>
          {config.label}
        </span>
        {isError && <AlertCircle style={{ width: 10, height: 10, color: 'var(--red)', marginLeft: 'auto' }} />}
      </div>
      <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {data.name}
      </div>
      {data.duration_ms && (
        <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'monospace', marginTop: 2 }}>
          {data.duration_ms > 1000 ? `${(data.duration_ms / 1000).toFixed(2)}s` : `${Math.round(data.duration_ms)}ms`}
        </div>
      )}
    </div>
  )
}

const nodeTypes = { span: SpanNode }

function RunDetailPage() {
  const { runId } = useParams()
  const [spans, setSpans] = useState([])
  const [run, setRun] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedSpan, setSelectedSpan] = useState(null)
  const [viewMode, setViewMode] = useState('dag')

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  useEffect(() => {
    const fetchRun = async () => {
      setLoading(true)
      setError(null)
      try {
        const [runRes, graphRes, timelineRes] = await Promise.all([
          fetch(`${API_URL}/v1/runs/${runId}`),
          fetch(`${API_URL}/v1/runs/${runId}/graph`),
          fetch(`${API_URL}/v1/runs/${runId}/timeline`),
        ])
        if (!runRes.ok || !graphRes.ok || !timelineRes.ok) throw new Error('Failed to load run detail')
        const [runData, graphData, timelineData] = await Promise.all([
          runRes.json(), graphRes.json(), timelineRes.json(),
        ])
        setRun(runData.run)

        const artifactsBySpan = (runData.artifacts || []).reduce((acc, artifact) => {
          if (!acc[artifact.span_id]) acc[artifact.span_id] = []
          acc[artifact.span_id].push(artifact)
          return acc
        }, {})

        const enrichedSpans = (timelineData.items || []).map((span) => {
          const artifacts = artifactsBySpan[span.span_id] || []
          const inputArtifacts = artifacts.filter(a => a.role.startsWith('input'))
          const outputArtifacts = artifacts.filter(a => a.role.startsWith('output'))
          const mediaArtifacts = artifacts.filter(a =>
            a.role === 'derived.media_analysis' || a.role === 'input.image' ||
            a.role === 'output.media' || a.content_type?.startsWith('image/')
          )
          return {
            ...span,
            start_time: span.start_time ?? span.started_at ?? 0,
            end_time: span.end_time ?? span.ended_at ?? null,
            span_type: span.kind,
            inputArtifacts,
            outputArtifacts,
            mediaArtifacts,
            inputs: inputArtifacts[0]?.preview || inputArtifacts[0]?.inline_text || null,
            outputs: outputArtifacts[0]?.preview || outputArtifacts[0]?.inline_text || null,
          }
        })

        setSpans(enrichedSpans)
        const { nodes: newNodes, edges: newEdges } = buildGraphLayout(enrichedSpans, graphData.graph)
        setNodes(newNodes)
        setEdges(newEdges)
        const root = enrichedSpans.find(s => !s.parent_span_id)
        if (root) setSelectedSpan(root)
      } catch (err) {
        setError(err.message || 'Failed to fetch run')
      } finally {
        setLoading(false)
      }
    }
    fetchRun()
  }, [runId])

  const onNodeClick = useCallback((_, node) => {
    const span = spans.find(s => s.span_id === node.id)
    if (span) setSelectedSpan(span)
  }, [spans])

  const rootSpan = spans.find(s => !s.parent_span_id)

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
      <Loader2 style={{ width: 20, height: 20, color: 'var(--accent)' }} className="animate-spin" />
    </div>
  )

  if (error) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', padding: 24 }}>
      <div style={{
        maxWidth: 400,
        padding: '12px 16px',
        background: 'var(--red-bg)',
        border: '1px solid rgba(217,48,37,0.15)',
        borderRadius: 10,
        fontSize: 13,
        color: 'var(--red)',
      }}>
        {error}
      </div>
    </div>
  )

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>

      {/* Header */}
      <div style={{
        height: 48,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        borderBottom: '1px solid var(--border-faint)',
        background: 'var(--bg-surface)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Link
            to="/"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: 28, height: 28, borderRadius: 6,
              border: '1px solid var(--border-default)',
              color: 'var(--text-secondary)',
              textDecoration: 'none',
              transition: 'border-color 0.12s, color 0.12s',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--border-strong)'; e.currentTarget.style.color = 'var(--text-primary)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-default)'; e.currentTarget.style.color = 'var(--text-secondary)' }}
          >
            <ArrowLeft style={{ width: 13, height: 13 }} />
          </Link>

          <div style={{ width: 1, height: 16, background: 'var(--border-faint)' }} />

          <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {rootSpan?.name || run?.service_name || 'Run'}
          </span>
          <span style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--text-tertiary)' }}>{runId?.slice(0, 10)}...</span>
        </div>

        {/* View mode toggle */}
        <div style={{
          display: 'flex',
          background: 'var(--bg-sunken)',
          border: '1px solid var(--border-default)',
          borderRadius: 7,
          padding: 2,
          gap: 2,
        }}>
          {['dag', 'waterfall'].map(mode => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              style={{
                padding: '3px 10px',
                borderRadius: 5,
                fontSize: 11,
                fontWeight: 500,
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.12s',
                textTransform: 'capitalize',
                fontFamily: 'inherit',
                background: viewMode === mode ? 'var(--bg-surface)' : 'transparent',
                color: viewMode === mode ? 'var(--text-primary)' : 'var(--text-tertiary)',
                boxShadow: viewMode === mode ? 'var(--shadow-sm)' : 'none',
              }}
            >
              {mode === 'dag' ? 'DAG' : 'Waterfall'}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>

        {/* Visualization */}
        <div style={{ flex: 1, position: 'relative', background: 'var(--bg-primary)' }}>
          {viewMode === 'dag' ? (
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              minZoom={0.1}
              maxZoom={2}
              nodesDraggable={false}
            >
              <Background color="var(--border-faint)" gap={20} size={1} style={{ background: 'var(--bg-primary)' }} />
              <Controls style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 8,
                boxShadow: 'var(--shadow-sm)',
              }} />
            </ReactFlow>
          ) : (
            <WaterfallView spans={spans} selectedSpan={selectedSpan} onSelect={setSelectedSpan} />
          )}
        </div>

        {/* Inspector panel */}
        {selectedSpan && (
          <SpanInspector span={selectedSpan} onClose={() => setSelectedSpan(null)} />
        )}
      </div>
    </div>
  )
}

function WaterfallView({ spans, selectedSpan, onSelect }) {
  if (!spans.length) return null
  const minStart = Math.min(...spans.map(s => s.start_time))
  const maxEnd = Math.max(...spans.map(s => s.end_time || s.start_time))
  const totalDuration = maxEnd - minStart
  const sortedSpans = [...spans].sort((a, b) => a.start_time - b.start_time)

  return (
    <div style={{ padding: 12, overflowY: 'auto', height: '100%', background: 'var(--bg-primary)' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {sortedSpans.map((span) => {
          const leftPct = ((span.start_time - minStart) / totalDuration) * 100
          const widthPct = ((span.duration_ms || 1) / (totalDuration * 1000)) * 100
          const config = SPAN_TYPE_CONFIG[span.span_type] || SPAN_TYPE_CONFIG.function
          const isSelected = selectedSpan?.span_id === span.span_id

          let depth = 0
          let parentId = span.parent_span_id
          while (parentId) {
            depth++
            const parent = spans.find(s => s.span_id === parentId)
            parentId = parent?.parent_span_id
          }

          return (
            <div
              key={span.span_id}
              onClick={() => onSelect(span)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: `5px 10px 5px ${8 + depth * 14}px`,
                borderRadius: 7,
                cursor: 'pointer',
                background: isSelected ? 'var(--bg-surface)' : 'transparent',
                border: `1px solid ${isSelected ? 'var(--border-default)' : 'transparent'}`,
                transition: 'background 0.1s',
              }}
            >
              <div style={{ width: 120, display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0 }}>
                <config.icon style={{ width: 10, height: 10, color: config.color, flexShrink: 0 }} />
                <span style={{ fontSize: 11, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {span.name}
                </span>
              </div>
              <div style={{ flex: 1, height: 4, background: 'var(--bg-sunken)', borderRadius: 100, overflow: 'hidden', position: 'relative' }}>
                <div style={{
                  position: 'absolute',
                  height: '100%',
                  left: `${leftPct}%`,
                  width: `${Math.max(widthPct, 0.5)}%`,
                  background: span.status === 'error' ? 'var(--red)' : config.color,
                  borderRadius: 100,
                }} />
              </div>
              <span style={{ width: 48, fontSize: 10, fontFamily: 'monospace', color: 'var(--text-tertiary)', textAlign: 'right', flexShrink: 0 }}>
                {span.duration_ms ? `${Math.round(span.duration_ms)}ms` : ''}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function formatArtifactRole(role) {
  return role.split('.').map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(' / ')
}

function getArtifactPayload(artifact) {
  if (!artifact) return null
  if (artifact.inline_text != null) return artifact.inline_text
  if (artifact.preview != null) return artifact.preview
  return artifact.metadata || null
}

function ArtifactCard({ artifact }) {
  const payload = getArtifactPayload(artifact)
  const isImage = artifact.content_type?.startsWith('image/')
  const imageSrc = artifact.preview?.data || artifact.inline_text || null

  return (
    <div style={{
      background: 'var(--bg-raised)',
      border: '1px solid var(--border-faint)',
      borderRadius: 8,
      padding: '10px 12px',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-primary)' }}>{formatArtifactRole(artifact.role)}</div>
          <div style={{ fontSize: 10, color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{artifact.content_type}</div>
        </div>
        {artifact.size_bytes && (
          <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--text-tertiary)' }}>
            {(artifact.size_bytes / 1024).toFixed(1)} KB
          </span>
        )}
      </div>

      {isImage && imageSrc && (
        <div style={{ borderRadius: 6, overflow: 'hidden', marginBottom: 8 }}>
          <img src={imageSrc} alt={artifact.role} style={{ width: '100%', maxHeight: 200, objectFit: 'contain' }} />
        </div>
      )}

      {typeof payload === 'string' ? (
        <pre style={{
          fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
          padding: '8px 10px', background: 'var(--bg-surface)',
          border: '1px solid var(--border-faint)', borderRadius: 6,
          overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
          margin: 0, maxHeight: 150,
        }}>
          {payload}
        </pre>
      ) : payload ? (
        <pre style={{
          fontSize: 11, fontFamily: 'monospace', color: 'var(--text-secondary)',
          padding: '8px 10px', background: 'var(--bg-surface)',
          border: '1px solid var(--border-faint)', borderRadius: 6,
          overflow: 'auto', margin: 0, maxHeight: 150,
        }}>
          {JSON.stringify(payload, null, 2)}
        </pre>
      ) : (
        <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>No preview</span>
      )}
    </div>
  )
}

function ArtifactSection({ title, artifacts, emptyLabel }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
          {title}
        </span>
        {artifacts.length > 0 && (
          <span className="badge badge-neutral" style={{ fontSize: 9, padding: '0 5px' }}>{artifacts.length}</span>
        )}
      </div>
      {artifacts.length > 0
        ? artifacts.map(a => <ArtifactCard key={a.artifact_id} artifact={a} />)
        : <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{emptyLabel}</span>
      }
    </div>
  )
}

function SpanInspector({ span, onClose }) {
  const [activeTab, setActiveTab] = useState('overview')
  const [feedbackState, setFeedbackState] = useState({ submitting: false, score: span.score ?? null, error: null })
  const config = SPAN_TYPE_CONFIG[span.span_type] || SPAN_TYPE_CONFIG.function
  const Icon = config.icon

  useEffect(() => {
    setFeedbackState({ submitting: false, score: span.score ?? null, error: null })
  }, [span.span_id, span.score])

  const submitFeedback = async (score) => {
    setFeedbackState(c => ({ ...c, submitting: true, error: null }))
    try {
      const res = await fetch(`${API_URL}/v1/spans/${span.span_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ score }),
      })
      if (!res.ok) throw new Error('Failed to save feedback')
      setFeedbackState({ submitting: false, score, error: null })
    } catch (err) {
      setFeedbackState(c => ({ ...c, submitting: false, error: err.message }))
    }
  }

  return (
    <div style={{
      width: 360,
      borderLeft: '1px solid var(--border-faint)',
      background: 'var(--bg-surface)',
      display: 'flex',
      flexDirection: 'column',
      flexShrink: 0,
    }}>
      {/* Inspector header */}
      <div style={{
        padding: '12px 14px',
        borderBottom: '1px solid var(--border-faint)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 10,
      }}>
        <div style={{
          width: 32, height: 32, borderRadius: 8,
          background: 'var(--bg-raised)', border: '1px solid var(--border-subtle)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>
          <Icon style={{ width: 14, height: 14, color: config.color }} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {span.name}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginTop: 3 }}>
            <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              {config.label}
            </span>
            {span.status === 'error' && (
              <span className="badge badge-error" style={{ fontSize: 10, padding: '0 5px' }}>Error</span>
            )}
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          {span.duration_ms && (
            <div style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-primary)' }}>
              {span.duration_ms.toFixed(1)}ms
            </div>
          )}
          {span.usage?.total_cost > 0 && (
            <div style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--green)', marginTop: 2 }}>
              ${span.usage.total_cost.toFixed(5)}
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--border-faint)' }}>
        {['overview', 'input', 'output'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              flex: 1,
              padding: '8px 0',
              fontSize: 11,
              fontWeight: 500,
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              color: activeTab === tab ? 'var(--text-primary)' : 'var(--text-tertiary)',
              borderBottom: `2px solid ${activeTab === tab ? 'var(--accent)' : 'transparent'}`,
              marginBottom: -1,
              transition: 'color 0.12s',
              textTransform: 'capitalize',
              fontFamily: 'inherit',
            }}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '14px' }}>
        {activeTab === 'overview' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {span.model && (
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
                  Model
                </div>
                <div style={{ fontSize: 12, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{span.model}</div>
              </div>
            )}

            {span.usage && (
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                  Token Usage
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                  {[
                    { label: 'Prompt', val: span.usage.prompt_tokens },
                    { label: 'Compl.', val: span.usage.completion_tokens },
                    { label: 'Total', val: span.usage.total_tokens },
                  ].map(({ label, val }) => (
                    <div key={label} style={{
                      padding: '7px 8px', background: 'var(--bg-raised)',
                      border: '1px solid var(--border-faint)', borderRadius: 7,
                    }}>
                      <div style={{ fontSize: 9, color: 'var(--text-tertiary)', marginBottom: 2 }}>{label}</div>
                      <div style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--text-primary)' }}>{val}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <ArtifactSection title="Input Artifacts" artifacts={span.inputArtifacts || []} emptyLabel="No input artifacts" />
            {span.mediaArtifacts?.length > 0 && (
              <ArtifactSection title="Media Artifacts" artifacts={span.mediaArtifacts} emptyLabel="No media artifacts" />
            )}
            <ArtifactSection title="Output Artifacts" artifacts={span.outputArtifacts || []} emptyLabel="No output artifacts" />

            {span.error_message && (
              <div style={{
                padding: '8px 10px',
                background: 'var(--red-bg)', border: '1px solid rgba(217,48,37,0.15)',
                borderRadius: 8,
              }}>
                <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--red)', marginBottom: 4 }}>Error</div>
                <div style={{ fontSize: 11, color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{span.error_message}</div>
              </div>
            )}

            <div>
              <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                Metadata
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <IdRow label="Span ID" value={span.span_id} />
                <IdRow label="Run ID" value={span.run_id} />
                {span.parent_span_id && <IdRow label="Parent" value={span.parent_span_id} />}
              </div>
            </div>

            {/* Feedback */}
            <div style={{
              paddingTop: 12,
              borderTop: '1px solid var(--border-faint)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Was this useful?</span>
                {feedbackState.error && (
                  <div style={{ fontSize: 10, color: 'var(--red)', marginTop: 2 }}>{feedbackState.error}</div>
                )}
                {feedbackState.score != null && !feedbackState.error && (
                  <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 2 }}>Feedback saved</div>
                )}
              </div>
              <div style={{ display: 'flex', gap: 4 }}>
                {[
                  { score: 1, icon: ThumbsUp },
                  { score: -1, icon: ThumbsDown },
                ].map(({ score, icon: FeedbackIcon }) => (
                  <button
                    key={score}
                    onClick={() => submitFeedback(score)}
                    disabled={feedbackState.submitting}
                    style={{
                      width: 28, height: 28, borderRadius: 6, border: 'none',
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      background: feedbackState.score === score ? 'var(--accent-muted)' : 'transparent',
                      color: feedbackState.score === score ? 'var(--accent)' : 'var(--text-tertiary)',
                      cursor: feedbackState.submitting ? 'not-allowed' : 'pointer',
                      transition: 'background 0.12s, color 0.12s',
                    }}
                    onMouseEnter={e => { if (feedbackState.score !== score) e.currentTarget.style.background = 'var(--bg-sunken)' }}
                    onMouseLeave={e => { if (feedbackState.score !== score) e.currentTarget.style.background = 'transparent' }}
                  >
                    <FeedbackIcon style={{ width: 12, height: 12 }} />
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'input' && (
          <ArtifactSection title="Input Artifacts" artifacts={span.inputArtifacts || []} emptyLabel="No input artifacts" />
        )}
        {activeTab === 'output' && (
          <ArtifactSection title="Output Artifacts" artifacts={span.outputArtifacts || []} emptyLabel="No output artifacts" />
        )}
      </div>
    </div>
  )
}

function IdRow({ label, value }) {
  const copy = () => navigator.clipboard.writeText(value)
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{label}</span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 10, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
          {value.slice(0, 8)}...
        </span>
        <button
          onClick={copy}
          className="btn btn-ghost"
          style={{ width: 20, height: 20, padding: 0, borderRadius: 4 }}
        >
          <Copy style={{ width: 9, height: 9 }} />
        </button>
      </div>
    </div>
  )
}

function buildDag(spans) {
  const nodes = [], edges = []
  const spanMap = new Map()
  spans.forEach(span => spanMap.set(span.span_id, span))
  const levels = new Map()
  spans.forEach(span => {
    let depth = 0, parentId = span.parent_span_id, attempts = 0
    while (parentId && attempts < 100) {
      depth++; attempts++
      const parent = spanMap.get(parentId)
      if (!parent) break
      parentId = parent.parent_span_id
    }
    levels.set(span.span_id, depth)
  })
  const spansByLevel = new Map()
  spans.forEach(span => {
    const level = levels.get(span.span_id) || 0
    if (!spansByLevel.has(level)) spansByLevel.set(level, [])
    spansByLevel.get(level).push(span)
  })
  spansByLevel.forEach((levelSpans, level) => {
    levelSpans.sort((a, b) => a.start_time - b.start_time)
    levelSpans.forEach((span, index) => {
      nodes.push({ id: span.span_id, type: 'span', position: { x: level * 280, y: index * 100 }, data: span })
      if (span.parent_span_id) {
        edges.push({
          id: `${span.parent_span_id}-${span.span_id}`,
          source: span.parent_span_id,
          target: span.span_id,
          type: 'smoothstep',
          animated: span.status === 'running',
          style: { stroke: 'var(--border-default)', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#D4D2CF' }
        })
      }
    })
  })
  return { nodes, edges }
}

function buildGraphLayout(spans, graph) {
  const fallback = buildDag(spans)
  if (!graph?.nodes?.length) return fallback
  const spanMap = new Map(spans.map(span => [span.span_id, span]))
  const positions = new Map(fallback.nodes.map(node => [node.id, node.position]))
  const nodes = graph.nodes.map(node => ({
    id: node.id,
    type: 'span',
    position: positions.get(node.id) || { x: 0, y: 0 },
    data: spanMap.get(node.id) || { ...node.data, span_type: node.type, span_id: node.id, start_time: 0, end_time: null },
  }))
  const nodeIds = new Set(nodes.map(n => n.id))
  fallback.nodes.forEach(node => { if (!nodeIds.has(node.id)) nodes.push(node) })
  const edges = (graph.edges?.length ? graph.edges : fallback.edges).map(edge => ({
    ...edge,
    type: 'smoothstep',
    animated: spanMap.get(edge.target)?.status === 'running',
    style: { stroke: 'var(--border-default)', strokeWidth: 1.5 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#D4D2CF' },
  }))
  return { nodes, edges }
}

export default RunDetailPage
