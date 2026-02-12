import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
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
  llm: { icon: Cpu, color: '#8b5cf6', label: 'LLM' },
  tool: { icon: Wrench, color: '#10b981', label: 'Tool' },
  retrieval: { icon: Database, color: '#3b82f6', label: 'Retrieval' },
  guardrail: { icon: Shield, color: '#f59e0b', label: 'Guardrail' },
  agent: { icon: Bot, color: '#ec4899', label: 'Agent' },
  function: { icon: Zap, color: '#6b7280', label: 'Function' },
  embedding: { icon: Hash, color: '#06b6d4', label: 'Embedding' },
  chain: { icon: LinkIcon, color: '#f97316', label: 'Chain' },
  memory: { icon: HardDrive, color: '#14b8a6', label: 'Memory' },
  http: { icon: Globe, color: '#6366f1', label: 'HTTP' },
  database: { icon: Database, color: '#0ea5e9', label: 'Database' },
  cache: { icon: Archive, color: '#84cc16', label: 'Cache' },
  handoff: { icon: ArrowRightLeft, color: '#d946ef', label: 'Handoff' },
  rerank: { icon: ListOrdered, color: '#eab308', label: 'Rerank' },
  file_io: { icon: FileText, color: '#64748b', label: 'File I/O' },
  web_request: { icon: Send, color: '#7c3aed', label: 'Web Request' },
  prompt_template: { icon: FileCode, color: '#22c55e', label: 'Template' },
  validation: { icon: CheckSquare, color: '#facc15', label: 'Validation' },
  logging: { icon: FileText, color: '#94a3b8', label: 'Logging' },
  custom: { icon: Puzzle, color: '#9ca3af', label: 'Custom' },
}

// Custom node component for the DAG
function SpanNode({ data, selected }) {
  const config = SPAN_TYPE_CONFIG[data.span_type] || SPAN_TYPE_CONFIG.function
  const Icon = config.icon
  const isError = data.status === 'error'
  const isRunning = data.status === 'running'

  return (
    <div
      className={`
        relative px-4 py-3 rounded-lg border transition-all cursor-pointer min-w-[180px]
        ${selected ? 'border-white bg-[#151518]' : 'border-[#27272A] bg-[#0C0C0E] hover:border-[#52525B]'}
      `}
    >
      {/* Running indicator */}
      {isRunning && (
        <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-500 rounded-full animate-pulse" />
      )}

      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-3.5 h-3.5" style={{ color: config.color }} />
        <span className="text-[10px] uppercase font-bold tracking-wider text-[#71717A]">{config.label}</span>
        {isError && <AlertCircle className="w-3 h-3 text-red-500 ml-auto" />}
        {data.status === 'success' && <CheckCircle className="w-3 h-3 text-[#27272A] ml-auto" />}
      </div>

      {/* Name */}
      <p className="text-sm font-medium text-white truncate mb-1">
        {data.name}
      </p>

      {/* Duration */}
      {data.duration_ms && (
        <p className="text-xs text-[#71717A] font-mono">
          {data.duration_ms > 1000
            ? `${(data.duration_ms / 1000).toFixed(2)}s`
            : `${Math.round(data.duration_ms)}ms`
          }
        </p>
      )}
    </div>
  )
}

const nodeTypes = { span: SpanNode }

function TraceDetailPage() {
  const { traceId } = useParams()
  const [spans, setSpans] = useState([])
  const [loading, setLoading] = useState(true)
  const [selectedSpan, setSelectedSpan] = useState(null)
  const [viewMode, setViewMode] = useState('dag') // 'dag' | 'waterfall'

  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  useEffect(() => {
    const fetchTrace = async () => {
      setLoading(true)
      try {
        const res = await fetch(`${API_URL}/traces/${traceId}?include_blobs=true`)
        const data = await res.json()
        setSpans(data)

        // Build DAG
        const { nodes: newNodes, edges: newEdges } = buildDag(data)
        setNodes(newNodes)
        setEdges(newEdges)

        // Select root span by default
        const root = data.find(s => !s.parent_span_id)
        if (root) setSelectedSpan(root)
      } catch (err) {
        console.error('Failed to fetch trace:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchTrace()
  }, [traceId])

  const onNodeClick = useCallback((_, node) => {
    const span = spans.find(s => s.span_id === node.id)
    if (span) setSelectedSpan(span)
  }, [spans])

  const rootSpan = spans.find(s => !s.parent_span_id)

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin text-[#52525B]" />
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="h-14 border-b border-[#27272A] bg-[#0C0C0E] flex items-center px-6 justify-between">
        <div className="flex items-center gap-4">
          <Link to="/" className="p-1.5 hover:bg-[#151518] rounded-md transition-colors text-[#A1A1AA] hover:text-white">
            <ArrowLeft className="w-4 h-4" />
          </Link>

          <div className="h-4 w-px bg-[#27272A]" />

          <h1 className="text-sm font-medium text-white truncate max-w-sm">
            {rootSpan?.name}
          </h1>
          <span className="text-xs font-mono text-[#52525B]">{traceId}</span>
        </div>

        <div className="flex items-center gap-2 bg-[#151518] p-1 rounded-lg border border-[#27272A]">
          <button
            onClick={() => setViewMode('dag')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${viewMode === 'dag' ? 'bg-[#27272A] text-white' : 'text-[#71717A] hover:text-white'}`}
          >DAG</button>
          <button
            onClick={() => setViewMode('waterfall')}
            className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${viewMode === 'waterfall' ? 'bg-[#27272A] text-white' : 'text-[#71717A] hover:text-white'}`}
          >Waterfall</button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex min-h-0">
        {/* Visualization */}
        <div className="flex-1 relative bg-[#0C0C0E]">
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
              <Background color="#151518" gap={20} size={1} />
              <Controls className="!bg-[#151518] !border-[#27272A]" />
            </ReactFlow>
          ) : (
            <WaterfallView spans={spans} selectedSpan={selectedSpan} onSelect={setSelectedSpan} />
          )}
        </div>

        {/* Span inspector */}
        {selectedSpan && (
          <SpanInspector
            span={selectedSpan}
            onClose={() => setSelectedSpan(null)}
          />
        )}
      </div>
    </div>
  )
}

function WaterfallView({ spans, selectedSpan, onSelect }) {
  if (spans.length === 0) return null

  const minStart = Math.min(...spans.map(s => s.start_time))
  const maxEnd = Math.max(...spans.map(s => s.end_time || s.start_time))
  const totalDuration = maxEnd - minStart

  const sortedSpans = [...spans].sort((a, b) => a.start_time - b.start_time)

  return (
    <div className="p-4 overflow-auto h-full bg-[#0C0C0E]">
      <div className="space-y-1">
        {sortedSpans.map((span) => {
          const leftPercent = ((span.start_time - minStart) / totalDuration) * 100
          const widthPercent = ((span.duration_ms || 1) / (totalDuration * 1000)) * 100
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
              className={`
                flex items-center gap-3 p-1.5 rounded-md cursor-pointer transition-all
                ${isSelected ? 'bg-[#151518] ring-1 ring-white' : 'hover:bg-[#151518]'}
              `}
              style={{ paddingLeft: 8 + depth * 12 }}
              onClick={() => onSelect(span)}
            >
              <div className="w-32 flex items-center gap-2 flex-shrink-0">
                <config.icon className="w-3 h-3" style={{ color: config.color }} />
                <span className="text-xs text-[#A1A1AA] truncate">{span.name}</span>
              </div>

              <div className="flex-1 h-1.5 bg-[#151518] rounded-full overflow-hidden relative">
                <div
                  className="absolute h-full rounded-full"
                  style={{
                    left: `${leftPercent}%`,
                    width: `${Math.max(widthPercent, 0.5)}%`,
                    backgroundColor: span.status === 'error' ? '#ef4444' : config.color,
                  }}
                />
              </div>

              <span className="w-12 text-[10px] font-mono text-[#52525B] text-right flex-shrink-0">
                {span.duration_ms ? `${Math.round(span.duration_ms)}ms` : ''}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function SpanInspector({ span, onClose }) {
  const [activeTab, setActiveTab] = useState('overview')
  const config = SPAN_TYPE_CONFIG[span.span_type] || SPAN_TYPE_CONFIG.function
  const Icon = config.icon

  return (
    <div className="w-[400px] border-l border-[#27272A] bg-[#0C0C0E] flex flex-col">
      {/* Inspector Header */}
      <div className="p-4 border-b border-[#27272A] flex items-start gap-3">
        <div className="p-2 bg-[#151518] rounded-lg border border-[#27272A]">
          <Icon className="w-4 h-4" style={{ color: config.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-white truncate">{span.name}</h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-[10px] uppercase font-bold text-[#52525B] tracking-wider">{config.label}</span>
            {span.status === 'error' && <span className="text-[10px] text-red-500 font-medium">Error</span>}
          </div>
        </div>
        <div className="flex-shrink-0 text-right">
          <p className="text-sm font-mono text-white">
            {span.duration_ms ? `${span.duration_ms.toFixed(1)}ms` : '--'}
          </p>
          {span.usage?.total_cost > 0 && (
            <p className="text-xs text-[#10B981] font-mono mt-0.5">
              ${span.usage.total_cost.toFixed(5)}
            </p>
          )}
        </div>
      </div>

      {/* Custom Minimal Tabs */}
      <div className="flex border-b border-[#27272A] px-2">
        {['overview', 'input', 'output'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-3 text-xs font-medium transition-all relative ${activeTab === tab
              ? 'text-white'
              : 'text-[#52525B] hover:text-[#A1A1AA]'
              }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
            {activeTab === tab && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white mx-4 rounded-full" />
            )}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-auto p-4 content-start">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {span.model && (
              <div className="space-y-1">
                <label className="text-[10px] font-medium text-[#52525B] uppercase">Model</label>
                <p className="text-sm text-[#A1A1AA] font-mono">{span.model}</p>
              </div>
            )}

            {span.usage && (
              <div className="space-y-2">
                <label className="text-[10px] font-medium text-[#52525B] uppercase">Token Usage</label>
                <div className="grid grid-cols-3 gap-2">
                  <div className="bg-[#151518] p-2 rounded border border-[#27272A]">
                    <p className="text-[10px] text-[#52525B]">Prompt</p>
                    <p className="text-xs font-mono text-white">{span.usage.prompt_tokens}</p>
                  </div>
                  <div className="bg-[#151518] p-2 rounded border border-[#27272A]">
                    <p className="text-[10px] text-[#52525B]">Compl.</p>
                    <p className="text-xs font-mono text-white">{span.usage.completion_tokens}</p>
                  </div>
                  <div className="bg-[#151518] p-2 rounded border border-[#27272A]">
                    <p className="text-[10px] text-[#52525B]">Total</p>
                    <p className="text-xs font-mono text-white">{span.usage.total_tokens}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Media Inputs Section */}
            {span.media_inputs && span.media_inputs.length > 0 && (
              <div className="space-y-2">
                <label className="text-[10px] font-medium text-[#52525B] uppercase flex items-center gap-2">
                  <span>📷 Media Inputs</span>
                  <span className="bg-[#3B82F6] text-white text-[10px] px-1.5 rounded">{span.media_inputs.length}</span>
                </label>
                <div className="space-y-2">
                  {span.media_inputs.map((media, idx) => (
                    <div key={idx} className="bg-[#151518] p-3 rounded border border-[#27272A]">
                      {/* Image Preview */}
                      {media.data && media.media_type === 'image' && (
                        <div className="mb-3 rounded overflow-hidden border border-[#27272A] bg-[#0C0C0E]">
                          <img
                            src={media.data}
                            alt="Input"
                            className="w-full h-auto max-h-[200px] object-contain"
                          />
                        </div>
                      )}

                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-sm">
                          {media.media_type === 'image' ? '🖼️' : media.media_type === 'audio' ? '🎵' : '🎬'}
                        </span>
                        <span className="text-xs font-medium text-white capitalize">{media.media_type}</span>
                        <span className="text-[10px] text-[#52525B] uppercase">{media.format}</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {media.width && media.height && (
                          <div>
                            <span className="text-[#52525B]">Size: </span>
                            <span className="text-[#A1A1AA]">{media.width}×{media.height}</span>
                          </div>
                        )}
                        {media.size_bytes && (
                          <div>
                            <span className="text-[#52525B]">Bytes: </span>
                            <span className="text-[#A1A1AA]">{(media.size_bytes / 1024).toFixed(1)} KB</span>
                          </div>
                        )}
                        {media.duration_seconds && (
                          <div>
                            <span className="text-[#52525B]">Duration: </span>
                            <span className="text-[#A1A1AA]">{media.duration_seconds.toFixed(1)}s</span>
                          </div>
                        )}
                        {media.estimated_tokens > 0 && (
                          <div>
                            <span className="text-[#52525B]">Est. Tokens: </span>
                            <span className="text-[#10B981] font-mono">{media.estimated_tokens}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {span.error_message && (
              <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                <p className="text-xs text-red-500 font-medium mb-1">Error</p>
                <p className="text-xs text-[#A1A1AA] break-all">{span.error_message}</p>
              </div>
            )}

            <div className="space-y-1">
              <label className="text-[10px] font-medium text-[#52525B] uppercase">Metadata</label>
              <div className="space-y-1">
                <IdRow label="Span ID" value={span.span_id} />
                <IdRow label="Trace ID" value={span.trace_id} />
                {span.parent_span_id && <IdRow label="Parent" value={span.parent_span_id} />}
              </div>
            </div>

            {/* Feedback Mini */}
            <div className="pt-4 border-t border-[#27272A] flex items-center justify-between">
              <span className="text-xs text-[#52525B]">Was this useful?</span>
              <div className="flex gap-1">
                <button className="p-1 hover:text-white text-[#52525B]"><ThumbsUp className="w-3 h-3" /></button>
                <button className="p-1 hover:text-white text-[#52525B]"><ThumbsDown className="w-3 h-3" /></button>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'input' && <JsonViewer data={span.inputs} />}
        {activeTab === 'output' && <JsonViewer data={span.outputs} />}
      </div>
    </div>
  )
}

function IdRow({ label, value }) {
  const copyToClipboard = () => navigator.clipboard.writeText(value)
  return (
    <div className="flex items-center justify-between group">
      <span className="text-xs text-[#52525B]">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono text-[#A1A1AA]">{value.slice(0, 8)}...</span>
        <button onClick={copyToClipboard} className="text-[#52525B] hover:text-white opacity-0 group-hover:opacity-100 transition-opacity">
          <Copy className="w-3 h-3" />
        </button>
      </div>
    </div>
  )
}

function JsonViewer({ data }) {
  if (!data) return <div className="text-center py-8 text-[#52525B] text-xs">No data</div>
  return (
    <pre className="text-xs font-mono p-3 bg-[#151518] rounded-lg overflow-auto border border-[#27272A] text-[#A1A1AA]">
      {JSON.stringify(data, null, 2)}
    </pre>
  )
}

// Keep buildDag same logic, but update edge style in future if needed
// Keep buildDag same logic, but update edge style in future if needed
function buildDag(spans) {
  const nodes = []
  const edges = []
  const spanMap = new Map()
  spans.forEach(span => spanMap.set(span.span_id, span))

  // Calculate depth for every span (robust walk-up)
  const levels = new Map()

  spans.forEach(span => {
    let depth = 0
    let parentId = span.parent_span_id
    let attempts = 0
    // Walk up to find root distance
    while (parentId && attempts < 100) {
      depth++
      attempts++
      const parent = spanMap.get(parentId)
      if (!parent) break
      parentId = parent.parent_span_id
    }
    levels.set(span.span_id, depth)
  })

  // Group by level
  const spansByLevel = new Map()
  spans.forEach(span => {
    const level = levels.get(span.span_id) || 0
    if (!spansByLevel.has(level)) spansByLevel.set(level, [])
    spansByLevel.get(level).push(span)
  })

  // Create nodes
  spansByLevel.forEach((levelSpans, level) => {
    // Sort by start_time for logical flow
    levelSpans.sort((a, b) => a.start_time - b.start_time)

    levelSpans.forEach((span, index) => {
      nodes.push({
        id: span.span_id,
        type: 'span',
        position: { x: level * 300, y: index * 120 },
        data: span
      })

      if (span.parent_span_id) {
        edges.push({
          id: `${span.parent_span_id}-${span.span_id}`,
          source: span.parent_span_id,
          target: span.span_id,
          type: 'smoothstep',
          animated: span.status === 'running',
          style: { stroke: '#27272A', strokeWidth: 1.5 },
          markerEnd: { type: MarkerType.ArrowClosed, color: '#27272A' }
        })
      }
    })
  })

  return { nodes, edges }
}

export default TraceDetailPage
