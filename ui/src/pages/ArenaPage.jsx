import React, { useState, useEffect } from 'react'
import { API_URL } from '../App'
import { Play, Cpu, Plus, X, AlertTriangle } from 'lucide-react'

const MODEL_ACCENT = [
  { text: '#2563EB', bg: 'rgba(37,99,235,0.07)', border: 'rgba(37,99,235,0.15)' },
  { text: '#7C3AED', bg: 'rgba(124,58,237,0.07)', border: 'rgba(124,58,237,0.15)' },
  { text: '#1A8A5A', bg: 'rgba(26,138,90,0.07)', border: 'rgba(26,138,90,0.15)' },
  { text: '#C07A00', bg: 'rgba(192,122,0,0.07)', border: 'rgba(192,122,0,0.15)' },
]

export default function ArenaPage() {
  const [prompt, setPrompt] = useState("Explain the concept of 'Agency' in AI to a non-technical person.")
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.")
  const [panels, setPanels] = useState([
    { id: 1, model: '', result: null },
    { id: 2, model: '', result: null },
  ])
  const [availableModels, setAvailableModels] = useState([])
  const [warning, setWarning] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${API_URL}/arena/models`)
      .then(res => res.json())
      .then(data => {
        const models = data.models || []
        setAvailableModels(models)
        setPanels(prev => {
          const available = models.filter(m => m.available)
          return prev.map((panel, idx) => {
            const stillValid = panel.model && models.some(m => m.id === panel.model)
            if (stillValid) return panel
            const fallback = available[idx]?.id || models[idx]?.id || available[0]?.id || models[0]?.id || ''
            return { ...panel, model: fallback }
          })
        })
        if (data.warning) setWarning(data.warning)
      })
      .catch(() => setWarning('Could not fetch models. Is the server running?'))
  }, [])

  const addPanel = () => {
    if (panels.length >= 4) return
    const nextModel = availableModels.find(m => m.available && !panels.some(p => p.model === m.id))
    setPanels([...panels, {
      id: Date.now(),
      model: nextModel?.id || availableModels[0]?.id || '',
      result: null,
    }])
  }

  const removePanel = (id) => {
    if (panels.length <= 2) return
    setPanels(panels.filter(p => p.id !== id))
  }

  const updatePanelModel = (id, model) => {
    setPanels(panels.map(p => p.id === id ? { ...p, model, result: null } : p))
  }

  const handleRun = async () => {
    const runnable = panels.filter(p => p.model)
    if (runnable.length === 0) {
      setWarning('Select at least one model before running.')
      return
    }
    setLoading(true)
    setPanels(panels.map(p => ({ ...p, result: null })))

    const runModel = async (panel) => {
      try {
        const res = await fetch(`${API_URL}/arena/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, system_prompt: systemPrompt, model: panel.model }),
        })
        return await res.json()
      } catch (e) {
        return { status: 'error', error: e.message }
      }
    }

    const results = await Promise.all(panels.map(runModel))
    setPanels(panels.map((p, i) => ({ ...p, result: results[i] })))
    setLoading(false)
  }

  const gridTemplate = panels.length === 2 ? '1fr 1fr'
    : panels.length === 3 ? '1fr 1fr 1fr'
    : '1fr 1fr 1fr 1fr'

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-primary)' }}>

      {/* Header */}
      <div className="page-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <span style={{ fontSize: 14, color: 'var(--accent)' }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14.5 10c-.83 0-1.5-.67-1.5-1.5v-5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5z"/>
                <path d="M20.5 10H19V8.5c0-.83.67-1.5 1.5-1.5s1.5.67 1.5 1.5-.67 1.5-1.5 1.5z"/>
                <path d="M9.5 14c.83 0 1.5.67 1.5 1.5v5c0 .83-.67 1.5-1.5 1.5S8 21.33 8 20.5v-5c0-.83.67-1.5 1.5-1.5z"/>
                <path d="M3.5 14H5v1.5c0 .83-.67 1.5-1.5 1.5S2 16.33 2 15.5 2.67 14 3.5 14z"/>
                <path d="M14 14.5c0-.83.67-1.5 1.5-1.5h5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5h-5c-.83 0-1.5-.67-1.5-1.5z"/>
                <path d="M15.5 19H14v1.5c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5z"/>
                <path d="M10 9.5C10 8.67 9.33 8 8.5 8h-5C2.67 8 2 8.67 2 9.5S2.67 11 3.5 11h5c.83 0 1.5-.67 1.5-1.5z"/>
                <path d="M8.5 5H10V3.5C10 2.67 9.33 2 8.5 2S7 2.67 7 3.5 7.67 5 8.5 5z"/>
              </svg>
            </span>
            <span className="page-title">Model Arena</span>
          </div>
          <div className="page-subtitle">Compare model outputs side by side on the same prompt</div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {panels.length < 4 && (
            <button onClick={addPanel} className="btn btn-secondary">
              <Plus style={{ width: 13, height: 13 }} />
              Add Model
            </button>
          )}
          <button
            onClick={handleRun}
            disabled={loading || !availableModels.some(m => m.available)}
            className="btn btn-primary"
            style={{ gap: 6 }}
          >
            {loading
              ? 'Running...'
              : <><Play style={{ width: 12, height: 12 }} /> Run</>
            }
          </button>
        </div>
      </div>

      {/* Warning */}
      {warning && (
        <div style={{
          margin: '0 20px',
          marginTop: 12,
          padding: '8px 12px',
          background: 'var(--amber-bg)',
          border: '1px solid rgba(192,122,0,0.15)',
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: 12,
          color: 'var(--amber)',
        }}>
          <AlertTriangle style={{ width: 12, height: 12, flexShrink: 0 }} />
          {warning}
        </div>
      )}

      {/* Content */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', gap: 12, padding: '12px 20px 20px' }}>

        {/* Input column */}
        <div style={{ width: 260, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>

          <div style={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 5 }}>
              System Prompt
            </label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-default)',
                borderRadius: 8,
                padding: '8px 10px',
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--text-primary)',
                resize: 'none',
                outline: 'none',
                height: 90,
                marginBottom: 10,
                lineHeight: 1.6,
              }}
              onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-ring)' }}
              onBlur={e => { e.target.style.borderColor = 'var(--border-default)'; e.target.style.boxShadow = 'none' }}
            />

            <label style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 5 }}>
              User Prompt
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-default)',
                borderRadius: 8,
                padding: '8px 10px',
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--text-primary)',
                resize: 'none',
                outline: 'none',
                flex: 1,
                lineHeight: 1.6,
              }}
              onFocus={e => { e.target.style.borderColor = 'var(--accent)'; e.target.style.boxShadow = '0 0 0 3px var(--accent-ring)' }}
              onBlur={e => { e.target.style.borderColor = 'var(--border-default)'; e.target.style.boxShadow = 'none' }}
            />
          </div>
        </div>

        {/* Model panels */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: gridTemplate, gap: 10, minWidth: 0 }}>
          {panels.map((panel, idx) => (
            <ModelPanel
              key={panel.id}
              panel={panel}
              accent={MODEL_ACCENT[idx % MODEL_ACCENT.length]}
              models={availableModels}
              canRemove={panels.length > 2}
              onModelChange={(model) => updatePanelModel(panel.id, model)}
              onRemove={() => removePanel(panel.id)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function ModelPanel({ panel, accent, models, canRemove, onModelChange, onRemove }) {
  const { model, result } = panel
  const modelInfo = models.find(m => m.id === model)
  const usage = result?.usage || {}
  const inTokens = usage.prompt_tokens ?? usage.input_tokens
  const outTokens = usage.completion_tokens ?? usage.output_tokens
  const totalCost = Number(usage.total_cost || 0)

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg-surface)',
      border: `1px solid ${result?.status === 'success' ? accent.border : 'var(--border-subtle)'}`,
      borderRadius: 10,
      overflow: 'hidden',
      boxShadow: 'var(--shadow-sm)',
    }}>
      {/* Model header */}
      <div style={{
        padding: '10px 12px',
        borderBottom: '1px solid var(--border-faint)',
        background: 'var(--bg-raised)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 8,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, flex: 1, minWidth: 0 }}>
          <Cpu style={{ width: 12, height: 12, color: accent.text, flexShrink: 0 }} />
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            style={{
              background: 'transparent',
              border: 'none',
              outline: 'none',
              fontSize: 12,
              fontWeight: 500,
              color: 'var(--text-primary)',
              cursor: 'pointer',
              flex: 1,
              minWidth: 0,
              fontFamily: 'inherit',
            }}
          >
            {models.length > 0
              ? models.map(m => (
                  <option key={m.id} value={m.id} disabled={!m.available}>
                    {m.name} {!m.available ? '(No API Key)' : ''}
                  </option>
                ))
              : <option value={model}>{model}</option>
            }
          </select>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {result?.status === 'success' && (
            <span style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--text-tertiary)' }}>
              {Math.round(result.duration_ms)}ms
            </span>
          )}
          {modelInfo?.provider && (
            <span style={{
              fontSize: 10, fontWeight: 500,
              padding: '1px 6px',
              borderRadius: 4,
              background: accent.bg,
              color: accent.text,
              border: `1px solid ${accent.border}`,
            }}>
              {modelInfo.provider}
            </span>
          )}
          {canRemove && (
            <button
              onClick={onRemove}
              className="btn btn-ghost"
              style={{ width: 24, height: 24, padding: 0, borderRadius: 5 }}
            >
              <X style={{ width: 11, height: 11 }} />
            </button>
          )}
        </div>
      </div>

      {/* Output */}
      <div style={{ flex: 1, padding: '12px', overflowY: 'auto' }}>
        {result ? (
          result.status === 'error' ? (
            <div style={{
              padding: '8px 10px',
              background: 'var(--red-bg)',
              border: '1px solid rgba(217,48,37,0.15)',
              borderRadius: 7,
              fontSize: 12,
              color: 'var(--red)',
              fontFamily: 'monospace',
            }}>
              {result.error}
            </div>
          ) : (
            <p style={{
              fontSize: 13,
              lineHeight: 1.65,
              color: 'var(--text-primary)',
              whiteSpace: 'pre-wrap',
              margin: 0,
            }}>
              {result.output}
            </p>
          )
        ) : (
          <div style={{
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 12,
            color: 'var(--text-disabled)',
          }}>
            Waiting for run...
          </div>
        )}
      </div>

      {/* Token usage footer */}
      {result?.status === 'success' && result.usage && (
        <div style={{
          padding: '6px 12px',
          borderTop: '1px solid var(--border-faint)',
          background: 'var(--bg-raised)',
          display: 'flex',
          gap: 12,
          fontSize: 11,
          fontFamily: 'monospace',
          color: 'var(--text-tertiary)',
        }}>
          {inTokens !== undefined && (
            <span>In: {inTokens}</span>
          )}
          {outTokens !== undefined && (
            <span>Out: {outTokens}</span>
          )}
          {totalCost > 0 && (
            <span>Cost: ${totalCost.toFixed(6)}</span>
          )}
          {usage.pricing_status && (
            <span>{usage.pricing_status === 'known' ? 'Pricing: known' : 'Pricing: unknown'}</span>
          )}
        </div>
      )}
    </div>
  )
}
