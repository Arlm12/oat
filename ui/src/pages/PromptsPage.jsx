import React, { useState, useEffect } from 'react'
import { API_URL } from '../App'
import { AlertCircle, Clock, DollarSign, Terminal, Copy, Check } from 'lucide-react'

export default function PromptsPage() {
  const [prompts, setPrompts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(null)

  useEffect(() => {
    fetch(`${API_URL}/v1/analytics/prompts`)
      .then(res => res.json())
      .then(data => { setPrompts(data); setLoading(false) })
      .catch(err => { setError('Failed to load prompts'); setLoading(false) })
  }, [])

  const copyPrompt = (text, idx) => {
    navigator.clipboard.writeText(text)
    setCopied(idx)
    setTimeout(() => setCopied(null), 1500)
  }

  if (loading) return (
    <div style={{ padding: 24, color: 'var(--text-tertiary)', fontSize: 13 }}>
      Loading prompts...
    </div>
  )

  return (
    <div style={{ background: 'var(--bg-primary)', minHeight: '100%' }}>

      <div className="page-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
            <Terminal style={{ width: 14, height: 14, color: 'var(--accent)' }} />
            <span className="page-title">Prompt Registry</span>
          </div>
          <div className="page-subtitle">Canonical input artifacts sent to your LLM spans</div>
        </div>
        <div className="badge badge-neutral">{prompts.length} prompts</div>
      </div>

      <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 8 }}>

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

        {prompts.map((p, i) => (
          <div
            key={i}
            className="card"
            style={{ padding: '14px 16px', transition: 'border-color 0.15s' }}
          >
            {/* Header row */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                {/* Call count */}
                <span className="badge badge-neutral">{p.count} calls</span>

                {/* Error badge */}
                {p.error_rate > 0 && (
                  <span className="badge badge-error">
                    <AlertCircle style={{ width: 9, height: 9 }} />
                    {p.error_rate}% errors
                  </span>
                )}

                {/* Last used */}
                <span style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>
                  Last used: {p.last_used ? new Date(p.last_used * 1000).toLocaleString() : 'Never'}
                </span>
              </div>

              {/* Metrics */}
              <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                  <Clock style={{ width: 11, height: 11 }} />
                  {p.avg_latency}ms
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--green)', fontFamily: 'monospace' }}>
                  <DollarSign style={{ width: 11, height: 11 }} />
                  ${p.avg_cost}
                </div>
              </div>
            </div>

            {/* Prompt text */}
            <div style={{
              background: 'var(--bg-raised)',
              border: '1px solid var(--border-faint)',
              borderRadius: 7,
              padding: '10px 12px',
              position: 'relative',
              overflow: 'hidden',
            }}>
              <pre style={{
                fontSize: 12,
                fontFamily: 'monospace',
                color: 'var(--text-secondary)',
                whiteSpace: 'pre-wrap',
                overflow: 'hidden',
                display: '-webkit-box',
                WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical',
                margin: 0,
                lineHeight: 1.6,
              }}>
                {p.prompt_text}
              </pre>
              <div style={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                height: 28,
                background: 'linear-gradient(to top, var(--bg-raised), transparent)',
              }} />
            </div>

            {/* Footer */}
            <div style={{ marginTop: 8, display: 'flex', justifyContent: 'flex-end' }}>
              <button
                onClick={() => copyPrompt(p.prompt_text, i)}
                className="btn btn-ghost"
                style={{ height: 26, padding: '0 10px', fontSize: 11, gap: 4 }}
              >
                {copied === i
                  ? <><Check style={{ width: 11, height: 11 }} /> Copied</>
                  : <><Copy style={{ width: 11, height: 11 }} /> Copy</>
                }
              </button>
            </div>
          </div>
        ))}

        {prompts.length === 0 && !error && (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            padding: '60px 0',
            gap: 10,
          }}>
            <Terminal style={{ width: 28, height: 28, color: 'var(--text-disabled)' }} />
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>No prompts recorded yet</span>
          </div>
        )}
      </div>
    </div>
  )
}
