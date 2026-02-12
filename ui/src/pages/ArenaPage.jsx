import React, { useState, useEffect } from 'react'
import { API_URL } from '../App'
import { Play, Cpu, Plus, X, AlertTriangle } from 'lucide-react'

const COLORS = [
    { text: 'text-blue-400', border: 'border-blue-500/30', bg: 'bg-blue-500/10' },
    { text: 'text-purple-400', border: 'border-purple-500/30', bg: 'bg-purple-500/10' },
    { text: 'text-emerald-400', border: 'border-emerald-500/30', bg: 'bg-emerald-500/10' },
    { text: 'text-amber-400', border: 'border-amber-500/30', bg: 'bg-amber-500/10' },
]

export default function ArenaPage() {
    const [prompt, setPrompt] = useState("Explain the concept of 'Agency' in AI to a non-technical person.")
    const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant.")

    // Dynamic model panels (2-4)
    const [panels, setPanels] = useState([
        { id: 1, model: 'gpt-4o-mini', result: null },
        { id: 2, model: 'gpt-4o', result: null },
    ])

    const [availableModels, setAvailableModels] = useState([])
    const [warning, setWarning] = useState(null)
    const [loading, setLoading] = useState(false)

    // Fetch available models on mount
    useEffect(() => {
        fetch(`${API_URL}/arena/models`)
            .then(res => res.json())
            .then(data => {
                setAvailableModels(data.models || [])
                if (data.warning) setWarning(data.warning)
            })
            .catch(() => setWarning("Could not fetch models. Is the server running?"))
    }, [])

    const addPanel = () => {
        if (panels.length >= 4) return
        const nextModel = availableModels.find(m =>
            m.available && !panels.some(p => p.model === m.id)
        )
        setPanels([...panels, {
            id: Date.now(),
            model: nextModel?.id || availableModels[0]?.id || 'gpt-4o-mini',
            result: null
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
        setLoading(true)
        // Clear all results
        setPanels(panels.map(p => ({ ...p, result: null })))

        const runModel = async (panel) => {
            try {
                const res = await fetch(`${API_URL}/arena/run`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        system_prompt: systemPrompt,
                        model: panel.model
                    })
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

    // Calculate dynamic grid columns
    const gridCols = panels.length === 2 ? 'grid-cols-2'
        : panels.length === 3 ? 'grid-cols-3'
            : 'grid-cols-4'

    return (
        <div className="p-8 h-full flex flex-col max-w-[1900px] mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white">Model Comparison Arena</h2>
                    <p className="text-[#A1A1AA]">Battle test different models on the same prompt.</p>
                </div>
                <div className="flex items-center gap-3">
                    {panels.length < 4 && (
                        <button
                            onClick={addPanel}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-[#27272A] text-white hover:bg-[#3F3F46] transition-colors"
                        >
                            <Plus className="w-4 h-4" /> Add Model
                        </button>
                    )}
                    <button
                        onClick={handleRun}
                        disabled={loading || availableModels.length === 0}
                        className={`flex items-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all ${loading ? 'bg-[#27272A] text-[#A1A1AA]' : 'bg-white text-black hover:bg-gray-200'
                            }`}
                    >
                        {loading ? 'Running...' : <><Play className="w-4 h-4" /> Run Battle</>}
                    </button>
                </div>
            </div>

            {/* Warning Banner */}
            {warning && (
                <div className="mb-4 p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 text-amber-400" />
                    <span className="text-amber-200 text-sm">{warning}</span>
                </div>
            )}

            {/* Main Content */}
            <div className="flex gap-6 flex-1 min-h-0">
                {/* Input Column */}
                <div className="w-72 flex-shrink-0 flex flex-col gap-4">
                    <div className="flex-1 flex flex-col">
                        <label className="text-xs font-medium text-[#A1A1AA] uppercase tracking-wider mb-2">
                            System Prompt
                        </label>
                        <textarea
                            value={systemPrompt}
                            onChange={(e) => setSystemPrompt(e.target.value)}
                            className="w-full bg-[#151518] border border-[#27272A] rounded-lg p-4 text-white font-mono text-sm focus:outline-none focus:border-[#8B5CF6] h-28 resize-none mb-4"
                        />

                        <label className="text-xs font-medium text-[#A1A1AA] uppercase tracking-wider mb-2">
                            User Prompt
                        </label>
                        <textarea
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            className="w-full bg-[#151518] border border-[#27272A] rounded-lg p-4 text-white font-mono text-sm focus:outline-none focus:border-[#8B5CF6] flex-1 resize-none"
                        />
                    </div>
                </div>

                {/* Model Panels Grid */}
                <div className={`flex-1 grid ${gridCols} gap-4 min-w-0`}>
                    {panels.map((panel, idx) => (
                        <ModelPanel
                            key={panel.id}
                            panel={panel}
                            color={COLORS[idx % COLORS.length]}
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

function ModelPanel({ panel, color, models, canRemove, onModelChange, onRemove }) {
    const { model, result } = panel
    const modelInfo = models.find(m => m.id === model)

    return (
        <div className={`flex flex-col h-full bg-[#0C0C0E] border rounded-xl overflow-hidden ${result?.status === 'success' ? color.border : 'border-[#27272A]'
            }`}>
            {/* Header */}
            <div className="p-3 border-b border-[#27272A] flex justify-between items-center bg-[#151518]">
                <div className="flex items-center gap-2 flex-1 min-w-0">
                    <Cpu className={`w-4 h-4 flex-shrink-0 ${color.text}`} />
                    <select
                        value={model}
                        onChange={(e) => onModelChange(e.target.value)}
                        className="bg-transparent text-white text-sm font-medium focus:outline-none cursor-pointer truncate flex-1"
                    >
                        {models.length > 0 ? (
                            models.map(m => (
                                <option key={m.id} value={m.id} disabled={!m.available}>
                                    {m.name} {!m.available ? '(No API Key)' : ''}
                                </option>
                            ))
                        ) : (
                            <option value={model}>{model}</option>
                        )}
                    </select>
                </div>
                <div className="flex items-center gap-2">
                    {result?.status === 'success' && (
                        <span className="text-xs font-mono text-[#A1A1AA]">
                            {Math.round(result.duration_ms)}ms
                        </span>
                    )}
                    {canRemove && (
                        <button
                            onClick={onRemove}
                            className="p-1 rounded hover:bg-[#27272A] text-[#52525B] hover:text-white transition-colors"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    )}
                </div>
            </div>

            {/* Provider Badge */}
            {modelInfo?.provider && (
                <div className="px-3 py-1.5 bg-[#151518] border-b border-[#27272A]">
                    <span className={`text-xs font-medium px-2 py-0.5 rounded ${color.bg} ${color.text}`}>
                        {modelInfo.provider}
                    </span>
                </div>
            )}

            {/* Content */}
            <div className="flex-1 p-4 overflow-auto">
                {result ? (
                    result.status === 'error' ? (
                        <div className="text-red-400 text-sm font-mono bg-red-500/10 p-3 rounded-lg">
                            {result.error}
                        </div>
                    ) : (
                        <div className="prose prose-invert prose-sm max-w-none">
                            <p className="whitespace-pre-wrap text-[#D4D4D8] leading-relaxed text-sm">
                                {result.output}
                            </p>
                        </div>
                    )
                ) : (
                    <div className="h-full flex items-center justify-center text-[#3F3F46] text-sm">
                        Waiting for run...
                    </div>
                )}
            </div>

            {/* Footer with Usage */}
            {result?.status === 'success' && result.usage && (
                <div className="px-3 py-2 border-t border-[#27272A] bg-[#151518] flex items-center gap-4 text-xs font-mono text-[#71717A]">
                    {result.usage.prompt_tokens !== undefined && (
                        <span>In: {result.usage.prompt_tokens || result.usage.input_tokens}</span>
                    )}
                    {result.usage.completion_tokens !== undefined && (
                        <span>Out: {result.usage.completion_tokens || result.usage.output_tokens}</span>
                    )}
                </div>
            )}
        </div>
    )
}