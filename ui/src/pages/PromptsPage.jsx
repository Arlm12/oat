import React, { useState, useEffect } from 'react'
import { API_URL } from '../App'
import { AlertCircle, Clock, DollarSign, Terminal, Copy } from 'lucide-react'

export default function PromptsPage() {
    const [prompts, setPrompts] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch(`${API_URL}/analytics/prompts`)
            .then(res => res.json())
            .then(data => {
                setPrompts(data)
                setLoading(false)
            })
    }, [])

    if (loading) return <div className="p-8 text-[#52525B]">Loading prompts...</div>

    return (
        <div className="p-8 max-w-[1600px] mx-auto">
            <div className="mb-8">
                <h2 className="text-2xl font-bold text-white mb-2">Prompt Registry</h2>
                <p className="text-[#A1A1AA]">Analysis of unique prompts passed to your LLMs.</p>
            </div>

            <div className="grid gap-4">
                {prompts.map((p, i) => (
                    <div key={i} className="card p-6 bg-[#151518] border border-[#27272A] rounded-xl hover:border-[#3F3F46] transition-colors group">
                        <div className="flex justify-between items-start mb-4">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-[#27272A] rounded-lg">
                                    <Terminal className="w-4 h-4 text-[#8B5CF6]" />
                                </div>
                                <div>
                                    <div className="text-sm font-mono text-[#A1A1AA] mb-1">
                                        Last used: {p.last_used ? new Date(p.last_used * 1000).toLocaleString() : 'Never'}
                                    </div>
                                    <div className="text-white font-medium flex items-center gap-2">
                                        <span className="bg-[#27272A] px-2 py-0.5 rounded text-xs text-[#D4D4D8]">{p.count} calls</span>
                                    </div>
                                </div>
                            </div>

                            <div className="flex gap-4 text-sm font-mono">
                                <span className="flex items-center gap-1 text-[#A1A1AA]">
                                    <Clock className="w-3 h-3" /> {p.avg_latency}ms
                                </span>
                                <span className="flex items-center gap-1 text-[#10B981]">
                                    <DollarSign className="w-3 h-3" /> ${p.avg_cost}
                                </span>
                            </div>
                        </div>

                        <div className="bg-[#09090B] rounded-lg p-4 border border-[#27272A] font-mono text-sm text-[#D4D4D8] overflow-hidden relative group-hover:border-[#52525B] transition-colors">
                            <p className="whitespace-pre-wrap line-clamp-3">{p.prompt_text}</p>
                            <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[#09090B] to-transparent" />
                        </div>

                        <div className="mt-3 flex justify-between items-center">
                            {p.error_rate > 0 ? (
                                <div className="flex items-center gap-2 text-[#EF4444] text-xs font-medium bg-red-500/10 px-3 py-1 rounded-full">
                                    <AlertCircle className="w-3 h-3" />
                                    {p.error_rate}% Error Rate
                                </div>
                            ) : <div />}

                            <button
                                onClick={() => navigator.clipboard.writeText(p.prompt_text)}
                                className="text-xs text-[#52525B] hover:text-white flex items-center gap-1"
                            >
                                <Copy className="w-3 h-3" /> Copy Prompt
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}