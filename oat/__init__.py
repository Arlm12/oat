"""
OpenAgentTrace (OAT) - The Open Standard for AI Agent Observability

Usage:
    from oat import trace, span, AgentTracer
    
    @trace(span_type="llm")
    async def call_llm(prompt: str):
        ...

    # Or use context manager
    with span("tool_execution", span_type="tool"):
        result = execute_tool()
"""

__version__ = "0.1.0"

from .tracer import (
    trace,
    span,
    get_tracer,
    AgentTracer,
    set_trace_id,
    get_current_trace_id,
    get_current_span_id,
    # Specialized decorators
    trace_llm,
    trace_tool,
    trace_retrieval,
    trace_agent,
    trace_guardrail,
    trace_memory,
    trace_http,
    trace_chain,
    trace_file_io,
    trace_database,
    trace_cache,
)

from .models import (
    Span,
    SpanType,
    SpanStatus,
)

from .storage import StorageEngine

from .exporters import HTTPExporter, ConsoleExporter

from .media import (
    MediaMetadata,
    analyze_image,
    analyze_audio,
    analyze_video,
    analyze_media,
    estimate_image_tokens,
)

__all__ = [
    # Core tracing
    "trace",
    "span", 
    "get_tracer",
    "AgentTracer",
    "set_trace_id",
    "get_current_trace_id",
    "get_current_span_id",
    # Specialized decorators
    "trace_llm",
    "trace_tool",
    "trace_retrieval",
    "trace_agent",
    "trace_guardrail",
    "trace_memory",
    "trace_http",
    "trace_chain",
    "trace_file_io",
    "trace_database",
    "trace_cache",
    # Models
    "Span",
    "SpanType", 
    "SpanStatus",
    # Storage
    "StorageEngine",
    # Exporters
    "HTTPExporter",
    "ConsoleExporter",
    # Media analysis
    "MediaMetadata",
    "analyze_image",
    "analyze_audio",
    "analyze_video",
    "analyze_media",
    "estimate_image_tokens",
]

