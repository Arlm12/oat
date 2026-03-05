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
    AgentTracer,
    AgentLoop,
    agent_loop,
    get_current_run_id,
    get_current_span_id,
    get_current_trace_id,
    get_tracer,
    inject_context,
    extract_context,
    get_subprocess_env,
    restore_from_env,
    set_run_id,
    set_trace_id,
    span,
    trace,
    trace_llm,
    trace_tool,
    trace_retrieval,
)

from .models import (
    Artifact,
    Run,
    Span,
    SpanKind,
    SpanType,
    SpanStatus,
)

from .storage import StorageEngine

from .exporters import HTTPExporter, ConsoleExporter

__all__ = [
    # Core tracing
    "trace",
    "trace_llm",
    "trace_tool",
    "trace_retrieval",
    "span",
    "get_tracer",
    "AgentTracer",
    "set_run_id",
    "set_trace_id",
    "get_current_run_id",
    "get_current_trace_id",
    "get_current_span_id",
    # Agent loop iteration tracking
    "AgentLoop",
    "agent_loop",
    # Distributed context propagation (multi-service)
    "inject_context",
    "extract_context",
    # Subprocess context propagation
    "get_subprocess_env",
    "restore_from_env",
    # Models
    "Run",
    "Span",
    "Artifact",
    "SpanKind",
    "SpanType",
    "SpanStatus",
    # Storage
    "StorageEngine",
    # Exporters
    "HTTPExporter",
    "ConsoleExporter",
]
