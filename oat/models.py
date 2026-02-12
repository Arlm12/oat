"""
Core data models for OpenAgentTrace.
Defines the span schema that is the foundation of agent observability.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import time


class SpanType(str, Enum):
    """Semantic span types for agent observability."""
    # Core agent operations
    AGENT = "agent"           # Root agent execution
    LLM = "llm"               # LLM inference call
    TOOL = "tool"             # Tool/function execution
    RETRIEVAL = "retrieval"   # RAG/vector search
    GUARDRAIL = "guardrail"   # Safety/validation check
    HANDOFF = "handoff"       # Agent-to-agent handoff
    
    # Supporting operations
    CHAIN = "chain"           # Chain of operations
    EMBEDDING = "embedding"   # Embedding generation
    RERANK = "rerank"         # Reranking operation
    MEMORY = "memory"         # Memory read/write
    PROMPT_TEMPLATE = "prompt_template"  # Template rendering
    RESPONSE_VALIDATION = "validation"   # Output validation
    
    # Infrastructure
    HTTP = "http"             # HTTP request
    DATABASE = "database"     # Database query
    CACHE = "cache"           # Cache operation
    FILE_IO = "file_io"       # File operations
    WEB_REQUEST = "web_request"  # External API calls
    LOGGING = "logging"       # Log-based traces
    
    # Generic
    FUNCTION = "function"     # Generic function
    CUSTOM = "custom"         # User-defined


class SpanStatus(str, Enum):
    """Span execution status."""
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """Distributed tracing context for propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for distributed tracing."""
        headers = {
            "x-oat-trace-id": self.trace_id,
            "x-oat-span-id": self.span_id,
        }
        if self.parent_span_id:
            headers["x-oat-parent-span-id"] = self.parent_span_id
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        """Extract context from HTTP headers."""
        trace_id = headers.get("x-oat-trace-id")
        span_id = headers.get("x-oat-span-id")
        if trace_id and span_id:
            return cls(
                trace_id=trace_id,
                span_id=span_id,
                parent_span_id=headers.get("x-oat-parent-span-id"),
            )
        return None


@dataclass
class LLMUsage:
    """Token usage for LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost tracking (in USD)
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class Span:
    """
    The core unit of observability in OpenAgentTrace.
    
    A span represents a single operation within an agent's execution,
    whether it's an LLM call, tool execution, retrieval, or guardrail check.
    """
    # Identity
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    
    # Descriptors
    name: str = ""
    span_type: SpanType = SpanType.FUNCTION
    
    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Status
    status: SpanStatus = SpanStatus.RUNNING
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Payload references (hashes for blob storage)
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    
    # Inline small payloads for quick access
    input_preview: Optional[str] = None   # First 500 chars
    output_preview: Optional[str] = None  # First 500 chars
    
    # LLM-specific
    model: Optional[str] = None
    service_provider: Optional[str] = None  # e.g., "openai", "anthropic"
    usage: Optional[LLMUsage] = None
    service_name: str = "default_agent"

    # Tool-specific
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    
    # Retrieval-specific
    retrieval_query: Optional[str] = None
    retrieval_k: Optional[int] = None
    retrieval_scores: Optional[List[float]] = None
    
    # Guardrail-specific
    guardrail_triggered: Optional[bool] = None
    guardrail_action: Optional[str] = None  # "block", "warn", "pass"
    
    # Multimodal inputs/outputs
    media_inputs: List[Dict[str, Any]] = field(default_factory=list)
    media_outputs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Scoring/feedback (populated later)
    score: Optional[int] = None           # -1 to 1
    feedback: Optional[str] = None
    
    def finish(self, status: SpanStatus = SpanStatus.SUCCESS, error: Optional[Exception] = None):
        """Mark span as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if error:
            self.status = SpanStatus.ERROR
            self.error_message = str(error)
            self.error_type = type(error).__name__
        else:
            self.status = status
    
    def set_input(self, data: Any, preview_len: int = 500):
        """Store input data reference and preview."""
        try:
            import json
            text = json.dumps(data, default=str)
            self.input_preview = text[:preview_len] if len(text) > preview_len else text
        except:
            self.input_preview = str(data)[:preview_len]
    
    def set_output(self, data: Any, preview_len: int = 500):
        """Store output data reference and preview."""
        try:
            import json
            text = json.dumps(data, default=str)
            self.output_preview = text[:preview_len] if len(text) > preview_len else text
        except:
            self.output_preview = str(data)[:preview_len]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value if isinstance(self.span_type, SpanType) else self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value if isinstance(self.status, SpanStatus) else self.status,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
            "model": self.model,
            "service_provider": self.service_provider,
            "service_name": self.service_name,
            "usage": self.usage.to_dict() if self.usage else None,
            "tool_name": self.tool_name,
            "tool_parameters": self.tool_parameters,
            "retrieval_query": self.retrieval_query,
            "retrieval_k": self.retrieval_k,
            "retrieval_scores": self.retrieval_scores,
            "guardrail_triggered": self.guardrail_triggered,
            "guardrail_action": self.guardrail_action,
            "media_inputs": self.media_inputs,
            "media_outputs": self.media_outputs,
            "metadata": self.metadata,
            "tags": self.tags,
            "score": self.score,
            "feedback": self.feedback,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """Create span from dictionary."""
        usage_data = data.pop("usage", None)
        usage = LLMUsage(**usage_data) if usage_data else None
        
        span_type = data.pop("span_type", "function")
        if isinstance(span_type, str):
            try:
                span_type = SpanType(span_type)
            except ValueError:
                span_type = SpanType.CUSTOM
        
        status = data.pop("status", "running")
        if isinstance(status, str):
            try:
                status = SpanStatus(status)
            except ValueError:
                status = SpanStatus.RUNNING
        
        return cls(
            span_type=span_type,
            status=status,
            usage=usage,
            **data
        )


@dataclass
class Trace:
    """A collection of spans forming a complete agent execution."""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    
    # Computed fields
    root_span: Optional[Span] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.RUNNING
    
    # Aggregations
    total_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    error_count: int = 0
    
    def compute_metrics(self):
        """Calculate aggregate metrics from spans."""
        if not self.spans:
            return
            
        # Find root span
        for span in self.spans:
            if span.parent_span_id is None:
                self.root_span = span
                break
        
        # Time bounds
        self.start_time = min(s.start_time for s in self.spans)
        end_times = [s.end_time for s in self.spans if s.end_time]
        if end_times:
            self.end_time = max(end_times)
            self.duration_ms = (self.end_time - self.start_time) * 1000
        
        # Aggregations
        for span in self.spans:
            if span.usage:
                self.total_tokens += span.usage.total_tokens
                self.total_cost += span.usage.total_cost
            
            if span.span_type == SpanType.LLM:
                self.llm_calls += 1
            elif span.span_type == SpanType.TOOL:
                self.tool_calls += 1
            
            if span.status == SpanStatus.ERROR:
                self.error_count += 1
        
        # Overall status
        if self.error_count > 0:
            self.status = SpanStatus.ERROR
        elif self.root_span and self.root_span.status == SpanStatus.SUCCESS:
            self.status = SpanStatus.SUCCESS
        elif all(s.status == SpanStatus.SUCCESS for s in self.spans if s.end_time):
            self.status = SpanStatus.SUCCESS
