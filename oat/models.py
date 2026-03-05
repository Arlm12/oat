"""
Canonical telemetry models for OpenAgentTrace.

This module defines the strict Run -> Span -> Artifact contract and keeps
compatibility aliases for legacy trace-centric code paths during migration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time
import uuid


DEFAULT_SCHEMA_VERSION = "2026-03-01"


class SpanKind(str, Enum):
    """Stable semantic span kinds."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    GUARDRAIL = "guardrail"
    HTTP = "http"
    DATABASE = "database"
    FILE = "file"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    CHAIN = "chain"
    CACHE = "cache"
    HANDOFF = "handoff"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    """Run/span lifecycle state."""

    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"   # span/run exceeded its time budget
    SKIPPED = "skipped"   # span/run was intentionally not executed (guardrail short-circuit, etc.)


# Backward-compatible alias used by existing decorators/integrations.
SpanType = SpanKind


@dataclass
class SpanContext:
    """Distributed context for run/span propagation."""

    run_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    workflow_id: Optional[str] = None

    def to_headers(self) -> Dict[str, str]:
        headers = {
            "x-oat-run-id": self.run_id,
            "x-oat-span-id": self.span_id,
        }
        if self.parent_span_id:
            headers["x-oat-parent-span-id"] = self.parent_span_id
        if self.workflow_id:
            headers["x-oat-workflow-id"] = self.workflow_id
        return headers

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional["SpanContext"]:
        run_id = headers.get("x-oat-run-id") or headers.get("x-oat-trace-id")
        span_id = headers.get("x-oat-span-id")
        if not run_id or not span_id:
            return None
        return cls(
            run_id=run_id,
            span_id=span_id,
            parent_span_id=headers.get("x-oat-parent-span-id"),
            workflow_id=headers.get("x-oat-workflow-id"),
        )


@dataclass
class LLMUsage:
    """Normalized token/cost usage payload."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    pricing_status: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Run:
    """Top-level user-visible agent execution."""

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex}")
    service_name: str = "default"
    service_version: Optional[str] = None
    environment: Optional[str] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    root_span_id: Optional[str] = None

    status: SpanStatus = SpanStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_ms: Optional[float] = None

    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    total_tokens: int = 0
    total_cost: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    span_count: int = 0

    def finish(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        output_summary: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        self.ended_at = time.time()
        self.duration_ms = (self.ended_at - self.started_at) * 1000.0
        self.status = status
        if output_summary is not None:
            self.output_summary = output_summary
        if error_message is not None:
            self.error_message = error_message

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class Span:
    """Execution unit within a run."""

    span_id: str = field(default_factory=lambda: f"span_{uuid.uuid4().hex}")
    run_id: str = ""
    parent_span_id: Optional[str] = None
    trace_path: Optional[str] = None

    kind: SpanKind = SpanKind.CUSTOM
    name: str = ""

    status: SpanStatus = SpanStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_ms: Optional[float] = None

    model: Optional[str] = None
    provider: Optional[str] = None
    operation: Optional[str] = None

    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    attributes: Dict[str, Any] = field(default_factory=dict)
    usage: Optional[LLMUsage] = None
    cost: Dict[str, Any] = field(default_factory=dict)

    score: Optional[int] = None
    feedback: Optional[str] = None

    # Legacy convenience fields retained for adapter compatibility.
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None

    def finish(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[Exception] = None,
    ) -> None:
        self.ended_at = time.time()
        self.duration_ms = (self.ended_at - self.started_at) * 1000.0
        if error is not None:
            # Preserve caller's intent (TIMEOUT, CANCELLED, etc.) when status is non-SUCCESS,
            # but default to ERROR when only an exception is given.
            self.status = status if status not in (SpanStatus.SUCCESS, SpanStatus.RUNNING) else SpanStatus.ERROR
            self.error_message = str(error)
            self.error_type = type(error).__name__
        else:
            self.status = status

    def set_input(self, data: Any, preview_len: int = 500) -> None:
        text = _safe_preview_text(data)
        self.input_preview = text[:preview_len]
        self.input_summary = self.input_preview

    def set_output(self, data: Any, preview_len: int = 500) -> None:
        text = _safe_preview_text(data)
        self.output_preview = text[:preview_len]
        self.output_summary = self.output_preview

    @property
    def trace_id(self) -> str:
        """Legacy alias: trace_id maps directly to run_id."""
        return self.run_id

    @trace_id.setter
    def trace_id(self, value: str) -> None:
        self.run_id = value

    @property
    def span_type(self) -> str:
        """Legacy alias: span_type maps to kind."""
        return self.kind.value if isinstance(self.kind, SpanKind) else str(self.kind)

    @span_type.setter
    def span_type(self, value: Any) -> None:
        self.kind = _coerce_kind(value)

    @property
    def start_time(self) -> float:
        return self.started_at

    @start_time.setter
    def start_time(self, value: float) -> None:
        self.started_at = value

    @property
    def end_time(self) -> Optional[float]:
        return self.ended_at

    @end_time.setter
    def end_time(self, value: Optional[float]) -> None:
        self.ended_at = value

    def to_dict(self) -> Dict[str, Any]:
        usage_dict = self.usage.to_dict() if isinstance(self.usage, LLMUsage) else self.usage
        data: Dict[str, Any] = {
            "span_id": self.span_id,
            "run_id": self.run_id,
            "trace_id": self.run_id,
            "parent_span_id": self.parent_span_id,
            "trace_path": self.trace_path,
            "kind": self.kind.value if isinstance(self.kind, SpanKind) else str(self.kind),
            "span_type": self.span_type,
            "name": self.name,
            "status": self.status.value if isinstance(self.status, SpanStatus) else str(self.status),
            "started_at": self.started_at,
            "start_time": self.started_at,
            "ended_at": self.ended_at,
            "end_time": self.ended_at,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "provider": self.provider,
            "operation": self.operation,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "usage": usage_dict,
            "cost": self.cost,
            "score": self.score,
            "feedback": self.feedback,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        payload = dict(data)
        run_id = payload.pop("run_id", payload.pop("trace_id", ""))
        kind_raw = payload.pop("kind", payload.pop("span_type", SpanKind.CUSTOM.value))
        status_raw = payload.pop("status", SpanStatus.RUNNING.value)

        usage_data = payload.pop("usage", None)
        usage = None
        if isinstance(usage_data, dict):
            defaults = LLMUsage().__dict__
            usage = LLMUsage(**{k: usage_data.get(k, defaults[k]) for k in defaults.keys()})

        started = payload.pop("started_at", payload.pop("start_time", time.time()))
        ended = payload.pop("ended_at", payload.pop("end_time", None))

        span = cls(
            span_id=payload.pop("span_id", f"span_{uuid.uuid4().hex}"),
            run_id=run_id,
            parent_span_id=payload.pop("parent_span_id", None),
            trace_path=payload.pop("trace_path", None),
            kind=_coerce_kind(kind_raw),
            name=payload.pop("name", ""),
            status=_coerce_status(status_raw),
            started_at=started,
            ended_at=ended,
            duration_ms=payload.pop("duration_ms", None),
            model=payload.pop("model", None),
            provider=payload.pop("provider", None),
            operation=payload.pop("operation", None),
            input_summary=payload.pop("input_summary", None) or payload.get("input_preview"),
            output_summary=payload.pop("output_summary", None) or payload.get("output_preview"),
            error_type=payload.pop("error_type", None),
            error_message=payload.pop("error_message", None),
            attributes=payload.pop("attributes", payload.pop("metadata", {})) or {},
            usage=usage,
            cost=payload.pop("cost", {}) or {},
            score=payload.pop("score", None),
            feedback=payload.pop("feedback", None),
            input_hash=payload.pop("input_hash", None),
            output_hash=payload.pop("output_hash", None),
            input_preview=payload.pop("input_preview", None),
            output_preview=payload.pop("output_preview", None),
        )
        return span


@dataclass
class Artifact:
    """Large or structured payload attached to a span."""

    artifact_id: str = field(default_factory=lambda: f"art_{uuid.uuid4().hex}")
    run_id: str = ""
    span_id: str = ""
    role: str = "output.message"
    content_type: str = "application/json"
    storage_uri: Optional[str] = None
    inline_text: Optional[str] = None
    preview: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "role": self.role,
            "content_type": self.content_type,
            "storage_uri": self.storage_uri,
            "inline_text": self.inline_text,
            "preview": self.preview,
            "metadata": self.metadata,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "created_at": self.created_at,
        }


@dataclass
class EventEnvelope:
    """Versioned canonical event envelope used by ingestion/export."""

    event_type: str
    run_id: str
    payload: Dict[str, Any]
    schema_version: str = DEFAULT_SCHEMA_VERSION
    emitted_at: float = field(default_factory=time.time)
    span_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "emitted_at": self.emitted_at,
            "run_id": self.run_id,
            "span_id": self.span_id,
            "payload": self.payload,
        }


@dataclass
class Trace:
    """Legacy container retained for older API surfaces."""

    trace_id: str
    spans: List[Span] = field(default_factory=list)

    def compute_metrics(self) -> Dict[str, Any]:
        if not self.spans:
            return {
                "trace_id": self.trace_id,
                "span_count": 0,
                "status": SpanStatus.RUNNING.value,
                "duration_ms": 0.0,
            }

        starts = [s.started_at for s in self.spans]
        ends = [s.ended_at for s in self.spans if s.ended_at is not None]
        error_count = sum(1 for s in self.spans if s.status == SpanStatus.ERROR)
        running_count = sum(1 for s in self.spans if s.status == SpanStatus.RUNNING)

        if error_count > 0:
            status = SpanStatus.ERROR.value
        elif running_count > 0:
            status = SpanStatus.RUNNING.value
        else:
            status = SpanStatus.SUCCESS.value

        duration_ms = 0.0
        if ends:
            duration_ms = (max(ends) - min(starts)) * 1000.0

        return {
            "trace_id": self.trace_id,
            "span_count": len(self.spans),
            "status": status,
            "duration_ms": duration_ms,
        }


def _coerce_kind(value: Any) -> SpanKind:
    if isinstance(value, SpanKind):
        return value
    if isinstance(value, str):
        try:
            return SpanKind(value)
        except ValueError:
            return SpanKind.CUSTOM
    return SpanKind.CUSTOM


def _coerce_status(value: Any) -> SpanStatus:
    if isinstance(value, SpanStatus):
        return value
    if isinstance(value, str):
        try:
            return SpanStatus(value)
        except ValueError:
            return SpanStatus.RUNNING
    return SpanStatus.RUNNING


def _safe_preview_text(data: Any) -> str:
    try:
        import json

        if isinstance(data, (str, bytes)):
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return data
        return json.dumps(data, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(data)
