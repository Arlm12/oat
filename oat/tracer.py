"""
Canonical tracer for OpenAgentTrace.

Primary model:
- explicit run lifecycle
- explicit span lifecycle
- artifact-first multimodal capture
"""

from __future__ import annotations

import atexit
import asyncio
import contextlib
import contextvars
import functools
import os
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

# Batch-flush configuration for the background worker.
_BATCH_MAX_SIZE = 20         # flush when this many events accumulate
_BATCH_MAX_AGE_S = 0.05     # flush at least every 50 ms

# ── Distributed context propagation ─────────────────────────────────────────
# HTTP headers used to carry OAT context across service boundaries.
_HEADER_RUN_ID = "x-oat-run-id"
_HEADER_SPAN_ID = "x-oat-span-id"
_HEADER_WORKFLOW_ID = "x-oat-workflow-id"

# Environment variable names used to carry OAT context into child processes.
_ENV_RUN_ID = "OAT_PROPAGATED_RUN_ID"
_ENV_SPAN_ID = "OAT_PROPAGATED_SPAN_ID"

from .models import Artifact, DEFAULT_SCHEMA_VERSION, EventEnvelope, LLMUsage, Run, Span, SpanKind, SpanStatus, SpanType
from .storage import StorageEngine, get_storage


# Context variables
# NOTE: _span_stack uses default=None (not []) to avoid a single shared list
# being mutated across all contexts that have never explicitly set their own value.
_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("oat_run_id", default=None)
_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("oat_span_id", default=None)
_span_stack: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar("oat_span_stack", default=None)


def _get_span_stack() -> List[str]:
    """Return the current context's span stack, initializing lazily."""
    stack = _span_stack.get()
    if stack is None:
        stack = []
        _span_stack.set(stack)
    return stack


def get_current_run_id() -> Optional[str]:
    return _run_id.get()


def get_current_trace_id() -> Optional[str]:
    # legacy alias
    return _run_id.get()


def get_current_span_id() -> Optional[str]:
    return _span_id.get()


def set_run_id(run_id: str) -> contextvars.Token:
    return _run_id.set(run_id)


def set_trace_id(trace_id: str) -> contextvars.Token:
    # legacy alias
    return _run_id.set(trace_id)


class AgentTracer:
    def __init__(
        self,
        service_name: str = "default",
        data_dir: Optional[Path] = None,
        export_url: Optional[str] = None,
        auto_flush: bool = True,
        flush_interval: float = 1.0,
        strict_run_lifecycle: bool = False,
    ):
        self.service_name = service_name
        self.storage: StorageEngine = get_storage(data_dir)
        self.export_url = export_url.rstrip("/") if export_url else None
        self.flush_interval = flush_interval
        self.strict_run_lifecycle = strict_run_lifecycle

        self._queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=10_000)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._dropped_events: int = 0
        self._dropped_lock = threading.Lock()

        if auto_flush:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
            atexit.register(self._shutdown)

    @property
    def dropped_events(self) -> int:
        """Number of telemetry events silently dropped due to a full queue."""
        with self._dropped_lock:
            return self._dropped_events

    # ------------------------------------------------------------------
    # Queue/export internals
    # ------------------------------------------------------------------

    def _enqueue_event(self, envelope: Dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(envelope)
        except queue.Full:
            # Drop by design: observability must never block application code.
            # Increment the counter so callers can detect data loss.
            with self._dropped_lock:
                self._dropped_events += 1

    def _worker_loop(self) -> None:
        """Drain the queue in micro-batches to minimise SQLite transactions."""
        batch: List[Dict[str, Any]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            # Collect items until we hit the batch size or the flush window.
            try:
                item = self._queue.get(timeout=_BATCH_MAX_AGE_S)
                if item is None:
                    break
                batch.append(item)
                self._queue.task_done()
            except queue.Empty:
                pass

            age = time.monotonic() - last_flush
            if batch and (len(batch) >= _BATCH_MAX_SIZE or age >= _BATCH_MAX_AGE_S):
                self._flush_batch(batch)
                batch = []
                last_flush = time.monotonic()

        # Flush remaining items on shutdown.
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        try:
            self.storage.save_events_batch(batch)
        except Exception:
            pass
        if self.export_url:
            self._export_events(batch)

    def _export_events(self, events: List[Dict[str, Any]]) -> None:
        try:
            import httpx

            httpx.post(
                f"{self.export_url}/v1/ingest/events",
                json={"schema_version": DEFAULT_SCHEMA_VERSION, "events": events},
                timeout=2.0,
            )
        except Exception:
            # Never fail application logic for telemetry export errors.
            return

    # Keep old name as alias so existing call sites still work.
    def _export_event(self, event: Dict[str, Any]) -> None:
        self._export_events([event])

    def _shutdown(self) -> None:
        self._stop_event.set()
        # Signal the worker to exit, then wait for it to finish its current batch.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        # Drain anything still in the queue after the worker exits.
        remaining: List[Dict[str, Any]] = []
        while True:
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    remaining.append(item)
            except queue.Empty:
                break
        if remaining:
            self._flush_batch(remaining)
        self.storage.close()

    # ------------------------------------------------------------------
    # Canonical run/span/artifact lifecycle
    # ------------------------------------------------------------------

    def start_run(
        self,
        service_name: Optional[str] = None,
        session_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        input_summary: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> Run:
        run = Run(
            run_id=run_id or f"run_{uuid.uuid4().hex}",
            service_name=service_name or self.service_name,
            session_id=session_id,
            workflow_id=workflow_id,
            input_summary=input_summary,
            metadata=metadata or {},
        )
        _run_id.set(run.run_id)
        envelope = EventEnvelope(event_type="run.started", run_id=run.run_id, payload=run.to_dict(), emitted_at=run.started_at)
        self._enqueue_event(envelope.to_dict())
        return run

    def finish_run(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        output_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        active_run_id = run_id or _run_id.get()
        if not active_run_id:
            return None

        now = time.time()
        payload = {
            "run_id": active_run_id,
            "status": status.value if isinstance(status, SpanStatus) else str(status),
            "ended_at": now,
            "output_summary": output_summary,
            "error_message": error_message,
        }
        envelope = EventEnvelope(event_type="run.finished", run_id=active_run_id, payload=payload, emitted_at=now)
        self._enqueue_event(envelope.to_dict())

        if _run_id.get() == active_run_id:
            _run_id.set(None)
        return active_run_id

    def start_span(
        self,
        name: str,
        kind: Union[SpanKind, SpanType, str] = SpanKind.CUSTOM,
        parent_span_id: Optional[str] = None,
        run_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Span:
        active_run_id = run_id or _run_id.get()
        if not active_run_id:
            if self.strict_run_lifecycle:
                raise RuntimeError("No active run context. Call start_run() before start_span().")
            implicit = self.start_run(
                service_name=self.service_name,
                input_summary=f"implicit:{name}",
                metadata={"implicit_run": True},
            )
            active_run_id = implicit.run_id

        if isinstance(kind, str):
            try:
                kind = SpanKind(kind)
            except ValueError:
                kind = SpanKind.CUSTOM

        span = Span(
            span_id=f"span_{uuid.uuid4().hex}",
            run_id=active_run_id,
            parent_span_id=parent_span_id or _span_id.get(),
            kind=kind,
            name=name,
            attributes=attributes or {},
            model=kwargs.get("model"),
            provider=kwargs.get("provider"),
            operation=kwargs.get("operation"),
        )

        # Legacy compatibility: support metadata kwarg from old API.
        if kwargs.get("metadata") and isinstance(kwargs["metadata"], dict):
            span.attributes.update(kwargs["metadata"])

        _span_id.set(span.span_id)
        stack = list(_get_span_stack())
        stack.append(span.span_id)
        _span_stack.set(stack)

        envelope = EventEnvelope(
            event_type="span.started",
            run_id=span.run_id,
            span_id=span.span_id,
            payload=span.to_dict(),
            emitted_at=span.started_at,
        )
        self._enqueue_event(envelope.to_dict())
        return span

    def finish_span(
        self,
        span: Span,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[Exception] = None,
    ) -> Span:
        span.finish(status=status, error=error)
        envelope = EventEnvelope(
            event_type="span.finished",
            run_id=span.run_id,
            span_id=span.span_id,
            payload=span.to_dict(),
            emitted_at=span.ended_at or time.time(),
        )
        self._enqueue_event(envelope.to_dict())

        stack = list(_get_span_stack())
        if stack and stack[-1] == span.span_id:
            stack.pop()
        elif span.span_id in stack:
            stack.remove(span.span_id)
        _span_stack.set(stack)
        _span_id.set(stack[-1] if stack else None)
        return span

    def record_artifact(
        self,
        span_or_id: Union[Span, str],
        role: str,
        content_type: str,
        content: Any = None,
        preview: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        inline_text: Optional[str] = None,
        storage_uri: Optional[str] = None,
    ) -> Artifact:
        span_id = span_or_id.span_id if isinstance(span_or_id, Span) else span_or_id
        run_id = span_or_id.run_id if isinstance(span_or_id, Span) else (_run_id.get() or "")

        artifact = Artifact(
            artifact_id=f"art_{uuid.uuid4().hex}",
            run_id=run_id,
            span_id=span_id,
            role=role,
            content_type=content_type,
            preview=preview,
            metadata=metadata or {},
            inline_text=inline_text,
            storage_uri=storage_uri,
        )

        payload = artifact.to_dict()
        if content is not None:
            payload["content"] = content

        envelope = EventEnvelope(
            event_type="artifact.created",
            run_id=artifact.run_id,
            span_id=artifact.span_id,
            payload=payload,
            emitted_at=artifact.created_at,
        )
        self._enqueue_event(envelope.to_dict())
        return artifact

    # ------------------------------------------------------------------
    # Legacy compatibility API
    # ------------------------------------------------------------------

    def create_span(
        self,
        name: str,
        span_type: Union[SpanType, SpanKind, str] = SpanKind.CUSTOM,
        parent_span_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Span:
        return self.start_span(name=name, kind=span_type, parent_span_id=parent_span_id, **kwargs)

    def _enqueue_span(self, span: Span, inputs: Any = None, outputs: Any = None) -> None:
        # Legacy path used by old integrations; map inputs/outputs to artifacts.
        if span.ended_at is None:
            self.finish_span(span)
        if inputs is not None:
            self.record_artifact(span, role="input.message", content_type="application/json", content=inputs, preview=inputs)
        if outputs is not None:
            self.record_artifact(span, role="output.message", content_type="application/json", content=outputs, preview=outputs)

    # ------------------------------------------------------------------
    # Decorators and context manager helpers
    # ------------------------------------------------------------------

    def trace(
        self,
        name: Optional[str] = None,
        span_type: Union[SpanType, SpanKind, str] = SpanKind.CUSTOM,
        capture_input: bool = True,
        capture_output: bool = True,
        **metadata: Any,
    ):
        def decorator(func: Callable):
            span_name = name or func.__name__
            is_async = asyncio.iscoroutinefunction(func)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any):
                created_run = None
                if _run_id.get() is None:
                    created_run = self.start_run(input_summary=span_name, metadata={"implicit_run": True})
                span = self.start_span(span_name, kind=span_type, attributes=metadata)
                try:
                    if capture_input:
                        self.record_artifact(span, "input.message", "application/json", content={"args": args, "kwargs": kwargs}, preview={"args": str(args), "kwargs": str(kwargs)})
                    result = await func(*args, **kwargs)
                    if capture_output:
                        self.record_artifact(span, "output.message", "application/json", content=result, preview=result)
                    self.finish_span(span, SpanStatus.SUCCESS)
                    if created_run:
                        self.finish_run(SpanStatus.SUCCESS)
                    return result
                except asyncio.TimeoutError as exc:
                    self.finish_span(span, SpanStatus.TIMEOUT, error=exc)
                    if created_run:
                        self.finish_run(SpanStatus.TIMEOUT, error_message="Timed out")
                    raise
                except Exception as exc:
                    self.finish_span(span, SpanStatus.ERROR, error=exc)
                    if created_run:
                        self.finish_run(SpanStatus.ERROR, error_message=str(exc))
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any):
                # Capture the calling context so that if this function is
                # dispatched to a thread pool (e.g. run_in_executor) it still
                # sees the correct run_id / span_id from its parent.
                ctx = contextvars.copy_context()

                def _run_in_ctx() -> Any:
                    created_run = None
                    if _run_id.get() is None:
                        created_run = self.start_run(input_summary=span_name, metadata={"implicit_run": True})
                    span = self.start_span(span_name, kind=span_type, attributes=metadata)
                    try:
                        if capture_input:
                            self.record_artifact(span, "input.message", "application/json", content={"args": args, "kwargs": kwargs}, preview={"args": str(args), "kwargs": str(kwargs)})
                        result = func(*args, **kwargs)
                        if capture_output:
                            self.record_artifact(span, "output.message", "application/json", content=result, preview=result)
                        self.finish_span(span, SpanStatus.SUCCESS)
                        if created_run:
                            self.finish_run(SpanStatus.SUCCESS)
                        return result
                    except TimeoutError as exc:
                        self.finish_span(span, SpanStatus.TIMEOUT, error=exc)
                        if created_run:
                            self.finish_run(SpanStatus.TIMEOUT, error_message="Timed out")
                        raise
                    except Exception as exc:
                        self.finish_span(span, SpanStatus.ERROR, error=exc)
                        if created_run:
                            self.finish_run(SpanStatus.ERROR, error_message=str(exc))
                        raise

                return ctx.run(_run_in_ctx)

            return async_wrapper if is_async else sync_wrapper

        return decorator


_global_tracer: Optional[AgentTracer] = None


def get_tracer(**kwargs: Any) -> AgentTracer:
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer(**kwargs)
    elif kwargs:
        import warnings
        warnings.warn(
            "get_tracer() called with kwargs after the global tracer was already initialized. "
            f"Ignoring: {list(kwargs.keys())}. Call get_tracer() without arguments to retrieve "
            "the existing tracer, or create a new AgentTracer() instance directly.",
            stacklevel=2,
        )
    return _global_tracer


def trace(name: Optional[str] = None, span_type: Union[SpanType, SpanKind, str] = SpanKind.CUSTOM, capture_input: bool = True, capture_output: bool = True, **metadata: Any):
    return get_tracer().trace(name=name, span_type=span_type, capture_input=capture_input, capture_output=capture_output, **metadata)


class SpanContext:
    def __init__(self, name: str, span_type: Union[SpanType, SpanKind, str] = SpanKind.CUSTOM, capture_input: Any = None, **metadata: Any):
        self.name = name
        self.span_type = span_type
        self.capture_input = capture_input
        self.metadata = metadata
        self.span: Optional[Span] = None
        self._tracer = get_tracer()
        self._output = None

    def __enter__(self) -> Span:
        self.span = self._tracer.start_span(self.name, kind=self.span_type, attributes=self.metadata)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.span:
            return False
        if exc_type:
            self._tracer.finish_span(self.span, SpanStatus.ERROR, error=exc_val)
        else:
            self._tracer.finish_span(self.span, SpanStatus.SUCCESS)
        if self.capture_input is not None:
            self._tracer.record_artifact(self.span, "input.message", "application/json", content=self.capture_input, preview=self.capture_input)
        if self._output is not None:
            self._tracer.record_artifact(self.span, "output.message", "application/json", content=self._output, preview=self._output)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)

    def set_output(self, output: Any):
        self._output = output


def span(name: str, span_type: Union[SpanType, SpanKind, str] = SpanKind.CUSTOM, **metadata: Any) -> SpanContext:
    return SpanContext(name, span_type, **metadata)


def trace_llm(name: str = "llm_call", model: Optional[str] = None, capture_input: bool = True, capture_output: bool = True):
    return trace(name=name, span_type=SpanKind.LLM, capture_input=capture_input, capture_output=capture_output, model=model)


def trace_tool(name: Optional[str] = None):
    return trace(name=name, span_type=SpanKind.TOOL)


def trace_retrieval(name: str = "retrieval", collection: Optional[str] = None):
    return trace(name=name, span_type=SpanKind.RETRIEVAL, collection=collection)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed context propagation
# ─────────────────────────────────────────────────────────────────────────────

def inject_context(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject the current OAT run/span context into an outbound headers dict.

    Call this before making an HTTP request to a downstream service so that
    OAT can reconstruct the trace hierarchy on the receiving side.

    Example::

        headers = {}
        oat.inject_context(headers)
        httpx.get("http://service-b/api", headers=headers)
    """
    run_id = _run_id.get()
    span_id = _span_id.get()
    if run_id:
        headers[_HEADER_RUN_ID] = run_id
    if span_id:
        headers[_HEADER_SPAN_ID] = span_id
    return headers


def extract_context(headers: Dict[str, str]) -> bool:
    """Restore OAT context from incoming HTTP headers in a downstream service.

    Returns True if a valid OAT context was found and restored, False otherwise.

    Example (FastAPI)::

        @app.middleware("http")
        async def oat_middleware(request: Request, call_next):
            oat.extract_context(dict(request.headers))
            return await call_next(request)
    """
    run_id = (
        headers.get(_HEADER_RUN_ID)
        or headers.get("x-oat-trace-id")  # legacy alias
    )
    span_id = headers.get(_HEADER_SPAN_ID)
    if not run_id:
        return False
    _run_id.set(run_id)
    if span_id:
        # Register the upstream span as the parent so the first local span
        # created will be a child of the caller's span.
        _span_id.set(span_id)
        stack = list(_get_span_stack())
        if span_id not in stack:
            stack.append(span_id)
        _span_stack.set(stack)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess context propagation
# ─────────────────────────────────────────────────────────────────────────────

def get_subprocess_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Return an environment dict with the current OAT context baked in.

    Pass the returned dict as ``env=`` to ``subprocess.run`` / ``Popen`` so that
    spans created in the child process are linked to the parent run.

    Example::

        env = oat.get_subprocess_env()
        subprocess.run(["python", "worker.py"], env=env)
    """
    env: Dict[str, str] = dict(base_env if base_env is not None else os.environ)
    run_id = _run_id.get()
    span_id = _span_id.get()
    if run_id:
        env[_ENV_RUN_ID] = run_id
    if span_id:
        env[_ENV_SPAN_ID] = span_id
    return env


def restore_from_env() -> bool:
    """In a child/worker process, restore OAT context from environment variables.

    Call this early in the child process — after calling ``get_tracer(...)`` —
    so that spans created in the child are attached to the parent's run.

    Example (child process)::

        import oat
        tracer = oat.get_tracer(service_name="worker", export_url="http://localhost:8787")
        oat.restore_from_env()
        # All spans created from here are linked to the parent run.
    """
    run_id = os.environ.get(_ENV_RUN_ID)
    span_id = os.environ.get(_ENV_SPAN_ID)
    if not run_id:
        return False
    _run_id.set(run_id)
    if span_id:
        _span_id.set(span_id)
        _span_stack.set([span_id])
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Agent loop iteration tracking
# ─────────────────────────────────────────────────────────────────────────────

class AgentLoop:
    """Track an agent's iterative reasoning loop with per-iteration spans.

    Creates a parent AGENT span for the whole loop and a child span for each
    iteration, automatically tagging each with ``iteration_index``.

    Example::

        loop = oat.AgentLoop("plan_execute_loop", max_iterations=20)
        while not done:
            with loop.step() as step_span:
                plan = await llm_plan(state)
                result = await execute_tool(plan)
                done = result.is_terminal
        loop.finish()

    The ``step_span`` yielded by ``loop.step()`` has
    ``step_span.attributes["iteration_index"]`` set automatically.
    """

    def __init__(
        self,
        name: str,
        span_type: Union[SpanKind, str] = SpanKind.AGENT,
        max_iterations: Optional[int] = None,
        tracer: Optional["AgentTracer"] = None,
    ) -> None:
        self.name = name
        self.span_type = span_type
        self.max_iterations = max_iterations
        self._tracer = tracer or get_tracer()
        self._iteration: int = 0
        self._loop_span: Optional[Span] = self._tracer.start_span(
            name,
            kind=span_type,
            attributes={"loop": True, "max_iterations": max_iterations},
        )

    # ------------------------------------------------------------------
    # Sync iteration context manager
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def step(self, name: Optional[str] = None) -> Generator[Span, None, None]:
        """Context manager for one loop iteration.

        Yields the iteration span so callers can attach extra attributes or
        record artifacts against it.
        """
        if self.max_iterations is not None and self._iteration >= self.max_iterations:
            raise RuntimeError(
                f"AgentLoop '{self.name}' hit max_iterations={self.max_iterations}. "
                "Call loop.finish() or increase max_iterations."
            )
        step_name = name or f"{self.name}.step"
        idx = self._iteration
        self._iteration += 1
        step_span = self._tracer.start_span(
            step_name,
            kind=self.span_type,
            attributes={"iteration_index": idx, "loop_name": self.name},
        )
        try:
            yield step_span
            self._tracer.finish_span(step_span, SpanStatus.SUCCESS)
        except asyncio.TimeoutError as exc:
            self._tracer.finish_span(step_span, SpanStatus.TIMEOUT, error=exc)
            raise
        except Exception as exc:
            self._tracer.finish_span(step_span, SpanStatus.ERROR, error=exc)
            raise

    # ------------------------------------------------------------------
    # Async iteration context manager
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def astep(self, name: Optional[str] = None):
        """Async context manager for one loop iteration (use inside async code)."""
        if self.max_iterations is not None and self._iteration >= self.max_iterations:
            raise RuntimeError(
                f"AgentLoop '{self.name}' hit max_iterations={self.max_iterations}."
            )
        step_name = name or f"{self.name}.step"
        idx = self._iteration
        self._iteration += 1
        step_span = self._tracer.start_span(
            step_name,
            kind=self.span_type,
            attributes={"iteration_index": idx, "loop_name": self.name},
        )
        try:
            yield step_span
            self._tracer.finish_span(step_span, SpanStatus.SUCCESS)
        except asyncio.TimeoutError as exc:
            self._tracer.finish_span(step_span, SpanStatus.TIMEOUT, error=exc)
            raise
        except Exception as exc:
            self._tracer.finish_span(step_span, SpanStatus.ERROR, error=exc)
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finish(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: Optional[Exception] = None,
    ) -> None:
        """Close the loop's parent span. Always call this when the loop ends."""
        if self._loop_span is None:
            return
        self._loop_span.attributes["completed_iterations"] = self._iteration
        self._tracer.finish_span(self._loop_span, status, error=error)
        self._loop_span = None

    @property
    def iteration_count(self) -> int:
        """Number of iterations completed so far."""
        return self._iteration


def agent_loop(
    name: str,
    span_type: Union[SpanKind, str] = SpanKind.AGENT,
    max_iterations: Optional[int] = None,
) -> AgentLoop:
    """Convenience factory that creates an :class:`AgentLoop` using the global tracer."""
    return AgentLoop(name=name, span_type=span_type, max_iterations=max_iterations, tracer=get_tracer())
