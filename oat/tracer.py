"""
Core tracing functionality for OpenAgentTrace.
Provides decorators, context managers, and the AgentTracer class.
"""

import uuid
import time
import queue
import atexit
import asyncio
import inspect
import functools
import threading
import contextvars
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from .models import Span, SpanType, SpanStatus, LLMUsage
from .storage import StorageEngine, get_storage
from .exporters import HTTPExporter, SpanExporter
from .pricing import calculate_cost


# ============ CONTEXT VARIABLES ============
# These provide the "invisible thread" for async context propagation

_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('oat_trace_id', default=None)
_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('oat_span_id', default=None)
_span_stack: contextvars.ContextVar[List[str]] = contextvars.ContextVar('oat_span_stack', default=[])


def get_current_trace_id() -> Optional[str]:
    """Get the current trace ID from context."""
    return _trace_id.get()


def get_current_span_id() -> Optional[str]:
    """Get the current span ID from context."""
    return _span_id.get()


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Manually set the trace ID (useful for distributed tracing)."""
    return _trace_id.set(trace_id)


def _ensure_trace_id() -> str:
    """Ensure we have a trace ID, creating one if needed."""
    tid = _trace_id.get()
    if not tid:
        tid = str(uuid.uuid4())
        _trace_id.set(tid)
    return tid


# ============ AGENT TRACER CLASS ============

class AgentTracer:
    """
    The main tracer class that manages span lifecycle and export.
    
    Features:
    - Async-safe context management
    - Dual-write (Local DB + Remote Server)
    - Graceful shutdown guarantees
    """
    
    def __init__(
        self,
        service_name: str = "default",
        data_dir: Optional[Path] = None,
        export_url: Optional[str] = None,
        auto_flush: bool = True,
        flush_interval: float = 1.0,
    ):
        self.service_name = service_name
        self.storage = get_storage(data_dir)
        self.export_url = export_url
        self.auto_flush = auto_flush
        
        # Initialize remote exporter if URL provided
        self.exporter: Optional[SpanExporter] = None
        if export_url:
            self.exporter = HTTPExporter(
                endpoint=export_url,
                flush_interval=flush_interval
            )
        
        # Background queue for processing spans (prevents blocking the agent)
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        
        if auto_flush:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
            atexit.register(self._shutdown)
    
    def _worker_loop(self):
        """Background worker that saves and exports spans."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
                if item is None:
                    break
                
                span, inputs, outputs = item
                
                # 1. Save to local storage (SQLite/DuckDB) for immediate availability
                self.storage.save_span(span, inputs, outputs)
                
                # 2. Send to remote exporter (HTTP) if configured
                if self.exporter:
                    self.exporter.export(span, inputs, outputs)
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[OAT] Worker error: {e}")
    
    def _shutdown(self):
        """Graceful shutdown - flush pending spans before exit."""
        # 1. Stop accepting new local work
        self._stop_event.set()
        
        # 2. Drain the local queue
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                if item:
                    span, inputs, outputs = item
                    self.storage.save_span(span, inputs, outputs)
                    if self.exporter:
                        self.exporter.export(span, inputs, outputs)
            except queue.Empty:
                break
        
        # 3. Shutdown the remote exporter (flushes network buffer)
        if self.exporter:
            print("[OAT] Flushing traces to server...")
            self.exporter.shutdown()
            print("[OAT] Export complete.")
    
    def _enqueue_span(self, span: Span, inputs: Any = None, outputs: Any = None):
        """Add span to processing queue (non-blocking)."""
        self._queue.put((span, inputs, outputs))
    
    def create_span(
        self,
        name: str,
        span_type: Union[SpanType, str] = SpanType.FUNCTION,
        parent_span_id: Optional[str] = None,
        **kwargs
    ) -> Span:
        """Create a new span object attached to the current trace context."""
        trace_id = _ensure_trace_id()
        
        if isinstance(span_type, str):
            try:
                span_type = SpanType(span_type)
            except ValueError:
                span_type = SpanType.CUSTOM
        
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_span_id=parent_span_id or _span_id.get(),
            name=name,
            span_type=span_type,
            service_name = self.service_name,
            **kwargs
        )
        
        return span
    
    def trace(
        self,
        name: Optional[str] = None,
        span_type: Union[SpanType, str] = SpanType.FUNCTION,
        capture_input: bool = True,
        capture_output: bool = True,
        **metadata
    ):
        """
        Generic decorator for tracing any function.
        Captures inputs (args/kwargs) and outputs automatically.
        """
        def decorator(func: Callable):
            span_name = name or func.__name__
            is_async = asyncio.iscoroutinefunction(func)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # =========================================================
                # FIX: Auto-reset Trace ID for Root Agents
                # Each AGENT span creates a NEW trace, all child operations
                # (LLM, tool, retrieval) inherit that trace_id.
                # When agent completes, trace context is restored.
                # =========================================================
                trace_token = None
                if span_type == SpanType.AGENT:
                    # Create new trace ID and capture token for cleanup
                    new_trace_id = str(uuid.uuid4())
                    trace_token = _trace_id.set(new_trace_id)

                try:
                    # Proceed with execution
                    return await self._execute_traced(
                        func, args, kwargs, span_name, span_type,
                        capture_input, capture_output, metadata, is_async=True
                    )
                finally:
                    # Restore previous trace context
                    if trace_token is not None:
                        _trace_id.reset(trace_token)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Apply same trace reset logic for synchronous agents
                trace_token = None
                if span_type == SpanType.AGENT:
                    # Create new trace ID and capture token for cleanup
                    new_trace_id = str(uuid.uuid4())
                    trace_token = _trace_id.set(new_trace_id)

                try:
                    return self._execute_traced_sync(
                        func, args, kwargs, span_name, span_type,
                        capture_input, capture_output, metadata
                    )
                finally:
                    # Restore previous trace context
                    if trace_token is not None:
                        _trace_id.reset(trace_token)
            
            return async_wrapper if is_async else sync_wrapper
        
        return decorator
    
    async def _execute_traced(
        self, func, args, kwargs, name, span_type,
        capture_input, capture_output, metadata, is_async
    ):
        """Execute an async function within a span."""
        span = self.create_span(name, span_type, metadata=metadata)
        token = _span_id.set(span.span_id)
        
        inputs = {"args": args, "kwargs": kwargs} if capture_input else None
        outputs = None
        
        try:
            result = await func(*args, **kwargs)
            outputs = result if capture_output else None
            span.finish(SpanStatus.SUCCESS)
            return result
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            raise
        finally:
            self._enqueue_span(span, inputs, outputs)
            _span_id.reset(token)
    
    def _execute_traced_sync(
        self, func, args, kwargs, name, span_type,
        capture_input, capture_output, metadata
    ):
        """Execute a synchronous function within a span."""
        span = self.create_span(name, span_type, metadata=metadata)
        token = _span_id.set(span.span_id)
        
        inputs = {"args": args, "kwargs": kwargs} if capture_input else None
        outputs = None
        
        try:
            result = func(*args, **kwargs)
            outputs = result if capture_output else None
            span.finish(SpanStatus.SUCCESS)
            return result
        except Exception as e:
            span.finish(SpanStatus.ERROR, error=e)
            raise
        finally:
            self._enqueue_span(span, inputs, outputs)
            _span_id.reset(token)


# ============ GLOBAL TRACER ============

_global_tracer: Optional[AgentTracer] = None


def get_tracer(**kwargs) -> AgentTracer:
    """Singleton accessor for the global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = AgentTracer(**kwargs)
    return _global_tracer


# ============ MANUAL CONTEXT MANAGER ============

class SpanContext:
    """
    Context manager for manual tracing blocks.
    
    Usage:
        with span("complicated_logic", span_type="chain") as s:
            x = do_step_1()
            s.set_output(x)
    """
    
    def __init__(
        self,
        name: str,
        span_type: Union[SpanType, str] = SpanType.FUNCTION,
        capture_input: Any = None,
        **metadata
    ):
        self.name = name
        self.span_type = span_type
        self.capture_input = capture_input
        self.metadata = metadata
        self.span: Optional[Span] = None
        self._token: Optional[contextvars.Token] = None
        self._trace_token: Optional[contextvars.Token] = None
        self._tracer = get_tracer()
        self._output = None
    
    def __enter__(self) -> Span:
        # Save trace token if creating agent span
        if self.span_type == SpanType.AGENT or (isinstance(self.span_type, str) and self.span_type == "agent"):
            new_trace_id = str(uuid.uuid4())
            self._trace_token = _trace_id.set(new_trace_id)

        self.span = self._tracer.create_span(
            self.name,
            self.span_type,
            metadata=self.metadata
        )
        self._token = _span_id.set(self.span.span_id)
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            if exc_type:
                self.span.finish(SpanStatus.ERROR, error=exc_val)
            else:
                self.span.finish(SpanStatus.SUCCESS)

            self._tracer._enqueue_span(self.span, self.capture_input, self._output)

        # Reset span context
        if self._token:
            _span_id.reset(self._token)

        # Reset trace context if we created one for AGENT span
        if self._trace_token:
            _trace_id.reset(self._trace_token)

        return False  # Propagate exceptions
    
    async def __aenter__(self) -> Span:
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
    
    def set_output(self, output: Any):
        """Manually capture output data."""
        self._output = output
        if self.span:
            self.span.set_output(output)


def span(
    name: str,
    span_type: Union[SpanType, str] = SpanType.FUNCTION,
    **metadata
) -> SpanContext:
    """Helper to create a SpanContext."""
    return SpanContext(name, span_type, **metadata)


def trace(
    name: Optional[str] = None,
    span_type: Union[SpanType, str] = SpanType.FUNCTION,
    capture_input: bool = True,
    capture_output: bool = True,
    **metadata
):
    """Global convenience decorator."""
    tracer = get_tracer()
    return tracer.trace(name, span_type, capture_input, capture_output, **metadata)


# ============ SPECIALIZED DECORATORS ============

def trace_llm(
    name: str = "llm_call",
    model: Optional[str] = None,
    service_provider: Optional[str] = None,
    capture_input: bool = True,
    capture_output: bool = True,
):
    """
    Traces LLM calls and extracts token usage/cost.
    Compatible with OpenAI-style 'usage' objects.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.create_span(name, SpanType.LLM, model=model, service_provider=service_provider)
            token = _span_id.set(span.span_id)
            inputs = {"args": args, "kwargs": kwargs} if capture_input else None
            try:
                result = await func(*args, **kwargs)
                if hasattr(result, 'usage') and result.usage:
                    prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                    total_tokens = getattr(result.usage, 'total_tokens', 0)
                    
                    # Calculate cost from pricing config
                    prompt_cost, completion_cost, total_cost = calculate_cost(
                        model or "", prompt_tokens, completion_tokens, total_tokens
                    )
                    
                    span.usage = LLMUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        prompt_cost=prompt_cost,
                        completion_cost=completion_cost,
                        total_cost=total_cost,
                    )
                span.finish(SpanStatus.SUCCESS)
                tracer._enqueue_span(span, inputs, result if capture_output else None)
                return result
            except Exception as e:
                span.finish(SpanStatus.ERROR, error=e)
                tracer._enqueue_span(span, inputs, None)
                raise
            finally:
                _span_id.reset(token)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.create_span(name, SpanType.LLM, model=model, service_provider=service_provider)
            token = _span_id.set(span.span_id)
            inputs = {"args": args, "kwargs": kwargs} if capture_input else None
            try:
                result = func(*args, **kwargs)
                if hasattr(result, 'usage') and result.usage:
                    prompt_tokens = getattr(result.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(result.usage, 'completion_tokens', 0)
                    total_tokens = getattr(result.usage, 'total_tokens', 0)
                    prompt_cost, completion_cost, total_cost = calculate_cost(
                        model or "", prompt_tokens, completion_tokens, total_tokens
                    )
                    span.usage = LLMUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        prompt_cost=prompt_cost,
                        completion_cost=completion_cost,
                        total_cost=total_cost,
                    )
                span.finish(SpanStatus.SUCCESS)
                tracer._enqueue_span(span, inputs, result if capture_output else None)
                return result
            except Exception as e:
                span.finish(SpanStatus.ERROR, error=e)
                tracer._enqueue_span(span, inputs, None)
                raise
            finally:
                _span_id.reset(token)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def trace_guardrail(name: str = "guardrail", action: str = "check"):
    """
    Traces safety checks.
    Automatically sets 'guardrail_triggered' to True if the function returns False or raises an exception.
    """
    def decorator(func: Callable):
        is_async = asyncio.iscoroutinefunction(func)
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.create_span(name, SpanType.GUARDRAIL, guardrail_action=action)
            token = _span_id.set(span.span_id)
            inputs = {"args": args, "kwargs": kwargs}
            try:
                result = await func(*args, **kwargs)
                # If guardrail returns boolean, False means "failed check" -> Triggered
                span.guardrail_triggered = not bool(result) if isinstance(result, bool) else False
                span.finish(SpanStatus.SUCCESS)
                tracer._enqueue_span(span, inputs, result)
                return result
            except Exception as e:
                span.guardrail_triggered = True
                span.finish(SpanStatus.ERROR, error=e)
                tracer._enqueue_span(span, inputs, None)
                raise
            finally:
                _span_id.reset(token)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            span = tracer.create_span(name, SpanType.GUARDRAIL, guardrail_action=action)
            token = _span_id.set(span.span_id)
            inputs = {"args": args, "kwargs": kwargs}
            try:
                result = func(*args, **kwargs)
                span.guardrail_triggered = not bool(result) if isinstance(result, bool) else False
                span.finish(SpanStatus.SUCCESS)
                tracer._enqueue_span(span, inputs, result)
                return result
            except Exception as e:
                span.guardrail_triggered = True
                span.finish(SpanStatus.ERROR, error=e)
                tracer._enqueue_span(span, inputs, None)
                raise
            finally:
                _span_id.reset(token)
        
        return async_wrapper if is_async else sync_wrapper
    return decorator


def trace_retrieval(name: str = "retrieval", collection: Optional[str] = None):
    """Traces RAG retrieval steps, capturing the collection name."""
    return trace(name=name, span_type=SpanType.RETRIEVAL, collection=collection)


def trace_tool(name: Optional[str] = None):
    """Traces generic tool execution."""
    return trace(name=name, span_type=SpanType.TOOL)


def trace_agent(name: str = "agent", capture_input: bool = True, capture_output: bool = True):
    """Traces the root agent execution loop."""
    return trace(name=name, span_type=SpanType.AGENT, capture_input=capture_input, capture_output=capture_output)


def trace_memory(name: str = "memory", operation: str = "read"):
    """Traces memory operations (read/write)."""
    return trace(name=name, span_type=SpanType.MEMORY, operation=operation)


def trace_http(name: str = "http_request", method: Optional[str] = None):
    """Traces outbound HTTP requests."""
    return trace(name=name, span_type=SpanType.HTTP, method=method)


def trace_chain(name: str = "chain"):
    """Traces a logical chain of steps."""
    return trace(name=name, span_type=SpanType.CHAIN)


def trace_file_io(name: str = "file_operation", operation: str = "read"):
    """Traces file system operations."""
    return trace(name=name, span_type=SpanType.FILE_IO, operation=operation)


def trace_database(name: str = "database_query", operation: str = "query"):
    """Traces database interactions."""
    return trace(name=name, span_type=SpanType.DATABASE, operation=operation)


def trace_cache(name: str = "cache", operation: str = "get"):
    """Traces cache hits/misses."""
    return trace(name=name, span_type=SpanType.CACHE, operation=operation)