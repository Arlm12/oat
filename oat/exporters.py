"""
Exporters for OpenAgentTrace.
Send spans to various backends: HTTP server, console, file, etc.
"""

import json
import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime

from .models import Span


class SpanExporter(ABC):
    """Base class for span exporters."""
    
    @abstractmethod
    def export(self, span: Span, inputs: Any = None, outputs: Any = None):
        """Export a single span."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Graceful shutdown."""
        pass


class ConsoleExporter(SpanExporter):
    """
    Export spans to console for debugging.
    
    Usage:
        exporter = ConsoleExporter(verbose=True)
    """
    
    def __init__(self, verbose: bool = False, colored: bool = True):
        self.verbose = verbose
        self.colored = colored
    
    def export(self, span: Span, inputs: Any = None, outputs: Any = None):
        # Status colors
        colors = {
            "success": "\033[92m",  # Green
            "error": "\033[91m",    # Red
            "running": "\033[93m",  # Yellow
        }
        reset = "\033[0m"
        
        status = span.status.value if hasattr(span.status, 'value') else str(span.status)
        color = colors.get(status, "") if self.colored else ""
        
        # Format duration
        duration_str = f"{span.duration_ms:.2f}ms" if span.duration_ms else "..."
        
        # Basic output
        print(f"{color}[OAT] {span.name} ({span.span_type.value}) - {status} - {duration_str}{reset}")
        
        if self.verbose:
            print(f"      trace_id: {span.trace_id[:8]}... span_id: {span.span_id[:8]}...")
            if span.parent_span_id:
                print(f"      parent: {span.parent_span_id[:8]}...")
            if span.model:
                print(f"      model: {span.model}")
            if span.usage:
                print(f"      tokens: {span.usage.total_tokens}, cost: ${span.usage.total_cost:.4f}")
            if span.error_message:
                print(f"      error: {span.error_message}")
    
    def shutdown(self):
        pass


class HTTPExporter(SpanExporter):
    """
    Export spans to an HTTP endpoint (the OAT server).
    
    Features:
    - Non-blocking async export
    - Automatic retry with backoff
    - Batch export support
    
    Usage:
        exporter = HTTPExporter("http://localhost:8000")
    """
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000",
        batch_size: int = 10,
        flush_interval: float = 1.0,
        max_retries: int = 3,
        timeout: float = 5.0,
    ):
        self.endpoint = endpoint.rstrip('/')
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
    
    def export(self, span: Span, inputs: Any = None, outputs: Any = None):
        """Queue span for async export."""
        payload = span.to_dict()
        payload['inputs'] = inputs
        payload['outputs'] = outputs
        self._queue.put(payload)
    
    def _worker_loop(self):
        """Background worker that batches and sends spans."""
        import requests
        
        batch: List[Dict] = []
        last_flush = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Get items with timeout
                try:
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)
                    self._queue.task_done()
                except queue.Empty:
                    pass
                
                # Flush conditions
                should_flush = (
                    len(batch) >= self.batch_size or
                    (batch and time.time() - last_flush >= self.flush_interval)
                )
                
                if should_flush and batch:
                    self._send_batch(batch)
                    batch = []
                    last_flush = time.time()
                    
            except Exception as e:
                print(f"[OAT] Export worker error: {e}")
        
        # Drain remaining items in queue on shutdown
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                batch.append(item)
                if len(batch) >= self.batch_size:
                    self._send_batch(batch)
                    batch = []
            except queue.Empty:
                break
        
        # Final flush of any remaining partial batch
        if batch:
            try:
                self._send_batch(batch)
            except Exception as e:
                with open("oat_debug.log", "a") as f:
                    f.write(f"Final flush error: {e}\n")
    
    def _send_batch(self, batch: List[Dict]):
        """Send a batch of spans to the server."""
        import requests
        
        for span_data in batch:
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(
                        f"{self.endpoint}/ingest",
                        json=span_data,
                        timeout=self.timeout,
                        proxies={"http": None, "https": None}
                    )
                    if response.status_code == 200:
                        break
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        print(f"[OAT] Failed to export span after {self.max_retries} attempts: {e}")
                    else:
                        time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    def shutdown(self):
        """Graceful shutdown."""
        self._stop_event.set()
        self._worker.join(timeout=5.0)


class FileExporter(SpanExporter):
    """
    Export spans to a JSON Lines file.
    
    Usage:
        exporter = FileExporter("traces.jsonl")
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self._lock = threading.Lock()
    
    def export(self, span: Span, inputs: Any = None, outputs: Any = None):
        payload = span.to_dict()
        payload['inputs'] = inputs
        payload['outputs'] = outputs
        payload['exported_at'] = datetime.now().isoformat()
        
        with self._lock:
            with open(self.filepath, 'a') as f:
                f.write(json.dumps(payload, default=str) + '\n')
    
    def shutdown(self):
        pass


class CompositeExporter(SpanExporter):
    """
    Combine multiple exporters.
    
    Usage:
        exporter = CompositeExporter([
            ConsoleExporter(),
            HTTPExporter("http://localhost:8000"),
        ])
    """
    
    def __init__(self, exporters: List[SpanExporter]):
        self.exporters = exporters
    
    def export(self, span: Span, inputs: Any = None, outputs: Any = None):
        for exporter in self.exporters:
            try:
                exporter.export(span, inputs, outputs)
            except Exception as e:
                print(f"[OAT] Exporter {type(exporter).__name__} failed: {e}")
    
    def shutdown(self):
        for exporter in self.exporters:
            exporter.shutdown()
