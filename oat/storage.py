"""Canonical local storage for OpenAgentTrace."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import sqlite3
import threading
import time
import uuid

from .models import Artifact, DEFAULT_SCHEMA_VERSION, EventEnvelope, Run, Span, SpanKind, SpanStatus


class StorageEngine:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(".oat")
        self.db_path = self.data_dir / "telemetry.db"
        self.blob_dir = self.data_dir / "blobs"
        self._local = threading.local()
        self._init_storage()

    def _init_storage(self) -> None:
        self.data_dir.mkdir(exist_ok=True)
        self.blob_dir.mkdir(exist_ok=True)
        with self._get_connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    service_version TEXT,
                    environment TEXT,
                    session_id TEXT,
                    workflow_id TEXT,
                    root_span_id TEXT,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_ms REAL,
                    input_summary TEXT,
                    output_summary TEXT,
                    error_message TEXT,
                    metadata_json TEXT,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    llm_calls INTEGER DEFAULT 0,
                    tool_calls INTEGER DEFAULT 0,
                    span_count INTEGER DEFAULT 0,
                    updated_at REAL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    trace_path TEXT,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    duration_ms REAL,
                    model TEXT,
                    provider TEXT,
                    operation TEXT,
                    input_summary TEXT,
                    output_summary TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    attributes_json TEXT,
                    usage_json TEXT,
                    cost_json TEXT,
                    score INTEGER,
                    feedback TEXT,
                    updated_at REAL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    storage_uri TEXT,
                    inline_text TEXT,
                    preview_json TEXT,
                    metadata_json TEXT,
                    size_bytes INTEGER,
                    sha256 TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    span_id TEXT,
                    score INTEGER NOT NULL,
                    comment TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS event_log (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schema_version TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    emitted_at REAL NOT NULL,
                    run_id TEXT NOT NULL,
                    span_id TEXT,
                    payload_json TEXT NOT NULL,
                    created_at REAL DEFAULT (unixepoch('now'))
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_service ON runs(service_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_artifacts_span ON artifacts(span_id)")
            conn.commit()

    @contextmanager
    def _get_connection(self) -> Iterable[sqlite3.Connection]:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
            self._local.conn.row_factory = sqlite3.Row
        conn = self._local.conn
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    # blob helpers
    def save_blob(self, data: Any) -> Optional[Tuple[str, str, int]]:
        if data is None:
            return None
        if isinstance(data, bytes):
            raw = data
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            try:
                raw = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
            except (TypeError, ValueError):
                raw = str(data).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        path = self.blob_dir / digest
        if not path.exists():
            path.write_bytes(raw)
        return (f"blob://sha256/{digest}", digest, len(raw))

    def get_blob(self, storage_uri_or_hash: Optional[str]) -> Optional[Any]:
        if not storage_uri_or_hash:
            return None
        digest = storage_uri_or_hash.split("/")[-1]
        path = self.blob_dir / digest
        if not path.exists():
            return None
        raw = path.read_bytes()
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw

    # canonical writes
    def save_event(self, event: EventEnvelope | Dict[str, Any]) -> None:
        env = event.to_dict() if isinstance(event, EventEnvelope) else dict(event)
        event_type = env.get("event_type")
        run_id = env.get("run_id")
        span_id = env.get("span_id")
        emitted_at = float(env.get("emitted_at") or time.time())
        payload = env.get("payload") or {}
        schema_version = env.get("schema_version") or DEFAULT_SCHEMA_VERSION

        if not event_type or not run_id:
            raise ValueError("event requires event_type and run_id")

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO event_log(schema_version,event_type,emitted_at,run_id,span_id,payload_json) VALUES(?,?,?,?,?,?)",
                (schema_version, event_type, emitted_at, run_id, span_id, json.dumps(payload, ensure_ascii=False, default=str)),
            )

            if event_type == "run.started":
                self._run_started(conn, run_id, emitted_at, payload)
            elif event_type == "run.finished":
                self._run_finished(conn, run_id, emitted_at, payload)
                # Full rollup only at run completion — keeps span events cheap.
                self._refresh_rollup(conn, run_id)
            elif event_type == "span.started":
                self._span_started(conn, run_id, span_id, emitted_at, payload)
                self._inc_run_span_count(conn, run_id, payload.get("kind") or payload.get("span_type") or "custom")
            elif event_type == "span.finished":
                self._span_finished(conn, run_id, span_id, emitted_at, payload)
                self._update_run_from_finished_span(conn, run_id, payload)
            elif event_type == "artifact.created":
                self._artifact_created(conn, run_id, span_id, emitted_at, payload)

            conn.commit()

    def upsert_run(self, run: Run) -> None:
        payload = run.to_dict()
        self.save_event(EventEnvelope(event_type="run.started", run_id=run.run_id, payload=payload, emitted_at=run.started_at))
        if run.ended_at is not None:
            self.save_event(EventEnvelope(event_type="run.finished", run_id=run.run_id, payload=payload, emitted_at=run.ended_at))

    def upsert_span(self, span: Span) -> None:
        payload = span.to_dict()
        self.save_event(EventEnvelope(event_type="span.started", run_id=span.run_id, span_id=span.span_id, payload=payload, emitted_at=span.started_at))
        if span.ended_at is not None:
            self.save_event(EventEnvelope(event_type="span.finished", run_id=span.run_id, span_id=span.span_id, payload=payload, emitted_at=span.ended_at))

    def save_artifact(self, artifact: Artifact, content: Any = None) -> Artifact:
        payload = artifact.to_dict()
        if content is not None:
            payload["content"] = content
        self.save_event(EventEnvelope(event_type="artifact.created", run_id=artifact.run_id, span_id=artifact.span_id, payload=payload, emitted_at=artifact.created_at))
        return artifact

    # event handlers
    def _run_started(self, conn: sqlite3.Connection, run_id: str, emitted_at: float, payload: Dict[str, Any]) -> None:
        existing = self._row(conn, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        started = float(payload.get("started_at") or emitted_at)
        meta = payload.get("metadata")
        if meta is None:
            meta = _json_load(existing.get("metadata_json") if existing else None, {})

        row = {
            "run_id": run_id,
            "service_name": payload.get("service_name") or (existing.get("service_name") if existing else "default"),
            "service_version": payload.get("service_version") if payload.get("service_version") is not None else (existing.get("service_version") if existing else None),
            "environment": payload.get("environment") if payload.get("environment") is not None else (existing.get("environment") if existing else None),
            "session_id": payload.get("session_id") if payload.get("session_id") is not None else (existing.get("session_id") if existing else None),
            "workflow_id": payload.get("workflow_id") if payload.get("workflow_id") is not None else (existing.get("workflow_id") if existing else None),
            "root_span_id": payload.get("root_span_id") if payload.get("root_span_id") is not None else (existing.get("root_span_id") if existing else None),
            "status": payload.get("status") or (existing.get("status") if existing else SpanStatus.RUNNING.value),
            "started_at": started,
            "ended_at": payload.get("ended_at") if payload.get("ended_at") is not None else (existing.get("ended_at") if existing else None),
            "duration_ms": payload.get("duration_ms") if payload.get("duration_ms") is not None else (existing.get("duration_ms") if existing else None),
            "input_summary": payload.get("input_summary") if payload.get("input_summary") is not None else (existing.get("input_summary") if existing else None),
            "output_summary": payload.get("output_summary") if payload.get("output_summary") is not None else (existing.get("output_summary") if existing else None),
            "error_message": payload.get("error_message") if payload.get("error_message") is not None else (existing.get("error_message") if existing else None),
            "metadata_json": json.dumps(meta, ensure_ascii=False, default=str),
            "total_tokens": int(payload.get("total_tokens") or (existing.get("total_tokens") if existing else 0) or 0),
            "total_cost": float(payload.get("total_cost") or (existing.get("total_cost") if existing else 0.0) or 0.0),
            "llm_calls": int(payload.get("llm_calls") or (existing.get("llm_calls") if existing else 0) or 0),
            "tool_calls": int(payload.get("tool_calls") or (existing.get("tool_calls") if existing else 0) or 0),
            "span_count": int(payload.get("span_count") or (existing.get("span_count") if existing else 0) or 0),
            "updated_at": time.time(),
        }
        self._upsert_run(conn, row)

    def _run_finished(self, conn: sqlite3.Connection, run_id: str, emitted_at: float, payload: Dict[str, Any]) -> None:
        existing = self._row(conn, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not existing:
            self._run_started(conn, run_id, emitted_at, payload)
            existing = self._row(conn, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        started = float(payload.get("started_at") or existing.get("started_at") or emitted_at)
        ended = float(payload.get("ended_at") or emitted_at)
        duration = payload.get("duration_ms")
        if duration is None:
            duration = max(0.0, (ended - started) * 1000.0)

        row = dict(existing)
        row.update(
            {
                "status": payload.get("status") or SpanStatus.SUCCESS.value,
                "ended_at": ended,
                "duration_ms": duration,
                "output_summary": payload.get("output_summary") if payload.get("output_summary") is not None else existing.get("output_summary"),
                "error_message": payload.get("error_message") if payload.get("error_message") is not None else existing.get("error_message"),
                "updated_at": time.time(),
            }
        )
        self._upsert_run(conn, row)

    def _span_started(self, conn: sqlite3.Connection, run_id: str, span_id: Optional[str], emitted_at: float, payload: Dict[str, Any]) -> None:
        sid = span_id or payload.get("span_id")
        if not sid:
            raise ValueError("span.started requires span_id")
        if not self._row(conn, "SELECT run_id FROM runs WHERE run_id=?", (run_id,)):
            self._run_started(conn, run_id, emitted_at, payload)

        existing = self._row(conn, "SELECT * FROM spans WHERE span_id=?", (sid,))
        attrs = payload.get("attributes")
        if attrs is None:
            attrs = payload.get("metadata")
        if attrs is None:
            attrs = _json_load(existing.get("attributes_json") if existing else None, {})
        usage = payload.get("usage")
        if usage is None:
            usage = _json_load(existing.get("usage_json") if existing else None, {})
        cost = payload.get("cost")
        if cost is None:
            cost = _json_load(existing.get("cost_json") if existing else None, {})

        row = {
            "span_id": sid,
            "run_id": run_id,
            "parent_span_id": payload.get("parent_span_id") if payload.get("parent_span_id") is not None else (existing.get("parent_span_id") if existing else None),
            "trace_path": payload.get("trace_path") if payload.get("trace_path") is not None else (existing.get("trace_path") if existing else None),
            "kind": payload.get("kind") or payload.get("span_type") or (existing.get("kind") if existing else SpanKind.CUSTOM.value),
            "name": payload.get("name") or (existing.get("name") if existing else "unnamed"),
            "status": payload.get("status") or (existing.get("status") if existing else SpanStatus.RUNNING.value),
            "started_at": float(payload.get("started_at") or payload.get("start_time") or (existing.get("started_at") if existing else emitted_at)),
            "ended_at": payload.get("ended_at") if payload.get("ended_at") is not None else (payload.get("end_time") if payload.get("end_time") is not None else (existing.get("ended_at") if existing else None)),
            "duration_ms": payload.get("duration_ms") if payload.get("duration_ms") is not None else (existing.get("duration_ms") if existing else None),
            "model": payload.get("model") if payload.get("model") is not None else (existing.get("model") if existing else None),
            "provider": payload.get("provider") if payload.get("provider") is not None else (existing.get("provider") if existing else None),
            "operation": payload.get("operation") if payload.get("operation") is not None else (existing.get("operation") if existing else None),
            "input_summary": payload.get("input_summary") if payload.get("input_summary") is not None else (payload.get("input_preview") if payload.get("input_preview") is not None else (existing.get("input_summary") if existing else None)),
            "output_summary": payload.get("output_summary") if payload.get("output_summary") is not None else (payload.get("output_preview") if payload.get("output_preview") is not None else (existing.get("output_summary") if existing else None)),
            "error_type": payload.get("error_type") if payload.get("error_type") is not None else (existing.get("error_type") if existing else None),
            "error_message": payload.get("error_message") if payload.get("error_message") is not None else (existing.get("error_message") if existing else None),
            "attributes_json": json.dumps(attrs, ensure_ascii=False, default=str),
            "usage_json": json.dumps(usage, ensure_ascii=False, default=str),
            "cost_json": json.dumps(cost, ensure_ascii=False, default=str),
            "score": payload.get("score") if payload.get("score") is not None else (existing.get("score") if existing else None),
            "feedback": payload.get("feedback") if payload.get("feedback") is not None else (existing.get("feedback") if existing else None),
            "updated_at": time.time(),
        }
        self._upsert_span(conn, row)

    def _span_finished(self, conn: sqlite3.Connection, run_id: str, span_id: Optional[str], emitted_at: float, payload: Dict[str, Any]) -> None:
        self._span_started(conn, run_id, span_id, emitted_at, payload)
        sid = span_id or payload.get("span_id")
        existing = self._row(conn, "SELECT * FROM spans WHERE span_id=?", (sid,))
        if not existing:
            return
        started = float(existing.get("started_at") or emitted_at)
        ended = float(payload.get("ended_at") or payload.get("end_time") or emitted_at)
        duration = payload.get("duration_ms")
        if duration is None:
            duration = max(0.0, (ended - started) * 1000.0)

        row = dict(existing)
        row.update(
            {
                "status": payload.get("status") or SpanStatus.SUCCESS.value,
                "ended_at": ended,
                "duration_ms": duration,
                "error_type": payload.get("error_type") if payload.get("error_type") is not None else existing.get("error_type"),
                "error_message": payload.get("error_message") if payload.get("error_message") is not None else existing.get("error_message"),
                "updated_at": time.time(),
            }
        )
        if payload.get("usage") is not None:
            row["usage_json"] = json.dumps(payload.get("usage") or {}, ensure_ascii=False, default=str)
        if payload.get("cost") is not None:
            row["cost_json"] = json.dumps(payload.get("cost") or {}, ensure_ascii=False, default=str)
        self._upsert_span(conn, row)

    def _artifact_created(self, conn: sqlite3.Connection, run_id: str, span_id: Optional[str], emitted_at: float, payload: Dict[str, Any]) -> None:
        sid = span_id or payload.get("span_id")
        if not sid:
            raise ValueError("artifact.created requires span_id")
        aid = payload.get("artifact_id") or f"art_{uuid.uuid4().hex}"
        preview = payload.get("preview")
        inline = payload.get("inline_text")
        storage_uri = payload.get("storage_uri")
        sha256 = payload.get("sha256")
        size_bytes = payload.get("size_bytes")
        content = payload.get("content")

        if content is not None and not storage_uri and inline is None:
            saved = self.save_blob(content)
            if saved:
                storage_uri, sha256, size_bytes = saved
            if isinstance(content, str) and len(content) <= 4096:
                inline = content
            elif preview is None:
                preview = _preview(content)

        conn.execute(
            """
            INSERT OR REPLACE INTO artifacts(
                artifact_id,run_id,span_id,role,content_type,storage_uri,inline_text,
                preview_json,metadata_json,size_bytes,sha256,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                aid,
                run_id,
                sid,
                payload.get("role") or "output.message",
                payload.get("content_type") or "application/json",
                storage_uri,
                inline,
                json.dumps(preview, ensure_ascii=False, default=str) if preview is not None else None,
                json.dumps(payload.get("metadata") or {}, ensure_ascii=False, default=str),
                size_bytes,
                sha256,
                float(payload.get("created_at") or emitted_at),
            ),
        )

    def _refresh_rollup(self, conn: sqlite3.Connection, run_id: str) -> None:
        run = self._row(conn, "SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not run:
            return
        rows = conn.execute("SELECT kind,status,started_at,ended_at,usage_json,cost_json FROM spans WHERE run_id=?", (run_id,)).fetchall()
        span_count = len(rows)
        llm_calls = 0
        tool_calls = 0
        total_tokens = 0
        total_cost = 0.0
        errors = 0
        has_running = False
        min_started = None
        max_ended = None

        for r in rows:
            if r["kind"] == SpanKind.LLM.value:
                llm_calls += 1
            if r["kind"] == SpanKind.TOOL.value:
                tool_calls += 1
            if r["status"] == SpanStatus.ERROR.value:
                errors += 1
            if r["status"] == SpanStatus.RUNNING.value:
                has_running = True
            s = r["started_at"]
            e = r["ended_at"]
            if s is not None:
                min_started = s if min_started is None else min(min_started, s)
            if e is not None:
                max_ended = e if max_ended is None else max(max_ended, e)
            usage = _json_load(r["usage_json"], {})
            cost = _json_load(r["cost_json"], {})
            total_tokens += int(usage.get("total_tokens") or 0)
            total_cost += float(cost.get("total_cost") or usage.get("total_cost") or 0.0)

        status = run["status"]
        if status == SpanStatus.RUNNING.value:
            if errors > 0:
                status = SpanStatus.ERROR.value
            elif has_running:
                status = SpanStatus.RUNNING.value
            elif span_count > 0:
                status = SpanStatus.SUCCESS.value

        started = run["started_at"] if run["started_at"] is not None else min_started
        ended = run["ended_at"] if run["ended_at"] is not None else max_ended
        duration = run["duration_ms"]
        if duration is None and started is not None and ended is not None:
            duration = max(0.0, (ended - started) * 1000.0)

        conn.execute(
            "UPDATE runs SET status=?,started_at=?,ended_at=?,duration_ms=?,total_tokens=?,total_cost=?,llm_calls=?,tool_calls=?,span_count=?,updated_at=? WHERE run_id=?",
            (status, started or time.time(), ended, duration, total_tokens, total_cost, llm_calls, tool_calls, span_count, time.time(), run_id),
        )

    def _inc_run_span_count(self, conn: sqlite3.Connection, run_id: str, kind: str) -> None:
        """Lightweight incremental update — no full-table scan required."""
        is_llm = 1 if kind == SpanKind.LLM.value else 0
        is_tool = 1 if kind == SpanKind.TOOL.value else 0
        conn.execute(
            "UPDATE runs SET span_count=span_count+1, llm_calls=llm_calls+?, tool_calls=tool_calls+?, updated_at=? WHERE run_id=?",
            (is_llm, is_tool, time.time(), run_id),
        )

    def _update_run_from_finished_span(self, conn: sqlite3.Connection, run_id: str, payload: Dict[str, Any]) -> None:
        """Add this span's tokens/cost to the run totals; mark run ERROR if needed."""
        usage = payload.get("usage") or {}
        cost = payload.get("cost") or {}
        tokens = int(usage.get("total_tokens") or 0)
        total_cost = float(cost.get("total_cost") or usage.get("total_cost") or 0.0)
        now = time.time()
        conn.execute(
            "UPDATE runs SET total_tokens=total_tokens+?, total_cost=total_cost+?, updated_at=? WHERE run_id=?",
            (tokens, total_cost, now, run_id),
        )
        # Escalate run status to error if this span errored, but never downgrade
        # an already-errored run.
        if (payload.get("status") or "") == SpanStatus.ERROR.value:
            conn.execute(
                "UPDATE runs SET status=? WHERE run_id=? AND status NOT IN (?,?)",
                (SpanStatus.ERROR.value, run_id, SpanStatus.ERROR.value, SpanStatus.CANCELLED.value),
            )

    # ------------------------------------------------------------------
    # Batch save — commits all events in one SQLite transaction
    # ------------------------------------------------------------------

    def save_events_batch(self, events: List[Dict[str, Any]]) -> None:
        """Process a list of events inside a single transaction.

        Called by the background worker instead of save_event() one-by-one,
        which reduces SQLite write amplification significantly.
        """
        with self._get_connection() as conn:
            for env in events:
                env = dict(env)
                event_type = env.get("event_type")
                run_id = env.get("run_id")
                span_id = env.get("span_id")
                if not event_type or not run_id:
                    continue
                emitted_at = float(env.get("emitted_at") or time.time())
                payload = env.get("payload") or {}
                schema_version = env.get("schema_version") or DEFAULT_SCHEMA_VERSION

                conn.execute(
                    "INSERT INTO event_log(schema_version,event_type,emitted_at,run_id,span_id,payload_json) VALUES(?,?,?,?,?,?)",
                    (schema_version, event_type, emitted_at, run_id, span_id,
                     json.dumps(payload, ensure_ascii=False, default=str)),
                )

                try:
                    if event_type == "run.started":
                        self._run_started(conn, run_id, emitted_at, payload)
                    elif event_type == "run.finished":
                        self._run_finished(conn, run_id, emitted_at, payload)
                        self._refresh_rollup(conn, run_id)
                    elif event_type == "span.started":
                        self._span_started(conn, run_id, span_id, emitted_at, payload)
                        self._inc_run_span_count(conn, run_id, payload.get("kind") or payload.get("span_type") or "custom")
                    elif event_type == "span.finished":
                        self._span_finished(conn, run_id, span_id, emitted_at, payload)
                        self._update_run_from_finished_span(conn, run_id, payload)
                    elif event_type == "artifact.created":
                        self._artifact_created(conn, run_id, span_id, emitted_at, payload)
                except Exception:
                    # Don't let one bad event abort the entire batch.
                    pass

            conn.commit()

    # ------------------------------------------------------------------
    # Orphaned-span cleanup
    # ------------------------------------------------------------------

    def mark_stale_spans_cancelled(self, cutoff_ts: float) -> int:
        """Mark spans and runs that are still RUNNING after *cutoff_ts* as CANCELLED.

        Returns the number of spans updated.  Called by the server's background
        housekeeping task so dashboards don't show forever-running executions
        after a process crash.
        """
        now = time.time()
        with self._get_connection() as conn:
            stale = conn.execute(
                "SELECT span_id, run_id, started_at FROM spans WHERE status=? AND started_at<?",
                (SpanStatus.RUNNING.value, cutoff_ts),
            ).fetchall()
            if not stale:
                return 0

            for row in stale:
                duration_ms = max(0.0, (now - float(row["started_at"])) * 1000.0)
                conn.execute(
                    "UPDATE spans SET status=?, ended_at=?, duration_ms=?, updated_at=? WHERE span_id=?",
                    (SpanStatus.CANCELLED.value, now, duration_ms, now, row["span_id"]),
                )

            # Also close any run that is still RUNNING and old enough.
            conn.execute(
                "UPDATE runs SET status=?, ended_at=?, updated_at=? WHERE status=? AND started_at<?",
                (SpanStatus.CANCELLED.value, now, now, SpanStatus.RUNNING.value, cutoff_ts),
            )
            conn.commit()

        return len(stale)

    def _upsert_run(self, conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs(
                run_id,service_name,service_version,environment,session_id,workflow_id,root_span_id,
                status,started_at,ended_at,duration_ms,input_summary,output_summary,error_message,
                metadata_json,total_tokens,total_cost,llm_calls,tool_calls,span_count,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["run_id"],
                row["service_name"],
                row.get("service_version"),
                row.get("environment"),
                row.get("session_id"),
                row.get("workflow_id"),
                row.get("root_span_id"),
                row.get("status") or SpanStatus.RUNNING.value,
                row.get("started_at") or time.time(),
                row.get("ended_at"),
                row.get("duration_ms"),
                row.get("input_summary"),
                row.get("output_summary"),
                row.get("error_message"),
                row.get("metadata_json"),
                int(row.get("total_tokens") or 0),
                float(row.get("total_cost") or 0.0),
                int(row.get("llm_calls") or 0),
                int(row.get("tool_calls") or 0),
                int(row.get("span_count") or 0),
                row.get("updated_at") or time.time(),
            ),
        )

    def _upsert_span(self, conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO spans(
                span_id,run_id,parent_span_id,trace_path,kind,name,status,started_at,ended_at,duration_ms,
                model,provider,operation,input_summary,output_summary,error_type,error_message,
                attributes_json,usage_json,cost_json,score,feedback,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                row["span_id"],
                row["run_id"],
                row.get("parent_span_id"),
                row.get("trace_path"),
                row.get("kind") or SpanKind.CUSTOM.value,
                row.get("name") or "unnamed",
                row.get("status") or SpanStatus.RUNNING.value,
                row.get("started_at") or time.time(),
                row.get("ended_at"),
                row.get("duration_ms"),
                row.get("model"),
                row.get("provider"),
                row.get("operation"),
                row.get("input_summary"),
                row.get("output_summary"),
                row.get("error_type"),
                row.get("error_message"),
                row.get("attributes_json"),
                row.get("usage_json"),
                row.get("cost_json"),
                row.get("score"),
                row.get("feedback"),
                row.get("updated_at") or time.time(),
            ),
        )

    @staticmethod
    def _row(conn: sqlite3.Connection, query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None

    # read models
    def list_runs(
        self,
        limit: int = 50,
        cursor: Optional[str] = None,
        service_name: Optional[str] = None,
        status: Optional[str] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        limit = max(1, min(limit, 500))
        offset = int(cursor or 0)
        query = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []
        if service_name:
            query += " AND service_name=?"
            params.append(service_name)
        if status:
            query += " AND status=?"
            params.append(status)
        if from_ts is not None:
            query += " AND started_at>=?"
            params.append(from_ts)
        if to_ts is not None:
            query += " AND started_at<=?"
            params.append(to_ts)
        query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        items = [self._run_public(dict(r)) for r in rows]
        return {"items": items, "next_cursor": str(offset + len(items)) if len(items) == limit else None}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._run_public(dict(row), include_metadata=True) if row else None

    def get_run_spans(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM spans WHERE run_id=? ORDER BY started_at ASC, span_id ASC", (run_id,)).fetchall()
        return [self._span_public(dict(r)) for r in rows]

    def get_run_artifacts(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at ASC", (run_id,)).fetchall()
        return [self._artifact_public(dict(r)) for r in rows]

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return self._artifact_public(dict(row)) if row else None

    def get_artifact_content(self, artifact_id: str) -> Optional[Any]:
        art = self.get_artifact(artifact_id)
        if not art:
            return None
        if art.get("inline_text") is not None:
            return art["inline_text"]
        if art.get("storage_uri"):
            return self.get_blob(art["storage_uri"])
        return art.get("preview")

    def compose_run_detail(self, run_id: str) -> Optional[Dict[str, Any]]:
        run = self.get_run(run_id)
        if not run:
            return None
        spans = self.get_run_spans(run_id)
        artifacts = self.get_run_artifacts(run_id)
        return {
            "run": run,
            "spans": spans,
            "artifacts": artifacts,
            "graph": self.build_graph(spans),
            "timeline": {"items": sorted(spans, key=lambda s: ((s.get("started_at") or 0), s.get("span_id") or ""))},
        }

    def build_graph(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes = []
        edges = []
        for span in spans:
            nodes.append(
                {
                    "id": span["span_id"],
                    "type": span.get("kind") or SpanKind.CUSTOM.value,
                    "data": {
                        "label": span.get("name"),
                        "duration_ms": span.get("duration_ms"),
                        "status": span.get("status"),
                    },
                }
            )
            if span.get("parent_span_id"):
                edges.append({"id": f"{span['parent_span_id']}->{span['span_id']}", "source": span["parent_span_id"], "target": span["span_id"]})
        return {"nodes": nodes, "edges": edges}

    def add_feedback(self, score: int, run_id: Optional[str] = None, span_id: Optional[str] = None, comment: Optional[str] = None) -> Dict[str, Any]:
        fid = f"fb_{uuid.uuid4().hex}"
        ts = time.time()
        with self._get_connection() as conn:
            conn.execute("INSERT INTO feedback(feedback_id,run_id,span_id,score,comment,created_at) VALUES(?,?,?,?,?,?)", (fid, run_id, span_id, score, comment, ts))
            if span_id:
                conn.execute("UPDATE spans SET score=?,feedback=?,updated_at=? WHERE span_id=?", (score, comment, ts, span_id))
            conn.commit()
        return {"feedback_id": fid, "run_id": run_id, "span_id": span_id, "score": score, "comment": comment, "created_at": ts}

    # compatibility api
    def save_span(self, span: Span, inputs: Any = None, outputs: Any = None):
        if not span.run_id:
            span.run_id = f"run_{uuid.uuid4().hex}"
        if not self.get_run(span.run_id):
            self.upsert_run(Run(run_id=span.run_id, service_name="legacy", started_at=span.started_at, input_summary=span.input_summary or span.input_preview))
        self.upsert_span(span)
        if inputs is not None:
            self.save_artifact(Artifact(run_id=span.run_id, span_id=span.span_id, role="input.message", content_type="application/json", preview=_preview(inputs)), content=inputs)
        if outputs is not None:
            self.save_artifact(Artifact(run_id=span.run_id, span_id=span.span_id, role="output.message", content_type="application/json", preview=_preview(outputs)), content=outputs)

    def get_span(self, span_id: str, include_blobs: bool = False) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM spans WHERE span_id=?", (span_id,)).fetchone()
        if not row:
            return None
        span = self._span_public(dict(row))
        if include_blobs:
            artifacts = [a for a in self.get_run_artifacts(span["run_id"]) if a["span_id"] == span_id]
            span["inputs"] = [a for a in artifacts if a.get("role", "").startswith("input")]
            span["outputs"] = [a for a in artifacts if a.get("role", "").startswith("output")]
        return span

    def get_trace_spans(self, trace_id: str, include_blobs: bool = False) -> List[Dict[str, Any]]:
        spans = self.get_run_spans(trace_id)
        if include_blobs:
            artifacts = self.get_run_artifacts(trace_id)
            by_span: Dict[str, List[Dict[str, Any]]] = {}
            for art in artifacts:
                by_span.setdefault(art["span_id"], []).append(art)
            for span in spans:
                arts = by_span.get(span["span_id"], [])
                span["inputs"] = [a for a in arts if a.get("role", "").startswith("input")]
                span["outputs"] = [a for a in arts if a.get("role", "").startswith("output")]
        return spans

    def list_traces(self, limit: int = 50, offset: int = 0, status: Optional[str] = None) -> List[Dict[str, Any]]:
        result = self.list_runs(limit=limit, cursor=str(offset), status=status)
        return [
            {
                "trace_id": run["run_id"],
                "name": run.get("input_summary") or run.get("service_name"),
                "start_time": run.get("started_at"),
                "end_time": run.get("ended_at"),
                "duration_ms": run.get("duration_ms"),
                "status": run.get("status"),
                "span_count": run.get("span_count", 0),
                "llm_calls": run.get("llm_calls", 0),
                "tool_calls": run.get("tool_calls", 0),
                "total_tokens": run.get("total_tokens", 0),
                "total_cost": run.get("total_cost", 0.0),
            }
            for run in result["items"]
        ]

    def search_traces(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return self.list_traces(limit=limit)
        like = f"%{q}%"
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT r.* FROM runs r
                LEFT JOIN spans s ON s.run_id=r.run_id
                LEFT JOIN artifacts a ON a.run_id=r.run_id
                WHERE r.run_id LIKE ? OR r.service_name LIKE ? OR COALESCE(r.input_summary,'') LIKE ?
                  OR COALESCE(r.output_summary,'') LIKE ? OR COALESCE(s.name,'') LIKE ?
                  OR COALESCE(s.input_summary,'') LIKE ? OR COALESCE(s.output_summary,'') LIKE ?
                  OR COALESCE(a.inline_text,'') LIKE ?
                ORDER BY r.started_at DESC LIMIT ?
                """,
                (like, like, like, like, like, like, like, like, limit),
            ).fetchall()
        out = []
        for r in rows:
            run = self._run_public(dict(r))
            out.append(
                {
                    "trace_id": run["run_id"],
                    "name": run.get("input_summary") or run.get("service_name"),
                    "start_time": run.get("started_at"),
                    "end_time": run.get("ended_at"),
                    "duration_ms": run.get("duration_ms"),
                    "status": run.get("status"),
                    "span_count": run.get("span_count", 0),
                    "llm_calls": run.get("llm_calls", 0),
                    "tool_calls": run.get("tool_calls", 0),
                    "total_tokens": run.get("total_tokens", 0),
                    "total_cost": run.get("total_cost", 0.0),
                }
            )
        return out

    def delete_run(self, run_id: str) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM feedback WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM spans WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM event_log WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
            conn.commit()

    def delete_trace(self, trace_id: str):
        self.delete_run(trace_id)

    def score_span(self, span_id: str, score: int, feedback: Optional[str] = None):
        with self._get_connection() as conn:
            row = conn.execute("SELECT run_id FROM spans WHERE span_id=?", (span_id,)).fetchone()
        run_id = row["run_id"] if row else None
        self.add_feedback(score=score, run_id=run_id, span_id=span_id, comment=feedback)

    def get_stats(self, hours: int = 24) -> Dict[str, Any]:
        cutoff = time.time() - hours * 3600
        with self._get_connection() as conn:
            spans = conn.execute("SELECT * FROM spans WHERE started_at>=?", (cutoff,)).fetchall()
            run_count = conn.execute("SELECT COUNT(*) AS c FROM runs WHERE started_at>=?", (cutoff,)).fetchone()["c"]
        parsed = [self._span_public(dict(s)) for s in spans]
        durations = [float(s["duration_ms"]) for s in parsed if s.get("duration_ms") is not None]
        errors = sum(1 for s in parsed if s.get("status") == SpanStatus.ERROR.value)
        llm = sum(1 for s in parsed if s.get("kind") == SpanKind.LLM.value)
        tool = sum(1 for s in parsed if s.get("kind") == SpanKind.TOOL.value)
        tokens = sum(int((s.get("usage") or {}).get("total_tokens") or 0) for s in parsed)
        cost = sum(float((s.get("cost") or {}).get("total_cost") or (s.get("usage") or {}).get("total_cost") or 0.0) for s in parsed)
        return {
            "trace_count": run_count,
            "run_count": run_count,
            "span_count": len(parsed),
            "error_count": errors,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "llm_calls": llm,
            "tool_calls": tool,
            "total_tokens": tokens,
            "total_cost": cost,
        }

    def analytics_overview(self, hours: int = 24, service_name: Optional[str] = None) -> Dict[str, Any]:
        cutoff = time.time() - hours * 3600
        with self._get_connection() as conn:
            q = "SELECT * FROM runs WHERE started_at>=?"
            params: List[Any] = [cutoff]
            if service_name:
                q += " AND service_name=?"
                params.append(service_name)
            runs_rows = conn.execute(q, params).fetchall()
            run_ids = [r["run_id"] for r in runs_rows]
            spans_rows: List[sqlite3.Row] = []
            if run_ids:
                ph = ",".join(["?" for _ in run_ids])
                spans_rows = conn.execute(f"SELECT * FROM spans WHERE run_id IN ({ph})", run_ids).fetchall()

        runs = [self._run_public(dict(r)) for r in runs_rows]
        spans = [self._span_public(dict(s)) for s in spans_rows]
        durations = [float(s["duration_ms"]) for s in spans if s.get("duration_ms") is not None]
        errors = sum(1 for s in spans if s.get("status") == SpanStatus.ERROR.value)

        by_type: Dict[str, Dict[str, Any]] = {}
        by_model: Dict[str, Dict[str, Any]] = {}
        ts: Dict[str, Dict[str, Any]] = {}

        for span in spans:
            kind = span.get("kind") or SpanKind.CUSTOM.value
            t = by_type.setdefault(kind, {"span_type": kind, "count": 0, "durations": [], "errors": 0})
            t["count"] += 1
            if span.get("duration_ms") is not None:
                t["durations"].append(float(span["duration_ms"]))
            if span.get("status") == SpanStatus.ERROR.value:
                t["errors"] += 1

            model = span.get("model")
            if model:
                usage = span.get("usage") or {}
                cost = span.get("cost") or {}
                m = by_model.setdefault(model, {"model": model, "count": 0, "tokens": 0, "cost": 0.0, "durations": []})
                m["count"] += 1
                m["tokens"] += int(usage.get("total_tokens") or 0)
                m["cost"] += float(cost.get("total_cost") or usage.get("total_cost") or 0.0)
                if span.get("duration_ms") is not None:
                    m["durations"].append(float(span["duration_ms"]))

            st = span.get("started_at")
            if st is not None:
                hour = datetime.utcfromtimestamp(float(st)).replace(minute=0, second=0, microsecond=0).isoformat()
                usage = span.get("usage") or {}
                cost = span.get("cost") or {}
                item = ts.setdefault(hour, {"hour": hour, "spans": 0, "tokens": 0, "cost": 0.0, "errors": 0})
                item["spans"] += 1
                item["tokens"] += int(usage.get("total_tokens") or 0)
                item["cost"] += float(cost.get("total_cost") or usage.get("total_cost") or 0.0)
                if span.get("status") == SpanStatus.ERROR.value:
                    item["errors"] += 1

        by_type_out = []
        for item in by_type.values():
            d = item.pop("durations")
            item["avg_duration_ms"] = round(sum(d) / len(d), 2) if d else 0.0
            item["p95_duration_ms"] = _percentile(d, 95)
            by_type_out.append(item)
        by_type_out.sort(key=lambda x: x["count"], reverse=True)

        by_model_out = []
        for item in by_model.values():
            d = item.pop("durations")
            item["avg_duration_ms"] = round(sum(d) / len(d), 2) if d else 0.0
            item["cost"] = round(item["cost"], 6)
            by_model_out.append(item)
        by_model_out.sort(key=lambda x: x["count"], reverse=True)

        timeseries = sorted(ts.values(), key=lambda x: x["hour"])
        for p in timeseries:
            p["cost"] = round(p["cost"], 6)

        total_tokens = sum(int((s.get("usage") or {}).get("total_tokens") or 0) for s in spans)
        total_cost = sum(float((s.get("cost") or {}).get("total_cost") or (s.get("usage") or {}).get("total_cost") or 0.0) for s in spans)

        overview = {
            "run_count": len(runs),
            "trace_count": len(runs),
            "span_count": len(spans),
            "error_count": errors,
            "error_rate": round((errors / len(spans)) * 100.0, 2) if spans else 0.0,
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
            "p95_duration_ms": _percentile(durations, 95),
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 6),
        }
        return {"overview": overview, "by_type": by_type_out, "by_model": by_model_out, "timeseries": timeseries}

    def analytics_prompts(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT a.inline_text,a.preview_json,a.created_at,s.status,s.duration_ms,s.usage_json,s.cost_json
                FROM artifacts a
                LEFT JOIN spans s ON s.span_id=a.span_id
                WHERE a.role IN ('input.message','input.text')
                ORDER BY a.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        buckets: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            prompt = r["inline_text"]
            if prompt is None:
                preview = _json_load(r["preview_json"], None)
                if isinstance(preview, str):
                    prompt = preview
                elif preview is not None:
                    prompt = json.dumps(preview, ensure_ascii=False, default=str)
            if not prompt:
                continue
            prompt = prompt.strip()
            if len(prompt) > 1200:
                prompt = prompt[:1200] + "..."

            b = buckets.setdefault(prompt, {"prompt_text": prompt, "count": 0, "error_count": 0, "latencies": [], "costs": [], "last_used": 0.0})
            b["count"] += 1
            if r["status"] == SpanStatus.ERROR.value:
                b["error_count"] += 1
            if r["duration_ms"] is not None:
                b["latencies"].append(float(r["duration_ms"]))
            usage = _json_load(r["usage_json"], {})
            cost = _json_load(r["cost_json"], {})
            b["costs"].append(float(cost.get("total_cost") or usage.get("total_cost") or 0.0))
            b["last_used"] = max(b["last_used"], float(r["created_at"] or 0.0))

        out = []
        for b in buckets.values():
            c = b["count"]
            out.append(
                {
                    "prompt_text": b["prompt_text"],
                    "count": c,
                    "error_rate": round((b["error_count"] / c) * 100.0, 2) if c else 0.0,
                    "last_used": b["last_used"],
                    "avg_latency": round(sum(b["latencies"]) / len(b["latencies"]), 2) if b["latencies"] else 0.0,
                    "avg_cost": round(sum(b["costs"]) / len(b["costs"]), 6) if b["costs"] else 0.0,
                }
            )
        out.sort(key=lambda x: (x["count"], x["last_used"]), reverse=True)
        return out

    def _run_public(self, row: Dict[str, Any], include_metadata: bool = False) -> Dict[str, Any]:
        out = {
            "run_id": row.get("run_id"),
            "service_name": row.get("service_name"),
            "service_version": row.get("service_version"),
            "environment": row.get("environment"),
            "session_id": row.get("session_id"),
            "workflow_id": row.get("workflow_id"),
            "root_span_id": row.get("root_span_id"),
            "status": row.get("status"),
            "started_at": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "duration_ms": row.get("duration_ms"),
            "input_summary": row.get("input_summary"),
            "output_summary": row.get("output_summary"),
            "error_message": row.get("error_message"),
            "total_tokens": int(row.get("total_tokens") or 0),
            "total_cost": float(row.get("total_cost") or 0.0),
            "llm_calls": int(row.get("llm_calls") or 0),
            "tool_calls": int(row.get("tool_calls") or 0),
            "span_count": int(row.get("span_count") or 0),
        }
        if include_metadata:
            out["metadata"] = _json_load(row.get("metadata_json"), {})
        return out

    def _span_public(self, row: Dict[str, Any]) -> Dict[str, Any]:
        attrs = _json_load(row.get("attributes_json"), {})
        usage = _json_load(row.get("usage_json"), {})
        cost = _json_load(row.get("cost_json"), {})
        return {
            "span_id": row.get("span_id"),
            "run_id": row.get("run_id"),
            "trace_id": row.get("run_id"),
            "parent_span_id": row.get("parent_span_id"),
            "trace_path": row.get("trace_path"),
            "kind": row.get("kind"),
            "span_type": row.get("kind"),
            "name": row.get("name"),
            "status": row.get("status"),
            "started_at": row.get("started_at"),
            "start_time": row.get("started_at"),
            "ended_at": row.get("ended_at"),
            "end_time": row.get("ended_at"),
            "duration_ms": row.get("duration_ms"),
            "model": row.get("model"),
            "provider": row.get("provider"),
            "operation": row.get("operation"),
            "input_summary": row.get("input_summary"),
            "output_summary": row.get("output_summary"),
            "input_preview": row.get("input_summary"),
            "output_preview": row.get("output_summary"),
            "error_type": row.get("error_type"),
            "error_message": row.get("error_message"),
            "attributes": attrs,
            "metadata": attrs,
            "usage": usage,
            "cost": cost,
            "score": row.get("score"),
            "feedback": row.get("feedback"),
        }

    def _artifact_public(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "artifact_id": row.get("artifact_id"),
            "run_id": row.get("run_id"),
            "span_id": row.get("span_id"),
            "role": row.get("role"),
            "content_type": row.get("content_type"),
            "storage_uri": row.get("storage_uri"),
            "inline_text": row.get("inline_text"),
            "preview": _json_load(row.get("preview_json"), None),
            "metadata": _json_load(row.get("metadata_json"), {}),
            "size_bytes": row.get("size_bytes"),
            "sha256": row.get("sha256"),
            "created_at": row.get("created_at"),
            "url": f"/v1/artifacts/{row.get('artifact_id')}/content",
        }

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


_storage: Optional[StorageEngine] = None


def get_storage(data_dir: Optional[Path] = None) -> StorageEngine:
    global _storage
    if _storage is None:
        _storage = StorageEngine(data_dir)
    return _storage


def _json_load(raw: Optional[str], default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _preview(data: Any, max_len: int = 1200) -> Any:
    if data is None:
        return None
    if isinstance(data, str):
        return data[:max_len]
    if isinstance(data, bytes):
        try:
            return data.decode("utf-8", errors="replace")[:max_len]
        except Exception:
            return f"<bytes:{len(data)}>"
    if isinstance(data, (dict, list, tuple)):
        txt = json.dumps(data, ensure_ascii=False, default=str)
        return data if len(txt) <= max_len else txt[:max_len] + "..."
    return str(data)[:max_len]


def _percentile(values: List[float], pct: int) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return round(vals[0], 2)
    rank = (pct / 100.0) * (len(vals) - 1)
    low = int(rank)
    high = min(low + 1, len(vals) - 1)
    weight = rank - low
    val = vals[low] * (1.0 - weight) + vals[high] * weight
    return round(val, 2)
