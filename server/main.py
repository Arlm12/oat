"""
OpenAgentTrace server implementing canonical Run -> Span -> Artifact APIs.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import json
import os
import re
import sqlite3
import time
import uuid

import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from oat.models import DEFAULT_SCHEMA_VERSION, EventEnvelope, Run, Span
from oat.media import extract_openai_multimodal_artifacts
from oat.pricing import calculate_cost
from oat.providers import resolve_service_provider
from oat.storage import StorageEngine, get_storage


DATA_DIR = Path(".oat")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROVIDER_CONFIG_ENV = "OAT_PROVIDER_CONFIG"
_DEFAULT_PROVIDER_CONFIG = "providers.yaml"
storage: Optional[StorageEngine] = None
_provider_discovery_cache: Dict[str, Dict[str, Any]] = {}
_provider_discovery_ttl = float(os.getenv("OAT_PROVIDER_DISCOVERY_TTL", "120"))


def _load_env_files() -> None:
    """Load .env-style files into process env without overriding existing vars."""
    for name in (".env", ".env.local"):
        path = Path(name)
        if not path.exists():
            continue
        try:
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"'").strip()
                if key:
                    os.environ.setdefault(key, value)
        except Exception:
            continue


_load_env_files()


def _provider_catalog_path() -> Path:
    raw = os.getenv(_PROVIDER_CONFIG_ENV, _DEFAULT_PROVIDER_CONFIG)
    candidate = Path(raw).expanduser()
    candidates: List[Path] = []

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.extend(
            [
                (Path.cwd() / candidate).resolve(),
                (PROJECT_ROOT / candidate).resolve(),
                (Path(__file__).resolve().parent / candidate).resolve(),
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    # Return most likely expected location for clear error/warning messaging.
    return candidates[1] if len(candidates) > 1 else candidates[0]


class SpanIngest(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    span_type: str = "function"
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "running"
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    inputs: Any = None
    outputs: Any = None
    model: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    media_inputs: Optional[List[Dict[str, Any]]] = None


class FeedbackRequest(BaseModel):
    score: int = Field(..., ge=-1, le=1)
    feedback: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 50


class IngestEventsRequest(BaseModel):
    schema_version: str = DEFAULT_SCHEMA_VERSION
    events: List[Dict[str, Any]]


class IngestRunRequest(BaseModel):
    run: Dict[str, Any]
    spans: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)


class ArenaRunRequest(BaseModel):
    prompt: str
    system_prompt: str = "You are a helpful assistant."
    model: str


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        stale: List[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                stale.append(conn)
        for conn in stale:
            self.disconnect(conn)


manager = ConnectionManager()


def _storage() -> StorageEngine:
    global storage
    if storage is None:
        storage = get_storage(DATA_DIR)
    return storage


def _run_public_storage_query(sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    conn = sqlite3.connect((_storage().db_path))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


_STALE_SPAN_TIMEOUT_S: float = float(os.getenv("OAT_STALE_SPAN_TIMEOUT", "600"))  # 10 minutes
_STALE_SPAN_CHECK_INTERVAL_S: float = 60.0


async def _housekeeping_loop() -> None:
    """Background task: mark spans/runs that are still RUNNING after the
    configured timeout as CANCELLED so dashboards don't show stale entries
    after a client process crash."""
    while True:
        await asyncio.sleep(_STALE_SPAN_CHECK_INTERVAL_S)
        try:
            cutoff = time.time() - _STALE_SPAN_TIMEOUT_S
            count = _storage().mark_stale_spans_cancelled(cutoff)
            if count:
                import logging
                logging.getLogger("oat.server").info("Cleaned up %d stale span(s)", count)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global storage
    _load_env_files()
    DATA_DIR.mkdir(exist_ok=True)
    storage = get_storage(DATA_DIR)
    task = asyncio.create_task(_housekeeping_loop())
    try:
        yield
    finally:
        task.cancel()
        if storage:
            storage.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenAgentTrace",
        description="Canonical telemetry backend for AI agent observability",
        version="0.2.0",
        lifespan=lifespan,
    )
    cors_origins_env = os.getenv("OAT_CORS_ORIGINS", "")
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else ["*"]
    # allow_credentials requires explicit origins (not "*"). Use False with wildcard.
    allow_credentials = cors_origins != ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


app = create_app()


def _extract_multimodal_inputs(messages: Any) -> List[Dict[str, Any]]:
    return extract_openai_multimodal_artifacts(messages)


def _legacy_span_to_events(span: SpanIngest) -> List[Dict[str, Any]]:
    run_id = span.trace_id
    events: List[Dict[str, Any]] = []
    now = time.time()

    run_payload = {
        "run_id": run_id,
        "service_name": span.metadata.get("service_name", "legacy-sdk"),
        "started_at": span.start_time,
        "status": "running",
        "input_summary": span.input_preview,
        "metadata": {"legacy_ingest": True},
    }
    events.append(EventEnvelope(event_type="run.started", run_id=run_id, payload=run_payload, emitted_at=span.start_time).to_dict())

    span_payload = {
        "span_id": span.span_id,
        "run_id": run_id,
        "parent_span_id": span.parent_span_id,
        "name": span.name,
        "kind": span.span_type,
        "status": span.status,
        "started_at": span.start_time,
        "ended_at": span.end_time,
        "duration_ms": span.duration_ms,
        "model": span.model,
        "error_message": span.error_message,
        "error_type": span.error_type,
        "input_summary": span.input_preview,
        "output_summary": span.output_preview,
        "usage": span.usage or {},
        "attributes": span.metadata or {},
    }
    events.append(EventEnvelope(event_type="span.started", run_id=run_id, span_id=span.span_id, payload=span_payload, emitted_at=span.start_time).to_dict())
    if span.end_time is not None or span.status != "running":
        events.append(EventEnvelope(event_type="span.finished", run_id=run_id, span_id=span.span_id, payload=span_payload, emitted_at=span.end_time or now).to_dict())

    if span.inputs is not None:
        events.append(
            EventEnvelope(
                event_type="artifact.created",
                run_id=run_id,
                span_id=span.span_id,
                emitted_at=span.start_time,
                payload={
                    "artifact_id": f"art_{span.span_id}_in",
                    "role": "input.message",
                    "content_type": "application/json",
                    "preview": span.input_preview or span.inputs,
                    "content": span.inputs,
                },
            ).to_dict()
        )

        if isinstance(span.inputs, dict):
            messages = span.inputs.get("messages")
            for item in _extract_multimodal_inputs(messages):
                events.append(
                    EventEnvelope(
                        event_type="artifact.created",
                        run_id=run_id,
                        span_id=span.span_id,
                        payload={
                            "artifact_id": f"art_{uuid.uuid4().hex}",
                            "role": item["role"],
                            "content_type": item["content_type"],
                            "preview": item.get("preview"),
                            "metadata": item.get("metadata"),
                            "inline_text": item.get("inline_text"),
                            "content": item.get("content"),
                        },
                        emitted_at=span.start_time,
                    ).to_dict()
                )

    if span.outputs is not None:
        events.append(
            EventEnvelope(
                event_type="artifact.created",
                run_id=run_id,
                span_id=span.span_id,
                emitted_at=span.end_time or now,
                payload={
                    "artifact_id": f"art_{span.span_id}_out",
                    "role": "output.message",
                    "content_type": "application/json",
                    "preview": span.output_preview or span.outputs,
                    "content": span.outputs,
                },
            ).to_dict()
        )

        if isinstance(span.outputs, dict) and span.outputs.get("content"):
            events.append(
                EventEnvelope(
                    event_type="artifact.created",
                    run_id=run_id,
                    span_id=span.span_id,
                    emitted_at=span.end_time or now,
                    payload={
                        "artifact_id": f"art_{uuid.uuid4().hex}",
                        "role": "output.text",
                        "content_type": "text/plain",
                        "inline_text": str(span.outputs["content"]),
                        "preview": str(span.outputs["content"])[:1200],
                    },
                ).to_dict()
            )

        if isinstance(span.outputs, dict):
            transcript = span.outputs.get("transcript") or span.outputs.get("audio_transcript")
            if transcript:
                events.append(
                    EventEnvelope(
                        event_type="artifact.created",
                        run_id=run_id,
                        span_id=span.span_id,
                        emitted_at=span.end_time or now,
                        payload={
                            "artifact_id": f"art_{uuid.uuid4().hex}",
                            "role": "derived.audio_transcript",
                            "content_type": "text/plain",
                            "inline_text": str(transcript),
                            "preview": str(transcript)[:1200],
                        },
                    ).to_dict()
                )

    if span.media_inputs:
        events.append(
            EventEnvelope(
                event_type="artifact.created",
                run_id=run_id,
                span_id=span.span_id,
                emitted_at=span.start_time,
                payload={
                    "artifact_id": f"art_{uuid.uuid4().hex}",
                    "role": "derived.media_analysis",
                    "content_type": "application/json",
                    "preview": span.media_inputs,
                    "metadata": {"source": "legacy.media_inputs"},
                    "content": span.media_inputs,
                },
            ).to_dict()
        )

    if span.end_time is not None:
        events.append(
            EventEnvelope(
                event_type="run.finished",
                run_id=run_id,
                emitted_at=span.end_time,
                payload={"run_id": run_id, "ended_at": span.end_time, "status": "success"},
            ).to_dict()
        )
    return events


def _load_provider_catalog() -> Dict[str, Any]:
    _load_env_files()
    provider_config_path = _provider_catalog_path()
    if not provider_config_path.exists():
        return {"providers": {}}
    try:
        data = yaml.safe_load(provider_config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return {"providers": {}}
        if "providers" not in data or not isinstance(data["providers"], dict):
            return {"providers": {}}
        return data
    except Exception:
        return {"providers": {}}


def _resolve_env_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw)
        if match:
            return os.getenv(match.group(1))
        return os.getenv(raw)
    if isinstance(value, list):
        for item in value:
            val = _resolve_env_value(item)
            if val:
                return val
    return None


def _resolve_secret_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            resolved = _resolve_secret_value(item)
            if resolved:
                return resolved
        return None
    if not isinstance(value, str):
        return str(value)
    raw = value.strip()
    if not raw:
        return None
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw)
    if match:
        return os.getenv(match.group(1))
    # If this looks like an env var name, use env lookup first.
    if re.fullmatch(r"[A-Z0-9_]+", raw):
        from_env = os.getenv(raw)
        if from_env:
            return from_env
        return None
    return raw


def _provider_api_key(provider_cfg: Dict[str, Any]) -> Optional[str]:
    direct = _resolve_secret_value(provider_cfg.get("api_key"))
    if direct:
        return direct
    return _resolve_env_value(provider_cfg.get("api_key_env"))


def _provider_base_url(provider_cfg: Dict[str, Any]) -> Optional[str]:
    return (
        _resolve_secret_value(provider_cfg.get("base_url"))
        or _resolve_env_value(provider_cfg.get("base_url_env"))
        or _resolve_secret_value(provider_cfg.get("default_base_url"))
    )


def _provider_headers(provider_cfg: Dict[str, Any], api_key: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    raw_headers = provider_cfg.get("headers")
    if isinstance(raw_headers, dict):
        for key, value in raw_headers.items():
            resolved = _resolve_secret_value(value)
            if resolved:
                headers[str(key)] = str(resolved)
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _discover_models_from_provider(
    provider_key: str,
    provider_cfg: Dict[str, Any],
    api_key: Optional[str],
    base_url: Optional[str],
) -> List[Dict[str, str]]:
    if not provider_cfg.get("discover_models"):
        return []
    if not base_url:
        return []

    now = time.time()
    cache_key = f"{provider_key}|{base_url}"
    cached = _provider_discovery_cache.get(cache_key)
    if cached and (now - float(cached.get("at", 0.0)) < _provider_discovery_ttl):
        return list(cached.get("models", []))

    mode = str(provider_cfg.get("model_discovery") or "").strip().lower()
    if not mode:
        if provider_cfg.get("type") == "ollama" or provider_key == "ollama":
            mode = "ollama"
        elif "openrouter" in provider_key or "openrouter" in base_url:
            mode = "openrouter"
        else:
            mode = "openai"

    limit = int(provider_cfg.get("discovery_limit") or 100)
    headers = _provider_headers(provider_cfg, api_key)

    discovered: List[Dict[str, str]] = []
    try:
        import httpx

        if mode == "ollama":
            url = f"{base_url.rstrip('/')}/api/tags"
            payload = httpx.get(url, timeout=6.0, headers=headers).json()
            for row in payload.get("models", []) or []:
                name = row.get("name")
                if not name:
                    continue
                discovered.append(
                    {
                        "id": f"{provider_key}:{name}",
                        "model": str(name),
                        "name": str(name),
                    }
                )
        else:
            urls = [
                f"{base_url.rstrip('/')}/models",
                f"{base_url.rstrip('/')}/v1/models",
            ]
            payload = None
            for url in urls:
                try:
                    resp = httpx.get(url, timeout=6.0, headers=headers)
                    if resp.status_code >= 400:
                        continue
                    payload = resp.json()
                    break
                except Exception:
                    continue
            if isinstance(payload, dict):
                for row in payload.get("data", []) or []:
                    model_name = row.get("id") or row.get("name")
                    if not model_name:
                        continue
                    discovered.append(
                        {
                            "id": f"{provider_key}:{model_name}",
                            "model": str(model_name),
                            "name": str(model_name),
                        }
                    )
    except Exception:
        discovered = []

    deduped: Dict[str, Dict[str, str]] = {}
    for item in discovered:
        deduped[item["id"]] = item
    discovered = list(deduped.values())[: max(1, limit)]

    _provider_discovery_cache[cache_key] = {"at": now, "models": discovered}
    return discovered


def _flatten_models() -> Dict[str, Dict[str, Any]]:
    catalog = _load_provider_catalog().get("providers", {})
    by_id: Dict[str, Dict[str, Any]] = {}
    for provider_key, provider in catalog.items():
        if not isinstance(provider, dict):
            continue
        provider_type = str(provider.get("type", provider_key))
        label = str(provider.get("label", provider_key))
        requires_key = bool(provider.get("requires_api_key", True))
        api_key = _provider_api_key(provider)
        base_url = _provider_base_url(provider)
        requires_base_url = bool(
            provider.get(
                "requires_base_url",
                provider_type in {"openai_compatible", "ollama", "self_hosted", "vllm"},
            )
        )
        available = ((not requires_key) or bool(api_key)) and ((not requires_base_url) or bool(base_url))

        static_models = [m for m in (provider.get("models", []) or []) if isinstance(m, dict)]
        include_static = bool(provider.get("include_static_models", True))
        discovered_models = _discover_models_from_provider(
            provider_key=provider_key,
            provider_cfg=provider,
            api_key=api_key,
            base_url=base_url,
        )

        merged: List[Dict[str, Any]] = []
        if include_static or not discovered_models:
            merged.extend(static_models)
        merged.extend(discovered_models)

        for model in merged:
            if not isinstance(model, dict):
                continue
            model_id = model.get("id") or f"{provider_key}:{model.get('model')}"
            model_name = model.get("model") or model_id
            by_id[model_id] = {
                "id": model_id,
                "name": model.get("name") or model_name or model_id,
                "model": model_name or model_id,
                "provider": label,
                "provider_key": provider_key,
                "provider_type": provider_type,
                "available": available,
                "provider_config": provider,
                "base_url": base_url,
            }
    return by_id


# ---------------------------------------------------------------------------
# Ingestion endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/ingest/events")
async def ingest_events(req: IngestEventsRequest):
    accepted = 0
    rejected = 0
    errors: List[Dict[str, Any]] = []

    for idx, event in enumerate(req.events):
        try:
            _storage().save_event(
                {
                    "schema_version": req.schema_version or DEFAULT_SCHEMA_VERSION,
                    "event_type": event.get("event_type"),
                    "emitted_at": event.get("emitted_at") or time.time(),
                    "run_id": event.get("run_id"),
                    "span_id": event.get("span_id"),
                    "payload": event.get("payload") or {},
                }
            )
            accepted += 1
            await manager.broadcast({"type": event.get("event_type"), "run_id": event.get("run_id"), "span_id": event.get("span_id")})
        except Exception as exc:
            rejected += 1
            errors.append({"index": idx, "error": str(exc)})

    return {"accepted": accepted, "rejected": rejected, "errors": errors}


@app.post("/v1/ingest/run")
async def ingest_run(req: IngestRunRequest):
    run_id = req.run.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run.run_id is required")

    events: List[Dict[str, Any]] = []
    run_payload = dict(req.run)
    events.append(EventEnvelope(event_type="run.started", run_id=run_id, payload=run_payload, emitted_at=run_payload.get("started_at") or time.time()).to_dict())

    for span in req.spans:
        sid = span.get("span_id")
        if not sid:
            continue
        events.append(EventEnvelope(event_type="span.started", run_id=run_id, span_id=sid, payload=span, emitted_at=span.get("started_at") or time.time()).to_dict())
        if span.get("ended_at") is not None or span.get("status") != "running":
            events.append(EventEnvelope(event_type="span.finished", run_id=run_id, span_id=sid, payload=span, emitted_at=span.get("ended_at") or time.time()).to_dict())

    for artifact in req.artifacts:
        sid = artifact.get("span_id")
        if not sid:
            continue
        events.append(EventEnvelope(event_type="artifact.created", run_id=run_id, span_id=sid, payload=artifact, emitted_at=artifact.get("created_at") or time.time()).to_dict())

    if run_payload.get("ended_at") is not None or run_payload.get("status") in {"success", "error", "cancelled"}:
        events.append(EventEnvelope(event_type="run.finished", run_id=run_id, payload=run_payload, emitted_at=run_payload.get("ended_at") or time.time()).to_dict())

    return await ingest_events(IngestEventsRequest(schema_version=req.run.get("schema_version", DEFAULT_SCHEMA_VERSION), events=events))


@app.post("/ingest")
async def legacy_ingest_span(span: SpanIngest):
    events = _legacy_span_to_events(span)
    return await ingest_events(IngestEventsRequest(schema_version=DEFAULT_SCHEMA_VERSION, events=events))


# ---------------------------------------------------------------------------
# Canonical read APIs
# ---------------------------------------------------------------------------


@app.get("/v1/runs")
def list_runs(
    service_name: Optional[str] = None,
    status: Optional[str] = None,
    from_ts: Optional[float] = Query(None, alias="from"),
    to_ts: Optional[float] = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=500),
    cursor: Optional[str] = None,
):
    return _storage().list_runs(limit=limit, cursor=cursor, service_name=service_name, status=status, from_ts=from_ts, to_ts=to_ts)


@app.get("/v1/runs/{run_id}")
def get_run_detail(run_id: str):
    detail = _storage().compose_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@app.get("/v1/runs/{run_id}/timeline")
def get_run_timeline(run_id: str):
    detail = _storage().compose_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail["timeline"]


@app.get("/v1/runs/{run_id}/graph")
def get_run_graph(run_id: str):
    detail = _storage().compose_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"graph": detail["graph"]}


@app.get("/v1/runs/{run_id}/artifacts")
def get_run_artifacts(run_id: str):
    run = _storage().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    artifacts = _storage().get_run_artifacts(run_id)
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for art in artifacts:
        grouped.setdefault(art["span_id"], {})
        grouped[art["span_id"]].setdefault(art["role"], [])
        grouped[art["span_id"]][art["role"]].append(art)
    return {"items": artifacts, "grouped": grouped}


@app.get("/v1/artifacts/{artifact_id}/content")
def get_artifact_content(artifact_id: str):
    artifact = _storage().get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    content = _storage().get_artifact_content(artifact_id)
    return {"artifact": artifact, "content": content}


@app.post("/v1/spans/{span_id}/feedback")
def add_span_feedback(span_id: str, req: FeedbackRequest):
    result = _storage().add_feedback(score=req.score, span_id=span_id, comment=req.feedback)
    return {"status": "updated", "span_id": span_id, "feedback_id": result["feedback_id"]}


@app.get("/v1/analytics/overview")
def analytics_overview(hours: int = Query(24, ge=1, le=168), service_name: Optional[str] = None):
    return _storage().analytics_overview(hours=hours, service_name=service_name)


@app.get("/v1/analytics/prompts")
def analytics_prompts(limit: int = Query(200, ge=1, le=1000)):
    return _storage().analytics_prompts(limit=limit)


@app.get("/v1/services")
def list_services():
    rows = _run_public_storage_query("SELECT DISTINCT service_name FROM runs ORDER BY service_name ASC")
    return [r["service_name"] for r in rows if r.get("service_name")]


# ---------------------------------------------------------------------------
# Legacy API compatibility
# ---------------------------------------------------------------------------


@app.get("/traces")
def list_legacy_traces(limit: int = Query(50, ge=1, le=500), offset: int = 0, status: Optional[str] = None):
    return _storage().list_traces(limit=limit, offset=offset, status=status)


@app.get("/traces/{trace_id}")
def get_legacy_trace(trace_id: str, include_blobs: bool = False):
    spans = _storage().get_trace_spans(trace_id, include_blobs=include_blobs)
    if not spans:
        raise HTTPException(status_code=404, detail="Trace not found")
    return spans


@app.get("/spans/{span_id}")
def get_legacy_span(span_id: str, include_blobs: bool = True):
    span = _storage().get_span(span_id, include_blobs=include_blobs)
    if not span:
        raise HTTPException(status_code=404, detail="Span not found")
    return span


@app.delete("/traces/{trace_id}")
def delete_legacy_trace(trace_id: str):
    _storage().delete_trace(trace_id)
    return {"status": "deleted", "trace_id": trace_id}


@app.post("/spans/{span_id}/feedback")
def legacy_feedback(span_id: str, req: FeedbackRequest):
    _storage().score_span(span_id, req.score, req.feedback)
    return {"status": "updated", "span_id": span_id}


@app.post("/search")
def legacy_search(req: SearchRequest):
    return _storage().search_traces(req.query, req.limit)


@app.get("/analytics/overview")
def legacy_analytics_overview(hours: int = Query(24, ge=1, le=168)):
    return _storage().analytics_overview(hours=hours)


@app.get("/analytics/latency")
def legacy_analytics_latency(hours: int = Query(24, ge=1, le=168)):
    data = _storage().analytics_overview(hours=hours)
    return [
        {
            "span_type": row["span_type"],
            "p50": row.get("avg_duration_ms", 0.0),
            "p90": row.get("p90_duration_ms") or row.get("p95_duration_ms", 0.0),
            "p95": row.get("p95_duration_ms", 0.0),
            "p99": row.get("p99_duration_ms") or row.get("p95_duration_ms", 0.0),
            "min": row.get("min_duration_ms", 0.0),
            "max": row.get("max_duration_ms") or row.get("p95_duration_ms", 0.0),
            "count": row.get("count", 0),
        }
        for row in data.get("by_type", [])
    ]


# ---------------------------------------------------------------------------
# Arena APIs
# ---------------------------------------------------------------------------


@app.get("/arena/models")
def arena_models():
    models_by_id = _flatten_models()
    models = [
        {
            "id": m["id"],
            "name": m["name"],
            "model": m["model"],
            "provider": m["provider"],
            "provider_key": m["provider_key"],
            "provider_type": m["provider_type"],
            "available": m["available"],
        }
        for m in models_by_id.values()
    ]
    models.sort(key=lambda x: (not x["available"], x["provider"], x["name"]))
    warning = None
    if not models:
        warning = f"No models discovered from {_provider_catalog_path()}"
    elif not any(m["available"] for m in models):
        warning = "No models are currently available. Check API keys in your .env file."
    return {"models": models, "warning": warning}


@app.post("/arena/run")
def arena_run(req: ArenaRunRequest):
    models = _flatten_models()
    selected = models.get(req.model)
    if not selected:
        raise HTTPException(status_code=404, detail=f"Unknown model id: {req.model}")
    if not selected.get("available"):
        return {"status": "error", "error": f"Model {req.model} is not available (missing credentials)"}

    provider_cfg = selected.get("provider_config") or {}
    provider_type = selected.get("provider_type")
    model_name = selected.get("model")
    base_url = _provider_base_url(provider_cfg)
    api_key = _provider_api_key(provider_cfg)
    headers = _provider_headers(provider_cfg, api_key)
    provider_key = selected.get("provider_key")
    resolved_provider = resolve_service_provider(
        model=model_name,
        base_url=base_url,
        provider_hint=provider_key,
        default=str(provider_type or "openai"),
    )

    started = time.time()
    try:
        if provider_type in {
            "openai",
            "openai_compatible",
            "ollama",
            "mistral",
            "moonshot",
            "kimi",
            "qwen",
            "self_hosted",
            "vllm",
        }:
            from openai import OpenAI

            effective_api_key = api_key or ("local" if provider_type in {"ollama", "self_hosted"} else None)
            if not effective_api_key and provider_type not in {"ollama", "self_hosted"}:
                return {"status": "error", "error": f"Missing API key for provider {provider_key}"}

            client = OpenAI(api_key=effective_api_key, base_url=base_url, default_headers=headers or None)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": req.system_prompt},
                    {"role": "user", "content": req.prompt},
                ],
            )
            output = response.choices[0].message.content if response.choices else ""
            if isinstance(output, list):
                output = "\n".join(
                    str(item.get("text") if isinstance(item, dict) else item)
                    for item in output
                )
            output = str(output or "")
            prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0) if getattr(response, "usage", None) else 0)
            completion_tokens = int(getattr(response.usage, "completion_tokens", 0) if getattr(response, "usage", None) else 0)
            total_tokens = int(getattr(response.usage, "total_tokens", 0) if getattr(response, "usage", None) else (prompt_tokens + completion_tokens))
            pricing = calculate_cost(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider=resolved_provider,
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cost": pricing["prompt_cost"],
                "completion_cost": pricing["completion_cost"],
                "total_cost": pricing["total_cost"],
                "pricing_status": pricing["pricing_status"],
            }
            return {
                "status": "success",
                "provider": resolved_provider,
                "model": model_name,
                "output": output,
                "usage": usage,
                "pricing": pricing["pricing"],
                "duration_ms": (time.time() - started) * 1000.0,
            }

        if provider_type == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_name,
                system=req.system_prompt,
                max_tokens=1024,
                messages=[{"role": "user", "content": req.prompt}],
            )
            text = ""
            if getattr(response, "content", None):
                parts = [p.text for p in response.content if getattr(p, "type", "") == "text"]
                text = "\n".join(parts)
            prompt_tokens = int(getattr(response, "usage", None).input_tokens if getattr(response, "usage", None) else 0)
            completion_tokens = int(getattr(response, "usage", None).output_tokens if getattr(response, "usage", None) else 0)
            pricing = calculate_cost(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider="anthropic",
            )
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_cost": pricing["prompt_cost"],
                "completion_cost": pricing["completion_cost"],
                "total_cost": pricing["total_cost"],
                "pricing_status": pricing["pricing_status"],
            }
            return {
                "status": "success",
                "provider": "anthropic",
                "model": model_name,
                "output": text,
                "usage": usage,
                "pricing": pricing["pricing"],
                "duration_ms": (time.time() - started) * 1000.0,
            }

        if provider_type == "google":
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([req.system_prompt, req.prompt])
            text = getattr(response, "text", "") or ""
            prompt_tokens = int(getattr(response, "usage_metadata", None).prompt_token_count if getattr(response, "usage_metadata", None) else 0)
            completion_tokens = int(getattr(response, "usage_metadata", None).candidates_token_count if getattr(response, "usage_metadata", None) else 0)
            total_tokens = int(getattr(response, "usage_metadata", None).total_token_count if getattr(response, "usage_metadata", None) else (prompt_tokens + completion_tokens))
            pricing = calculate_cost(
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider="google",
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cost": pricing["prompt_cost"],
                "completion_cost": pricing["completion_cost"],
                "total_cost": pricing["total_cost"],
                "pricing_status": pricing["pricing_status"],
            }
            return {
                "status": "success",
                "provider": "google",
                "model": model_name,
                "output": text,
                "usage": usage,
                "pricing": pricing["pricing"],
                "duration_ms": (time.time() - started) * 1000.0,
            }

        return {"status": "error", "error": f"Unsupported provider type: {provider_type}"}

    except Exception as exc:
        return {"status": "error", "error": str(exc), "duration_ms": (time.time() - started) * 1000.0}


# ---------------------------------------------------------------------------
# Realtime + admin + health
# ---------------------------------------------------------------------------


@app.websocket("/v1/ws")
async def canonical_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws")
async def legacy_ws(websocket: WebSocket):
    await canonical_ws(websocket)


@app.delete("/admin/reset")
def admin_reset():
    runs = _storage().list_runs(limit=50000).get("items", [])
    for run in runs:
        _storage().delete_run(run["run_id"])
    return {"status": "all_data_cleared"}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "0.2.0", "data_dir": str(DATA_DIR.absolute())}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8787)


if __name__ == "__main__":
    main()
