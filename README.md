# OpenAgentTrace (OAT)

<div align="center">

**🔍 The Open Standard for AI Agent Observability**


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/yourusername/openagent-trace)


### Contributors
1. Arunachalam - (https://github.com/Arlm12)
</div>

---# OpenAgentTrace Detailed Guide

This document describes the repository as it exists in code on March 5, 2026. 

This file is the code-aligned technical reference for the current repository state.

## Snapshot

- Project name: OpenAgentTrace (OAT)
- Python package version: `0.1.0`
- FastAPI server version string: `0.2.0`
- Canonical telemetry model: `Run -> Span -> Artifact`
- Canonical schema version: `2026-03-01`
- Default local storage: SQLite database plus blob files under `.oat/`
- Backend API: FastAPI in `server/`
- Dashboard UI: React + Vite app in `ui/`
- Current examples: `examples/demo_agent.py`, `examples/multimodal_image_agent.py`

## What OAT Is Today

OpenAgentTrace is currently a local-first observability stack for agentic and LLM applications.

At the code level, the project is built around five pieces:

1. A Python SDK in `oat/` that creates runs, spans, and artifacts.
2. A local storage engine that writes canonical telemetry events into SQLite and blob files.
3. Optional HTTP export from the SDK to a backend server.
4. A FastAPI backend in `server/main.py` that ingests events and serves run, analytics, artifact, and arena APIs.
5. A React dashboard in `ui/` that visualizes runs, graphs, prompts, analytics, and model comparisons.

A practical way to think about the system is:

- The SDK always records locally first.
- Export to the server is optional.
- The dashboard only shows what exists in the server's storage, not whatever was recorded locally by another process unless that process exports to the server.

That last point is important. If you run an agent locally without `export_url` or `OAT_EXPORT_URL`, you can still have traces on disk, but they will not appear in the dashboard.

## Repository Layout

```text
OpenAgentTrace/
|-- oat/                      # Core SDK, models, storage, integrations, pricing
|-- server/                   # FastAPI backend and ingest API
|-- ui/                       # React/Vite dashboard
|-- examples/                 # Example agents and test image
|-- providers.yaml            # Arena provider catalog
|-- tracer.yaml               # Example tracer config file
|-- start.sh                  # Linux/macOS quick-start script
|-- start.ps1                 # Windows quick-start script
|-- README.md                 # Short overview
|-- README_DETAILED.md        # This detailed guide
|-- ARCHITECTURE.md           # Older standalone architecture notes
|-- pyproject.toml            # Python packaging and extras
|-- requirements.txt          # Flat dependency list for convenience
```

## Architecture

This section describes the current runtime architecture of the repository.

### Component Topology

```text
User application
    |
    | trace decorators / context managers / explicit tracer calls
    v
OAT SDK (`oat/`)
    |
    | canonical events
    | (run.started, span.started, span.finished, artifact.created, run.finished)
    +------------------------------+
    |                              |
    | local-first persistence      | optional export
    v                              v
Local `.oat/` storage          FastAPI server (`server/main.py`)
(`telemetry.db` + blobs)           |
                                   | same StorageEngine materialization model
                                   v
                           Server `.oat/` storage
                                   |
                                   | REST + WebSocket APIs
                                   v
                           React dashboard (`ui/`)
```
### Dashboard V2 Images
<img width="2481" height="1402" alt="Screenshot 2026-03-05 095901" src="https://github.com/user-attachments/assets/0a024b88-bbab-46a5-a4f4-2b3183221434" />
<img width="2481" height="1396" alt="Screenshot 2026-03-05 100021" src="https://github.com/user-attachments/assets/e2629612-afb4-422d-8c16-4ec9630a142c" />
<img width="2485" height="1401" alt="Screenshot 2026-03-05 102454" src="https://github.com/user-attachments/assets/96d33f61-2982-4564-9cf2-289cc7854f14" />
<img width="2483" height="1404" alt="Screenshot 2026-03-05 101043" src="https://github.com/user-attachments/assets/c98e3cd1-d78e-4095-9ab6-45bdd950c3e9" />
<img width="2483" height="1403" alt="Screenshot 2026-03-05 101718" src="https://github.com/user-attachments/assets/d7da22a3-9bff-4d06-b4e3-b21b011e0735" />

### Demo - V1
https://github.com/user-attachments/assets/494742fb-2dac-47fe-ac4c-c8c1a3e98897




### Runtime Responsibilities

#### SDK Layer

The SDK is responsible for:

- creating runs and spans
- tracking parent-child relationships
- recording artifacts for prompts, outputs, and media
- buffering canonical events in memory
- writing telemetry locally
- optionally exporting telemetry to the server

This logic lives mainly in:

- `oat/tracer.py`
- `oat/models.py`
- `oat/storage.py`
- `oat/media.py`
- `oat/integrations/`

#### Local Storage Layer

The storage layer materializes the event stream into queryable tables and blob content.

Current local storage shape:

- SQLite database: `.oat/telemetry.db`
- blob directory: `.oat/blobs/`

The SDK process and the server process both use the same `StorageEngine` design, but each process writes to its own `.oat/` directory unless they intentionally share a working directory or data path.

#### Server Layer

The FastAPI server is the central collector and query surface for the dashboard.

It is responsible for:

- receiving event batches from clients
- materializing runs, spans, and artifacts
- serving run lists, run detail, graphs, analytics, and prompts
- serving artifact content
- exposing the model arena APIs
- broadcasting websocket updates

#### UI Layer

The React UI is a read-heavy client over the server API.

It is responsible for:

- browsing services and runs
- rendering run DAG and waterfall views
- showing token usage and artifacts
- showing aggregate analytics and prompt analytics
- comparing models side-by-side in the arena

### Write Path

The current write path is:

1. User code calls the tracer API directly or through decorators / patched provider SDKs.
2. `AgentTracer` creates canonical lifecycle and artifact events.
3. The tracer queues those events in memory.
4. A background worker flushes the events into local SQLite and blob storage.
5. If `export_url` is configured, the same event batch is POSTed to `POST /v1/ingest/events`.
6. The server materializes those events into its own run/span/artifact tables.
7. The dashboard can then read the server-backed data.

This is why a local trace is not automatically visible in the dashboard unless it is exported to the server.

### Read Path

The current read path is:

1. The dashboard checks server health at `GET /health`.
2. The Runs page loads services from `GET /v1/services`.
3. The user selects a service, and the UI loads runs from `GET /v1/runs`.
4. The Run Detail page loads:
   - `GET /v1/runs/{run_id}`
   - `GET /v1/runs/{run_id}/graph`
   - `GET /v1/runs/{run_id}/timeline`
5. Analytics pages load aggregate summaries from:
   - `GET /v1/analytics/overview`
   - `GET /v1/analytics/prompts`
6. Artifact payloads can be fetched from `GET /v1/artifacts/{artifact_id}/content`.

### Integration Architecture

Built-in provider integrations follow a patch-and-wrap model.

Current flow:

1. `patch_openai()`, `patch_anthropic()`, `patch_google()`, or `patch_ollama()` is called.
2. The integration wraps the provider SDK method.
3. The wrapper creates an LLM or embedding span.
4. Input payloads are normalized into artifacts.
5. Usage and cost are normalized into a common shape.
6. The span is finished and emitted through the tracer.

In-tree integrations currently exist for:

- OpenAI
- Anthropic
- Google Gemini
- Ollama

### Arena Architecture

The arena is independent from SDK tracing. It is a server-driven model comparison feature.

Current flow:

1. The UI requests available models from `GET /arena/models`.
2. The server reads `providers.yaml` and any discovered models.
3. The user selects 2 to 4 models and submits a shared prompt.
4. The UI sends parallel `POST /arena/run` requests.
5. The server executes provider-specific calls and returns normalized output, usage, pricing, and latency.
6. The UI renders each model result in a separate comparison panel.

### Deployment Shapes

The repository currently supports three practical deployment shapes:

#### Local SDK Only

- user code writes to local `.oat/`
- no server required
- no dashboard visibility unless you inspect the local data directly

#### Local SDK Plus Server and Dashboard

- user code writes locally and exports to the server
- server writes to its own `.oat/`
- dashboard reads from the server

This is the main end-to-end development flow in the current repo.

#### Central Server for Multiple Clients

- multiple client processes export to one OAT server
- the server acts as the single read surface for the dashboard
- each client may still keep its own local `.oat/` store for safety and debugging

### Current Architectural Boundaries

Important current boundaries to keep in mind:

- the SDK is event-based internally
- the dashboard is server-backed, not client-storage-backed
- storage is SQLite-only in the current implementation
- `providers.yaml` drives the arena catalog, not the tracer itself
- `tracer.yaml` is illustrative and is not auto-loaded by the SDK
- the server is designed for trusted/local environments by default, not hardened multi-tenant deployments

## Core Data Model

The canonical model lives in `oat/models.py`.

### Run

A `Run` is the top-level execution visible in the dashboard.

Key fields:

- `run_id`
- `service_name`
- `service_version`
- `environment`
- `session_id`
- `workflow_id`
- `root_span_id`
- `status`
- `started_at`, `ended_at`, `duration_ms`
- `input_summary`, `output_summary`, `error_message`
- `metadata`
- aggregate counters: `total_tokens`, `total_cost`, `llm_calls`, `tool_calls`, `span_count`

### Span

A `Span` is an execution unit inside a run.

Key fields:

- `span_id`
- `run_id`
- `parent_span_id`
- `trace_path`
- `kind`
- `name`
- `status`
- `started_at`, `ended_at`, `duration_ms`
- `model`, `provider`, `operation`
- `input_summary`, `output_summary`
- `error_type`, `error_message`
- `attributes`
- `usage`
- `cost`
- `score`, `feedback`

The repo still keeps compatibility aliases so older code can use `trace_id` and `span_type`. In current code, `trace_id` maps directly to `run_id`, and `SpanType` is an alias of `SpanKind`.

### Artifact

An `Artifact` is the payload layer. This is where OAT stores prompts, model outputs, images, audio, previews, and metadata.

Key fields:

- `artifact_id`
- `run_id`
- `span_id`
- `role`
- `content_type`
- `storage_uri`
- `inline_text`
- `preview`
- `metadata`
- `size_bytes`
- `sha256`
- `created_at`

### Event Envelope

All canonical ingestion/export is event-based via `EventEnvelope`.

Current canonical event types are:

- `run.started`
- `run.finished`
- `span.started`
- `span.finished`
- `artifact.created`

### Supported Span Kinds

Current stable `SpanKind` values are:

- `agent`
- `llm`
- `tool`
- `retrieval`
- `memory`
- `guardrail`
- `http`
- `database`
- `file`
- `embedding`
- `rerank`
- `chain`
- `cache`
- `handoff`
- `custom`

### Supported Status Values

Current `SpanStatus` values are:

- `running`
- `success`
- `error`
- `cancelled`
- `timeout`
- `skipped`

## SDK and Tracer Internals

The main tracer implementation is `oat/tracer.py`.

### AgentTracer

`AgentTracer` is the main runtime object.

Constructor parameters:

- `service_name`
- `data_dir`
- `export_url`
- `auto_flush`
- `flush_interval`
- `strict_run_lifecycle`

Current behavior:

- Local persistence is always attempted through `StorageEngine`.
- If `export_url` is set, the tracer also POSTs event batches to `POST /v1/ingest/events`.
- Export failures never fail application logic.
- Telemetry is buffered in a bounded in-process queue.

Current queue and batching details:

- queue max size: `10_000`
- batch max size: `20`
- batch max age: `0.05s`

If the queue is full, events are dropped by design. The count is tracked as `AgentTracer.dropped_events`.

### Global Tracer Behavior

`get_tracer(**kwargs)` returns a process-global singleton.

Important current caveat:

- the first call with arguments creates the singleton
- later calls with different kwargs do not reconfigure it
- later kwargs are ignored and a warning is emitted

If you need separate tracer configurations in one process, instantiate `AgentTracer(...)` directly.

### Explicit Lifecycle API

The canonical API is explicit:

```python
from oat import SpanKind, SpanStatus, get_tracer

tracer = get_tracer(
    service_name="my-agent",
    export_url="http://localhost:8787",
)

run = tracer.start_run(input_summary="Summarize this document")
span = tracer.start_span(
    name="agent.summarize",
    kind=SpanKind.AGENT,
    run_id=run.run_id,
)

tracer.record_artifact(
    span,
    role="input.text",
    content_type="text/plain",
    content="Document text goes here",
    inline_text="Document text goes here",
)

tracer.finish_span(span, SpanStatus.SUCCESS)
tracer.finish_run(
    status=SpanStatus.SUCCESS,
    output_summary="Summary completed",
    run_id=run.run_id,
)
```

### Decorators and Context Helpers

The tracer also exposes convenience APIs:

- `@trace(...)`
- `trace_llm(...)`
- `trace_tool(...)`
- `trace_retrieval(...)`
- `span(...)` context manager

Current behavior of the decorator layer:

- it creates an implicit run if no run is active
- it records input and output artifacts by default
- async and sync functions are both supported
- timeouts and exceptions are mapped to span/run status

If `strict_run_lifecycle=True`, starting a span without an active run raises instead of creating an implicit run.

### Distributed Context Propagation

The tracer supports lightweight propagation across service boundaries.

Current outbound/inbound header names:

- `x-oat-run-id`
- `x-oat-span-id`
- `x-oat-parent-span-id`
- `x-oat-workflow-id`

Helpers:

- `inject_context(headers)`
- `extract_context(headers)`

### Subprocess Propagation

Helpers:

- `get_subprocess_env(...)`
- `restore_from_env()`

Environment keys used for child processes:

- `OAT_PROPAGATED_RUN_ID`
- `OAT_PROPAGATED_SPAN_ID`

### Agent Loop Tracking

`AgentLoop` and `agent_loop(...)` provide a first-class way to model iterative agent loops.

Current behavior:

- a parent span is created for the loop
- each iteration becomes a child span
- each step is tagged with `iteration_index`
- `finish()` must be called to close the loop span

## Local Storage Model

Storage is implemented in `oat/storage.py`.

### Current Storage Backend

The project currently uses SQLite, not DuckDB.

Default local layout:

- database: `.oat/telemetry.db`
- blob directory: `.oat/blobs/`

SQLite is initialized with:

- `PRAGMA journal_mode=WAL`
- `PRAGMA synchronous=NORMAL`

### Current Tables

- `runs`
- `spans`
- `artifacts`
- `feedback`
- `event_log`

### What Gets Stored

- `event_log` stores the raw canonical event stream.
- `runs` and `spans` are materialized rollups for read APIs and the dashboard.
- `artifacts` stores preview metadata and references to blob content.
- `feedback` stores thumbs-up and thumbs-down scoring.

### Artifact Materialization

Current artifact behavior:

- If artifact `content` is present and no `storage_uri` or `inline_text` is supplied, content is written to `.oat/blobs/` and referenced by `blob://sha256/...`.
- Short string content may also be copied into `inline_text`.
- `preview` is stored separately for fast dashboard rendering.
- Artifact content can later be fetched through the artifact content endpoint.

### Run Rollups

Run counters are derived from finished span data.

Rollups include:

- `span_count`
- `llm_calls`
- `tool_calls`
- `total_tokens`
- `total_cost`
- final run status and duration

A full run refresh is done when a `run.finished` event is processed.

## Local-First vs Server-Visible Data

This is one of the most important operational details in the repo.

### SDK Process Storage

If you create a tracer in a client process, it writes into that process's local `.oat/` directory unless you pass a different `data_dir`.

### Server Storage

The FastAPI server also writes into its own `.oat/` directory, relative to the server process working directory.

Examples:

- If you run `cd server && uvicorn main:app --port 8787 --host 0.0.0.0 --reload`, server data lands in `server/.oat/`.
- If you run a client example from repo root without export, its local data lands in the repo root `.oat/`.

### Dashboard Visibility Rule

The dashboard queries the server API, so it only sees traces stored by the server.

That means:

- local-only SDK traces are not automatically visible in the dashboard
- to see a client run in the dashboard, the client must export to the server

## Exporters

The repo includes exporter classes in `oat/exporters.py`.

Current exporters:

- `ConsoleExporter`
- `HTTPExporter`
- `FileExporter`
- `CompositeExporter`

Important current detail:

- `AgentTracer` exports canonical events to `POST /v1/ingest/events`
- `HTTPExporter` is an older compatibility path and posts legacy span payloads to `/ingest`

## Auto-Instrumentation and Integrations

Built-in integrations live in `oat/integrations/`.

Current in-tree patchers:

- OpenAI
- Anthropic
- Google Gemini
- Ollama

Entry points:

- `patch_openai()` / `unpatch_openai()`
- `patch_anthropic()` / `unpatch_anthropic()`
- `patch_google()` / `unpatch_google()`
- `patch_ollama()` / `unpatch_ollama()`
- `patch_all()` / `unpatch_all()`

`patch_all()` patches the built-in integrations above and then applies any integrations registered through the extension registry in `oat/integrations/base.py`.

### Important Note About Extras vs Built-In Patchers

`pyproject.toml` still declares optional extras for `litellm` and `langchain`, but this repository does not currently ship first-party in-tree patcher modules for those libraries.

That means:

- the extras can install those packages
- extension-based instrumentation is possible
- but built-in patch functions currently exist only for OpenAI, Anthropic, Google, and Ollama

### OpenAI Integration

The OpenAI integration is currently the most developed path in the repo.

Current behavior includes:

- sync and async chat completion tracing
- embeddings tracing
- streaming wrapper support
- provider inference from `base_url` and model naming
- multimodal artifact extraction from message payloads
- normalized token and cost recording on spans

Current provider inference covers common OpenAI-compatible endpoints such as:

- OpenAI
- OpenRouter
- Groq
- DeepSeek
- Qwen
- Moonshot
- Kimi
- Mistral
- Ollama
- self-hosted OpenAI-compatible endpoints
- vLLM-style endpoints

### Multimodal Capture

`oat/media.py` extracts artifacts from:

- OpenAI-style multimodal messages
- Anthropic-style multimodal messages
- Gemini-style contents and parts

Current roles used by the system include:

- `input.text`
- `input.message`
- `input.image`
- `input.audio`
- `derived.audio_transcript`
- `derived.media_analysis`
- `output.text`
- `output.embedding`

This is what enables the dashboard to show prompt payloads, output text, local images, remote image references, and media-related metadata.

## Provider Normalization and Pricing

### Provider Detection

Provider normalization lives in `oat/providers.py`.

Current detection inputs:

- explicit provider hint
- base URL
- model naming pattern

Known normalized provider names currently include:

- `openai`
- `anthropic`
- `google`
- `ollama`
- `mistral`
- `moonshot`
- `kimi`
- `qwen`
- `openrouter`
- `groq`
- `deepseek`
- `self_hosted`
- `vllm`
- `openai_compatible`

### Pricing

Pricing helpers live in `oat/pricing.py`.

Current behavior:

- built-in per-1K token pricing table for common model families
- family/prefix fallback matching for newer variant names
- alias normalization for versioned model names
- normalized usage payload generation via `build_llm_usage(...)`
- `pricing_status` flag set to `known` or `unknown`

### Pricing Overrides

Current override resolution:

1. `OAT_PRICING_FILE`, if set
2. `pricing_overrides.yaml`
3. `pricing_overrides.yml`
4. `pricing_overrides.json`

The override file can be either a top-level pricing map or a document with a `pricing:` key.

## Backend Server

The backend implementation is `server/main.py`.

### Current Responsibilities

The server currently does all of the following:

- ingests canonical event batches
- supports legacy span ingestion
- persists data with the same `StorageEngine`
- serves run lists, run detail, graphs, artifacts, analytics, and prompt analytics
- exposes a model arena API
- exposes websocket endpoints for realtime clients
- performs housekeeping for stale running spans and runs

### Environment Loading

The server loads `.env` and `.env.local` from the current working directory only.

This matters in practice:

- if you run the server from `server/`, it reads `server/.env`
- if you run it from the repo root, it reads root `.env`
- it does not automatically search multiple directories the way the multimodal example does

### Data Directory

The server uses:

- `DATA_DIR = Path(".oat")`

So the actual on-disk location depends on the working directory of the server process.

### Housekeeping

A background task marks stale running spans and runs as cancelled so the dashboard does not show orphaned `running` entries forever after a client crash.

Current config:

- `OAT_STALE_SPAN_TIMEOUT`, default `600` seconds
- housekeeping check interval: `60` seconds

### Provider Catalog

The model arena reads provider configuration from:

- environment variable `OAT_PROVIDER_CONFIG`, or
- default file `providers.yaml`

Provider discovery cache TTL is controlled by:

- `OAT_PROVIDER_DISCOVERY_TTL`, default `120` seconds

### CORS

Current server behavior:

- if `OAT_CORS_ORIGINS` is not set, allowed origins default to `*`
- credentials are only enabled when you provide explicit origins

This is practical for local development, but the server is not production-hardened by default.

## Canonical API Surface

Current canonical endpoints are:

### Ingest

- `POST /v1/ingest/events`
  - canonical batched event ingest
- `POST /v1/ingest/run`
  - direct ingest of a run plus spans and artifacts

### Runs and Detail

- `GET /v1/runs`
- `GET /v1/runs/{run_id}`
- `GET /v1/runs/{run_id}/timeline`
- `GET /v1/runs/{run_id}/graph`
- `GET /v1/runs/{run_id}/artifacts`

### Artifacts and Feedback

- `GET /v1/artifacts/{artifact_id}/content`
- `POST /v1/spans/{span_id}/feedback`

### Analytics

- `GET /v1/analytics/overview`
- `GET /v1/analytics/prompts`
- `GET /v1/services`

### Realtime

- `WS /v1/ws`

## Legacy Compatibility API Surface

The server still ships legacy routes for older clients and older UI assumptions.

Current legacy endpoints are:

- `POST /ingest`
- `GET /traces`
- `GET /traces/{trace_id}`
- `GET /spans/{span_id}`
- `DELETE /traces/{trace_id}`
- `POST /spans/{span_id}/feedback`
- `POST /search`
- `GET /analytics/overview`
- `GET /analytics/latency`
- `WS /ws`

## Arena API

The model comparison feature uses dedicated endpoints:

- `GET /arena/models`
- `POST /arena/run`

Current arena responses include fields such as:

- `status`
- `provider`
- `model`
- `output`
- `usage`
- `pricing`
- `duration_ms`

## Health and Admin API

Current utility endpoints:

- `GET /health`
- `DELETE /admin/reset`

`/health` returns server status, version, and resolved data directory.

`/admin/reset` deletes all stored runs. It is intended for trusted local environments, not for an exposed production deployment.

## Dashboard UI

The dashboard lives in `ui/` and is built with React 18 and Vite 5.

Current UI dependencies include:

- `react`
- `react-router-dom`
- `recharts`
- `lucide-react`
- `dagre`
- `reactflow`
- `date-fns`
- `clsx`

Current API base URL behavior from `ui/src/App.jsx`:

- in development: `http://localhost:8787`
- outside development builds: `/api`

That means a production-style deployment would need a reverse proxy or equivalent routing in front of the backend.

### Route Map

Current routes from `ui/src/App.jsx`:

- `/` -> Runs page
- `/runs/:runId` -> Run detail page
- `/analytics` -> Analytics page
- `/prompts` -> Prompt Registry page
- `/flow` -> Flow Graph page
- `/arena` -> Model Arena page

### Navigation Model

Current sidebar sections:

- Monitor
  - Runs
  - Analytics
- Intelligence
  - Prompt Registry
  - Flow Graph
  - Model Arena

### Runs Page

The Runs page is service-first, not run-first.

Current flow:

1. fetch services from `GET /v1/services`
2. let the user choose a service or agent name
3. fetch runs for that service from `GET /v1/runs?limit=100&service_name=...`

### Run Detail Page

The run detail page currently loads three data sources in parallel:

- `GET /v1/runs/{run_id}`
- `GET /v1/runs/{run_id}/graph`
- `GET /v1/runs/{run_id}/timeline`

Current capabilities:

- DAG view with React Flow
- waterfall timeline view
- span inspector with `overview`, `input`, and `output` tabs
- token usage display
- model and cost display
- artifact rendering for text and images
- feedback buttons for span scoring

### Analytics Page

The analytics page currently supports:

- time windows: `1H`, `24H`, `7D`
- optional service filter
- KPI cards for cost, run volume, latency, and error rate
- traffic and error chart
- token usage chart
- model performance table
- operation-type table

### Prompt Registry Page

The Prompt Registry page is built from `GET /v1/analytics/prompts`.

Current columns and behaviors:

- prompt text
- call count
- error rate
- last used timestamp
- average latency
- average cost
- copy button

Prompts are derived from artifact roles `input.message` and `input.text`.

### Flow Graph Page

The Flow Graph page is a run-level graph viewer.

Current workflow:

1. choose a service
2. choose a run under that service
3. fetch `GET /v1/runs/{run_id}/graph`
4. render a dagre-based flow chart

### Model Arena Page

The Arena page lets the user compare between 2 and 4 models side by side on the same system prompt and user prompt.

Current behavior:

- fetch available models from `GET /arena/models`
- render provider and model availability
- call `POST /arena/run` for each selected panel
- show output text, latency, token usage, cost, and pricing status

## Provider Catalog and Arena Configuration

The provider catalog is defined in `providers.yaml`.

Current provider entries include:

- `openai`
- `anthropic`
- `google`
- `openrouter`
- `groq`
- `deepseek`
- `mistral`
- `moonshot`
- `kimi`
- `qwen`
- `ollama`
- `self_hosted`

Current provider types used by the arena:

- `openai`
- `anthropic`
- `google`
- `openai_compatible`
- `ollama`

### Static vs Discoverable Providers

Current catalog behavior:

- some providers are fully static
- some providers also support model discovery

Discovery is currently enabled in the checked-in catalog for:

- `openrouter`
- `ollama`
- `self_hosted`

### Common Arena Environment Variables

The provider catalog resolves API keys and base URLs from environment variables.

Examples from the current catalog:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `DEEPSEEK_API_KEY`
- `MISTRAL_API_KEY`
- `MOONSHOT_API_KEY`
- `KIMI_API_KEY`
- `QWEN_API_KEY`
- `OLLAMA_BASE_URL`
- `SELF_HOSTED_LLM_BASE_URL`
- `OPENAI_COMPAT_BASE_URL`
- `VLLM_BASE_URL`
- `SELF_HOSTED_LLM_API_KEY`
- `OPENAI_COMPAT_API_KEY`

## Examples

### `examples/demo_agent.py`

This is a simulated agent example that demonstrates:

- decorator-based tracing
- nested spans
- retrieval, tool, and guardrail style spans
- successful and failing runs
- export to `http://localhost:8787`

Current notes:

- it is demo-oriented and not especially polished
- it still uses legacy names such as `SpanType`
- it prints simulated output and token usage rather than making real model calls

### `examples/multimodal_image_agent.py`

This is the main real multimodal example in the repo right now.

Current behavior:

- patches the OpenAI integration before importing `AsyncOpenAI`
- loads environment variables from multiple locations
- defaults to a local image file when available
- exports traces to the server by default
- sends a multimodal request with text plus image
- records explicit input and output artifacts

Current default behavior if you run it with no flags:

- image path defaults to `examples/test_image.png` if that file exists
- export URL defaults to `http://localhost:8787`
- model defaults to `gpt-4o-mini` unless overridden

Current env file search order in the example:

1. `server/.env`
2. `server/.env.local`
3. repo root `.env`
4. repo root `.env.local`
5. `examples/.env`
6. `examples/.env.local`
7. current working directory `.env`
8. current working directory `.env.local`

Current CLI options:

- `--image-url`
- `--image-path`
- `--use-remote-default`
- `--question`
- `--model`
- `--max-tokens`

## Installation

### Prerequisites

- Python `3.9+`
- `pip`
- Node.js and `npm` for the dashboard

`pyproject.toml` is the authoritative source for Python extras and package metadata. `requirements.txt` is present as a convenience list, but the install commands in this document are aligned to `pyproject.toml`.

### Core SDK Only

For local-only tracing with no server and no provider integrations:

```bash
pip install -e .
```

### Server

To run the backend server:

```bash
pip install -e ".[server]"
```

### HTTP Export

If you want the SDK to export over HTTP, install the HTTP extra:

```bash
pip install -e ".[http]"
```

### OpenAI Integration

For the multimodal example or OpenAI auto-instrumentation:

```bash
pip install -e ".[openai,http]"
```

### Everything Declared in `pyproject.toml`

```bash
pip install -e ".[all]"
```

Note that `.[all]` installs declared extras, but the repository's built-in patchers are currently limited to OpenAI, Anthropic, Google, and Ollama.

## Running the Full Local Stack

### Backend

From the repository root, one reliable path is:

```bash
cd server
uvicorn main:app --port 8787 --host 0.0.0.0 --reload
```

### Frontend

In another shell:

```bash
cd ui
npm install
npm run dev
```

The dashboard expects the API at `http://localhost:8787` in development.

### One-Command Scripts

The repository includes convenience scripts:

- `start.sh`
- `start.ps1`

Current behavior of the scripts:

- install Python dependencies with `pip install -e ".[server]"`
- install UI dependencies with `npm install`
- start backend on port `8787`
- start frontend on port `3000`

### CLI Entry Point

After installing the package, there is also a Python entry point:

```bash
oat-server
```

Current caveat:

- `oat-server` runs `server.main:main()` from the current working directory
- because server env loading is cwd-relative, where you run it from affects which `.env` file is read and where `.oat/` is created

## Running the Examples

### Demo Agent

```bash
python examples/demo_agent.py
```

### Multimodal Image Agent

```bash
python examples/multimodal_image_agent.py
```

Useful variants:

```bash
python examples/multimodal_image_agent.py --image-path examples/test_image.png
python examples/multimodal_image_agent.py --question "Describe the image in detail."
python examples/multimodal_image_agent.py --model gpt-4o
```

## Configuration Reference

### Common Environment Variables

| Variable | Used By | Current Meaning |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI integration, multimodal example, arena | API key for OpenAI or compatible endpoint |
| `OPENAI_BASE_URL` | OpenAI integration, multimodal example | Base URL for OpenAI-compatible providers |
| `OAT_EXPORT_URL` | SDK callers and examples | Server base URL for event export |
| `OAT_PROVIDER_CONFIG` | Server | Path to provider catalog YAML |
| `OAT_PROVIDER_DISCOVERY_TTL` | Server | Discovery cache TTL in seconds |
| `OAT_PRICING_FILE` | Pricing helpers | Path to pricing override file |
| `OAT_STALE_SPAN_TIMEOUT` | Server | Age threshold before stale running spans are cancelled |
| `OAT_CORS_ORIGINS` | Server | Comma-separated allowed origins |
| `OAT_MULTIMODAL_MODEL` | Multimodal example | Default vision-capable model name |
| `OAT_MULTIMODAL_IMAGE_URL` | Multimodal example | Optional default remote image |
| `OAT_MULTIMODAL_IMAGE_PATH` | Multimodal example | Optional default local image path |
| `OLLAMA_BASE_URL` | Arena and Ollama integration | Base URL for local or remote Ollama |

### `tracer.yaml`

A `tracer.yaml` file exists in the repo, but it is currently illustrative configuration, not a file that the core SDK automatically loads on startup.

Treat it as an example reference, not as the current runtime control plane.

## Development State and Caveats

### 1. Trusted-Environment Assumption

The server currently has no built-in authentication layer for ingest, reads, arena, or admin reset operations.

That is acceptable for local development and internal experimentation, but it is not a production security model.

### 2. Local Storage Separation Can Be Confusing

Because both the SDK and the server write to `.oat/` relative to their own working directories, it is easy to end up with multiple SQLite stores.

If you are debugging why a run does not appear in the dashboard, check:

- which process created the run
- whether that process exported to the server
- where the server was launched from
- which `.oat/telemetry.db` the dashboard backend is reading

### 3. Older Docs Lag the Code

Some older docs and comments in the repository still refer to earlier architecture ideas such as DuckDB-centric analytics, older integrations, or files that no longer exist.

For the current repository state, prefer:

- source code
- this `README_DETAILED.md`
- `README.md` for the short overview

### 4. Tests

The `tests/` directory is currently empty. There is project configuration for `pytest`, `pytest-asyncio`, coverage, Ruff, MyPy, and Black, but there is not yet a committed automated test suite that exercises the main flows.

### 5. Version Strings Are Not Yet Unified

At the time of writing:

- the Python package exports version `0.1.0`
- the FastAPI app reports version `0.2.0`

This does not break functionality, but it is worth knowing when comparing logs, docs, and API responses.

## Practical Source-of-Truth Files

If you need to confirm current behavior, these are the most useful files to read:

- `oat/models.py`
- `oat/tracer.py`
- `oat/storage.py`
- `oat/media.py`
- `oat/pricing.py`
- `oat/providers.py`
- `oat/integrations/openai_integration.py`
- `server/main.py`
- `providers.yaml`
- `ui/src/App.jsx`
- `ui/src/pages/TracesPage.jsx`
- `ui/src/pages/TraceDetailPage.jsx`
- `ui/src/pages/AnalyticsPage.jsx`
- `ui/src/pages/PromptsPage.jsx`
- `ui/src/pages/FlowPage.jsx`
- `ui/src/pages/ArenaPage.jsx`

## Summary

OpenAgentTrace, as currently implemented in this repository, is a canonical run/span/artifact observability stack with:

- explicit telemetry lifecycle APIs
- local-first SQLite storage
- optional HTTP export
- multimodal artifact capture
- provider normalization and pricing helpers
- a FastAPI backend for ingestion and analytics
- a React dashboard for run inspection and model comparison

The most important operational rule is still this: if you want a run to appear in the dashboard, the process that created the run must export it to the server that the dashboard is querying.


## Acknowledgments

OpenAgentTrace is inspired by and builds upon ideas from:

- **OpenTelemetry**: Distributed tracing standards
- **LangSmith**: Agent-specific observability patterns
- **Arize Phoenix**: ML observability concepts
- **Grafana/Prometheus**: Time-series monitoring best practices

Special thanks to the open-source community for their tools and frameworks.

---

## License

OpenAgentTrace is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 OpenAgentTrace Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Support

- **Website**: (https://oat.thelearnchain.com)
- **Email**: founder@thelearnchain.com

---

