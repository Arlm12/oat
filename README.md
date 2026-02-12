# OpenAgentTrace (OAT)

<div align="center">

**🔍 The Open Standard for AI Agent Observability**


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.0--beta-orange.svg)](https://github.com/yourusername/openagent-trace)


</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Why OpenAgentTrace?](#-why-openagent-trace)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Usage Guide](#-usage-guide)
  - [Tracing with Decorators](#tracing-with-decorators)
  - [Manual Span Creation](#manual-span-creation)
  - [Auto-Instrumentation](#auto-instrumentation)
- [Span Types](#-span-types)
- [Configuration](#-configuration)
- [Dashboard](#-dashboard)
- [Server API](#-server-api)
- [Integrations](#-integrations)
- [Advanced Topics](#-advanced-topics)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Roadmap](#-roadmap)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🌟 Overview

**OpenAgentTrace (OAT)** is an open-source, vendor-neutral observability platform purpose-built for AI agents. Unlike traditional observability tools that focus on request-response patterns, OAT captures the complete decision-making flow of autonomous agents including LLM calls, tool executions, reasoning chains, and multi-step workflows.

### What Makes OAT Different?

Traditional observability tools (DataDog, New Relic, Grafana) are built for microservices and APIs. AI agents require fundamentally different instrumentation:

| Traditional Observability | OpenAgentTrace |
|--------------------------|----------------|
| Request → Response | Agent → Plan → Tool → LLM → Memory → Decision |
| Fixed endpoints | Dynamic reasoning chains |
| Simple traces | Complex DAGs with multiple paths |
| HTTP metrics | Token usage, cost attribution, reasoning depth |
| Error tracking | Guardrail triggers, hallucination detection |

---

## 💡 Why OpenAgentTrace?

### The Problem

Building production AI agents is hard. Debugging them is harder. Current tools fall short:

- **❌ LangSmith**: Vendor lock-in, expensive at scale, closed-source
- **❌ Weights & Biases**: ML-focused, not agent-native
- **❌ Arize Phoenix**: Lacks real-time monitoring, limited agent semantics
- **❌ OpenTelemetry**: Too generic, requires heavy customization

### The Solution

OpenAgentTrace provides:

✅ **Agent-Native Semantics** - Span types designed for AI: LLM, Tool, Retrieval, Guardrail, Memory, Chain
✅ **Local-First Architecture** - SQLite storage with optional remote export, no vendor lock-in
✅ **Zero-Config Auto-Instrumentation** - Patch OpenAI/Anthropic once, trace everything automatically
✅ **Cost Attribution** - Track token usage and costs per operation, model, and agent
✅ **Real-Time Dashboard** - React UI with DAG visualization, waterfall charts, and live updates
✅ **Production Ready** - Non-blocking async export, graceful shutdown, thread-safe storage

---

## 🚀 Key Features

### Core Capabilities

1. **🎯 Semantic Span Types (14 Types)**
   - `agent`, `llm`, `tool`, `retrieval`, `guardrail`, `handoff`
   - `embedding`, `rerank`, `memory`, `chain`
   - `http`, `database`, `cache`, `file_io`

2. **💰 Automatic Cost Tracking**
   - Built-in pricing for 30+ models (OpenAI, Anthropic, Google, Mistral)
   - Per-call cost calculation (prompt + completion)
   - Aggregate cost reporting by trace, agent, or time period

3. **🖼️ Multimodal Support**
   - Image, audio, video metadata extraction
   - Vision model token estimation
   - Media content hashing and deduplication

4. **📊 Local-First Storage**
   - SQLite with WAL mode for concurrency
   - Blob storage with content-based deduplication
   - Optional HTTP export to remote servers

5. **🔄 Real-Time Updates**
   - WebSocket support for live dashboard updates
   - Non-blocking background workers
   - Graceful shutdown with flush guarantees

6. **📈 Advanced Analytics**
   - DuckDB-powered time-series queries
   - Latency percentiles by span type
   - Error rate tracking and alerting
   - Token usage trends

7. **🎨 Beautiful Dashboard**
   - DAG visualization with ReactFlow
   - Waterfall/timeline views
   - Interactive span explorer
   - Cost and performance metrics

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Your AI Agent                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  @trace      │  │  patch_openai│  │  span()      │      │
│  │  decorator   │  │  integration │  │  context mgr │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └─────────┬────────┴─────────┬────────┘
                    ▼                  ▼
          ┌─────────────────────────────────────┐
          │      OpenAgentTrace SDK (oat/)      │
          │                                     │
          │  ┌──────────┐    ┌──────────┐      │
          │  │ Tracer   │───▶│ Storage  │      │
          │  └──────────┘    └──────────┘      │
          │       │               │             │
          │       │          SQLite + Blobs     │
          │       │                             │
          │  ┌────▼─────┐                       │
          │  │ Exporter │                       │
          │  └──────────┘                       │
          └──────┬──────────────────────────────┘
                 │ HTTP POST /ingest
                 ▼
          ┌─────────────────────────────────────┐
          │    FastAPI Server (server/)         │
          │                                     │
          │  ┌──────────┐    ┌──────────┐      │
          │  │ REST API │    │ WebSocket│      │
          │  └──────────┘    └──────────┘      │
          │       │               │             │
          │  ┌────▼────┐    ┌─────▼─────┐      │
          │  │ SQLite  │    │  DuckDB   │      │
          │  │ Storage │    │ Analytics │      │
          │  └─────────┘    └───────────┘      │
          └──────┬──────────────────────────────┘
                 │ REST API
                 ▼
          ┌─────────────────────────────────────┐
          │      React Dashboard (ui/)          │
          │                                     │
          │  ┌──────────┐    ┌──────────┐      │
          │  │ Traces   │    │ Analytics│      │
          │  │ Explorer │    │ Charts   │      │
          │  └──────────┘    └──────────┘      │
          │                                     │
          │  ┌──────────┐    ┌──────────┐      │
          │  │ DAG View │    │ Prompts  │      │
          │  │ Waterfall│    │ Registry │      │
          │  └──────────┘    └──────────┘      │
          └─────────────────────────────────────┘
```

### Component Architecture

```
OpenAgentTrace/
├── oat/                          # Python SDK (Core Library)
│   ├── __init__.py              # Public API
│   ├── tracer.py                # Core tracing (decorators, context)
│   ├── models.py                # Data models (Span, Trace)
│   ├── storage.py               # SQLite + blob storage
│   ├── exporters.py             # HTTP/Console exporters
│   ├── pricing.py               # Cost calculation
│   ├── media.py                 # Multimodal analysis
│   └── integrations/            # Auto-instrumentation
│       ├── openai_integration.py
│       └── anthropic_integration.py
│
├── server/                       # FastAPI Backend
│   └── main.py                  # API endpoints, analytics
│
├── ui/                          # React Dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   └── pages/
│   │       ├── TracesPage.jsx
│   │       ├── TraceDetailPage.jsx
│   │       ├── AnalyticsPage.jsx
│   │       └── PromptsPage.jsx
│   └── package.json
│
├── pyproject.toml              # Package config
├── requirements.txt            # Dependencies
└── tracer.yaml                 # Configuration
```

---

## ⚡ Quick Start

### 1. Instrument Your Agent (30 seconds)

```python
from oat import trace, get_tracer
from oat.integrations import patch_openai

# Initialize tracer
tracer = get_tracer(
    service_name="my-agent",
    export_url="http://localhost:8787"  # Optional remote server
)

# Auto-patch OpenAI (one line!)
patch_openai()

# Decorate your agent function
@trace(name="my_agent", span_type="agent")
async def my_agent(query: str):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    # This call is automatically traced!
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}]
    )

    return response.choices[0].message.content

# Run your agent
import asyncio
result = asyncio.run(my_agent("What is Python?"))
print(result)
```

**That's it!** Every LLM call is now traced with:
- ✅ Token usage and cost
- ✅ Latency tracking
- ✅ Parent-child relationships
- ✅ Input/output capture

### 2. Start the Server (Optional but Recommended)

```bash
# Terminal 1: Start FastAPI server
uvicorn server.main:app --port 8787

# Server starts at http://localhost:8787
# API docs at http://localhost:8787/docs
```

### 3. Launch the Dashboard

```bash
# Terminal 2: Start React dashboard
cd ui
npm run dev

# Dashboard opens at http://localhost:3000
```

### 4. View Your Traces

1. Open browser to `http://localhost:3000`
2. Click on your trace to see:
   - 📊 **DAG visualization** of agent execution flow
   - ⏱️ **Waterfall chart** showing timing
   - 💰 **Cost breakdown** by operation
   - 📝 **Input/output inspection**

---

## 🧠 Core Concepts

### Trace Hierarchy

```
Trace (One per agent execution)
└── Agent Span (Root span, no parent)
    ├── LLM Span (GPT-4 call)
    │   └── Cost: $0.0042
    ├── Tool Span (Database query)
    │   └── Database Span (Child of tool)
    ├── Retrieval Span (Vector search)
    │   └── Rerank Span (Re-ranking results)
    └── LLM Span (Final response)
        └── Cost: $0.0031
```

### Key Entities

#### Trace
A **trace** represents one complete agent execution from start to finish. It contains multiple spans.

```python
{
    "trace_id": "abc-123-def",
    "start_time": 1699564800.0,
    "end_time": 1699564803.5,
    "duration_ms": 3500,
    "status": "success",
    "total_cost": 0.0073,
    "total_tokens": 1842,
    "llm_calls": 2,
    "tool_calls": 1
}
```

#### Span
A **span** represents a single operation within a trace (LLM call, tool execution, etc.).

```python
{
    "span_id": "span-456",
    "trace_id": "abc-123-def",
    "parent_span_id": "span-123",  # null for root
    "name": "openai.chat.completions.create",
    "span_type": "llm",
    "model": "gpt-4o-mini",
    "start_time": 1699564801.0,
    "end_time": 1699564802.5,
    "duration_ms": 1500,
    "status": "success",
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 75,
        "total_tokens": 225,
        "total_cost": 0.0042
    },
    "input_preview": "{\"messages\": [{\"role\": \"user\", ...}]}",
    "output_preview": "{\"content\": \"Python is a high-level...\"}"
}
```

### Context Propagation

OAT uses Python's `contextvars` to propagate trace and span context across async boundaries:

```python
@trace(span_type="agent")
async def agent():
    # New trace created, trace_id set in context

    await llm_call()  # Inherits trace_id, creates child span
    await tool_call()  # Inherits trace_id, creates child span

    # Trace context restored after agent completes
```

---

## 📚 Usage Guide

### Tracing with Decorators

#### Basic Agent Tracing

```python
from oat import trace

@trace(name="my_agent", span_type="agent")
async def my_agent(query: str):
    # Agent logic here
    return result
```

#### LLM Tracing (Manual)

```python
from oat import trace_llm

@trace_llm(name="gpt4_call", model="gpt-4o-mini", service_provider="openai")
async def call_gpt4(prompt: str):
    # LLM call here
    return response
```

#### Tool Tracing

```python
from oat import trace_tool

@trace_tool(name="web_search")
async def search_web(query: str):
    # Tool logic
    return results
```

#### Database Tracing

```python
from oat import trace_database

@trace_database(name="user_query", operation="select")
def get_user(user_id: int):
    # Database query
    return user
```

#### Retrieval Tracing (RAG)

```python
from oat import trace_retrieval

@trace_retrieval(name="vector_search", collection="knowledge_base")
async def search_vectors(query: str, k: int = 5):
    # Vector search logic
    return results
```

### Manual Span Creation

Use context managers when decorators aren't suitable:

```python
from oat import span

@trace(span_type="agent")
async def complex_agent(query: str):
    # Step 1: Manual span for custom logic
    with span("preprocessing", span_type="chain") as s:
        cleaned_query = preprocess(query)
        s.set_output(cleaned_query)

    # Step 2: Nested spans
    with span("retrieval_pipeline", span_type="chain"):
        with span("embedding", span_type="embedding") as s:
            embedding = await get_embedding(cleaned_query)
            s.set_output({"dimensions": len(embedding)})

        with span("search", span_type="retrieval") as s:
            results = await vector_search(embedding, k=5)
            s.set_output(f"Found {len(results)} results")

    # Step 3: LLM call (auto-traced if using patch_openai)
    response = await call_llm(cleaned_query, results)

    return response
```

### Auto-Instrumentation

#### OpenAI

```python
from oat.integrations import patch_openai

# Patch once at startup
patch_openai()

# Now all OpenAI calls are automatically traced
from openai import AsyncOpenAI

client = AsyncOpenAI()

# This is automatically a traced LLM span!
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)

# Embeddings are also traced
embedding = await client.embeddings.create(
    model="text-embedding-3-small",
    input="Some text"
)
```

#### Anthropic

```python
from oat.integrations import patch_anthropic

# Patch Anthropic
patch_anthropic()

from anthropic import AsyncAnthropic

client = AsyncAnthropic()

# Automatically traced!
response = await client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
```

---

## 🏷️ Span Types

OpenAgentTrace supports 14 semantic span types:

### Core Agent Operations

| Type | Description | Use Case | Example |
|------|-------------|----------|---------|
| `agent` | Root agent execution | Entry point for autonomous agents | `@trace(span_type="agent")` |
| `llm` | LLM inference call | GPT, Claude, Gemini API calls | Auto-patched via `patch_openai()` |
| `tool` | Tool/function execution | Calculator, search, API calls | `@trace_tool()` |
| `retrieval` | RAG/vector search | Pinecone, Chroma, Qdrant queries | `@trace_retrieval(collection="kb")` |
| `guardrail` | Safety/validation checks | Content moderation, PII detection | `@trace_guardrail(action="check")` |
| `handoff` | Agent-to-agent transfer | Multi-agent systems | `@trace(span_type="handoff")` |

### Supporting Operations

| Type | Description | Use Case | Example |
|------|-------------|----------|---------|
| `chain` | Chain of operations | LangChain, sequential steps | `with span("chain", span_type="chain")` |
| `embedding` | Embedding generation | Text/image embeddings | Auto-patched via integrations |
| `rerank` | Reranking operation | Cohere rerank, custom rerankers | `@trace(span_type="rerank")` |
| `memory` | Memory read/write | Conversation history, context | `@trace_memory(operation="read")` |

### Infrastructure

| Type | Description | Use Case | Example |
|------|-------------|----------|---------|
| `http` | HTTP request | External API calls | `@trace_http(method="POST")` |
| `database` | Database query | SQL, NoSQL operations | `@trace_database(operation="insert")` |
| `cache` | Cache operation | Redis, Memcached | `@trace_cache(operation="get")` |
| `file_io` | File operations | Read/write files | `@trace_file_io(operation="read")` |

---

## ⚙️ Configuration

### tracer.yaml

Create a `tracer.yaml` file in your project root:

```yaml
# Service Identification
service:
  name: "my-ai-agent"
  version: "1.0.0"
  environment: "production"  # dev, staging, production

# Export Configuration
export:
  # Local storage (always enabled)
  local:
    data_dir: ".oat"  # Where to store traces

  # Remote server (optional)
  remote:
    enabled: true
    url: "http://localhost:8787"
    flush_interval: 1.0  # seconds
    batch_size: 100

# Auto-Instrumentation
instrumentation:
  enabled: true
  targets:
    - module: "openai.chat.completions"
      function: "create"
      span_type: "llm"
    - module: "anthropic.messages"
      function: "create"
      span_type: "llm"

# Sampling (for high-volume agents)
sampling:
  rate: 1.0  # 1.0 = 100%, 0.5 = 50%
  error_sampling: 1.0  # Always sample errors
  slow_trace_threshold_ms: 5000  # Always sample slow traces

# Cost Configuration (optional overrides)
pricing:
  gpt-4o-mini:
    prompt_tokens_per_1k: 0.00015
    completion_tokens_per_1k: 0.0006
  custom-model:
    prompt_tokens_per_1k: 0.001
    completion_tokens_per_1k: 0.002
```

### Environment Variables

```bash
# Service configuration
export OAT_SERVICE_NAME="my-agent"
export OAT_ENVIRONMENT="production"

# Remote export
export OAT_EXPORT_URL="http://localhost:8787"
export OAT_EXPORT_ENABLED="true"

# Storage
export OAT_DATA_DIR=".oat"

# Sampling
export OAT_SAMPLING_RATE="1.0"
```

### Programmatic Configuration

```python
from oat import get_tracer
from pathlib import Path

tracer = get_tracer(
    service_name="my-agent",
    data_dir=Path(".oat"),
    export_url="http://localhost:8787",
    auto_flush=True,
    flush_interval=1.0
)
```

---

## 📊 Dashboard

### Features

#### 1. Traces Explorer
- **List View**: All traces with filters
  - Filter by status (success/error)
  - Filter by date range
  - Filter by span type
  - Search by trace ID or name
- **Metrics**: Total traces, error rate, avg cost, avg latency

#### 2. Trace Detail View
- **DAG Visualization**: Interactive node graph showing execution flow
- **Waterfall Chart**: Timeline view of span durations
- **Span Inspector**: Click any span to see:
  - Input/output data
  - Token usage and cost
  - Model and provider
  - Media inputs (for vision models)
  - Error details (if failed)

#### 3. Analytics Dashboard
- **Overview Metrics** (24 hours):
  - Total requests
  - Error rate
  - Total cost
  - Avg latency
- **Latency Percentiles** (by span type):
  - P50, P95, P99
  - Time-series charts
- **Token Usage Trends**:
  - Tokens per hour
  - Cost per hour
  - By model breakdown

#### 4. Prompts Registry
- **Template Management**: Store and version prompt templates
- **A/B Testing**: Compare prompt performance
- **Cost Analysis**: Track cost by prompt version

### Screenshots

<img width="2484" height="1110" alt="Screenshot 2026-02-04 212855" src="https://github.com/user-attachments/assets/7177192b-3a24-4475-a2ee-26d47445cec6" />
<img width="2101" height="1279" alt="Screenshot 2026-02-04 230013" src="https://github.com/user-attachments/assets/a81c4eb8-9f0b-4481-adc0-805856af0a19" />
<img width="2464" height="1371" alt="Screenshot 2026-02-04 163301" src="https://github.com/user-attachments/assets/8c696e49-1190-479c-8174-6fab0d37ae8c" />
<img width="2459" height="1179" alt="Screenshot 2026-02-04 163323" src="https://github.com/user-attachments/assets/a03000ed-656f-43a9-83a1-35027afb3877" />
<img width="2454" height="889" alt="Screenshot 2026-02-04 163340" src="https://github.com/user-attachments/assets/426c4460-1815-4935-b7ca-34fc285e18a6" />
<img width="2458" height="1220" alt="Screenshot 2026-02-04 163355" src="https://github.com/user-attachments/assets/847d2a78-b390-4999-957b-681fba5063cb" />
<img width="2473" height="1182" alt="Screenshot 2026-02-04 163444" src="https://github.com/user-attachments/assets/a2f960b9-6bed-4ca7-8606-d01dc0799d00" />


---

## 🔌 Server API

### REST Endpoints

#### Traces

```bash
# List all traces
GET /traces?limit=50&offset=0&status=success

# Get specific trace with all spans
GET /traces/{trace_id}?include_blobs=true

# Delete trace
DELETE /traces/{trace_id}
```

#### Spans

```bash
# Get specific span
GET /spans/{span_id}?include_blobs=true

# Ingest spans (used by SDK)
POST /ingest
```

#### Analytics

```bash
# Overview metrics (last 24h)
GET /analytics/overview

# Latency percentiles by span type
GET /analytics/latency

# Custom time-series query
POST /analytics/query
```

#### Feedback

```bash
# Add user feedback to trace
POST /traces/{trace_id}/feedback
{
  "score": 1,  # -1, 0, or 1
  "feedback": "Great response!"
}
```

### WebSocket (Real-Time Updates)

```javascript
const ws = new WebSocket('ws://localhost:8787/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('New span:', data);
};
```

### API Client Examples

#### Python

```python
import httpx

# Get traces
response = httpx.get("http://localhost:8787/traces")
traces = response.json()

# Get specific trace
trace_id = "abc-123"
response = httpx.get(f"http://localhost:8787/traces/{trace_id}?include_blobs=true")
trace_details = response.json()
```

#### JavaScript

```javascript
// Fetch traces
const response = await fetch('http://localhost:8787/traces');
const traces = await response.json();

// Add feedback
await fetch(`http://localhost:8787/traces/${traceId}/feedback`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ score: 1, feedback: 'Excellent!' })
});
```

---

## 🔗 Integrations

### Supported Providers

| Provider | Status | Auto-Patch | Models Supported |
|----------|--------|------------|------------------|
| **OpenAI** | ✅ Full | `patch_openai()` | GPT-4, GPT-4o, GPT-3.5, Embeddings, Vision |
| **Anthropic** | ✅ Full | `patch_anthropic()` | Claude 3.5 Sonnet, Claude 3 Opus/Haiku |
| **Google** | 🚧 WIP | Manual | Gemini Pro, Gemini Ultra |
| **Mistral** | 🚧 WIP | Manual | Mistral Large, Medium, Small |
| **Cohere** | 🚧 WIP | Manual | Command, Embed, Rerank |

### LangChain Integration

```python
from langchain.callbacks import OpenAgentTraceCallback
from langchain.chat_models import ChatOpenAI
from oat.integrations import patch_openai

# Patch OpenAI
patch_openai()

# Use LangChain normally - calls are auto-traced!
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is AI?")
```

### LlamaIndex Integration

```python
from llama_index import VectorStoreIndex
from oat import trace

# Wrap with trace decorator
@trace(span_type="agent")
def rag_query(query: str):
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return query_engine.query(query)
```

---

## 🎓 Advanced Topics

### Custom Exporters

Create your own exporter to send traces to custom destinations:

```python
from oat.exporters import SpanExporter
from oat.models import Span

class CustomExporter(SpanExporter):
    def export(self, span: Span, inputs=None, outputs=None):
        # Send to your custom backend
        print(f"Exporting {span.name} to custom backend")
        # Your logic here

    def shutdown(self):
        print("Shutting down custom exporter")

# Use custom exporter
from oat import get_tracer

tracer = get_tracer(service_name="my-agent")
tracer.exporter = CustomExporter()
```

### Distributed Tracing

Propagate traces across services:

```python
from oat import get_current_trace_id, set_trace_id

# Service A
@trace(span_type="agent")
async def service_a():
    trace_id = get_current_trace_id()

    # Send trace_id in headers
    headers = {"X-Trace-ID": trace_id}
    response = await httpx.post("http://service-b", headers=headers)
    return response

# Service B
@trace(span_type="agent")
async def service_b(request):
    # Extract and set trace_id
    trace_id = request.headers.get("X-Trace-ID")
    if trace_id:
        set_trace_id(trace_id)

    # This span will be part of Service A's trace!
    return await process_request(request)
```

### Custom Span Metadata

```python
@trace(span_type="llm", metadata={"experiment": "prompt-v2", "user_id": "123"})
async def experimental_llm_call(prompt: str):
    # Metadata is stored with span for filtering
    return await call_llm(prompt)
```

### Sampling Strategies

```python
from oat import get_tracer
import random

tracer = get_tracer(service_name="high-volume-agent")

# Only trace 10% of requests
if random.random() < 0.1:
    @trace(span_type="agent")
    async def agent(query):
        return await process(query)
else:
    # Don't trace
    async def agent(query):
        return await process(query)
```

### Cost Optimization

Track and optimize LLM costs:

```python
from oat import trace, get_tracer

@trace(span_type="agent")
async def cost_aware_agent(query: str):
    # Try cheaper model first
    with span("cheap_attempt", span_type="llm") as s:
        response = await call_gpt_mini(query)

        # Check if response is good enough
        if is_quality_sufficient(response):
            s.metadata["selected"] = True
            return response

    # Fall back to expensive model
    with span("expensive_fallback", span_type="llm") as s:
        s.metadata["selected"] = True
        return await call_gpt4(query)
```

---

## 📝 Examples

### Complete Agent Example

See `examples/demo_agent.py` for a full-featured agent with:
- ✅ OpenAI LLM calls
- ✅ SQLite database for memory
- ✅ Tool calling (search, calculator)
- ✅ Multi-step reasoning chains
- ✅ Vision model integration
- ✅ Error handling and retries

```python
# Run the demo
python examples/demo_agent.py
```

### Simple Coding Agent

See `examples/coding_agent.py` for a minimal example:

```python
from oat import trace, get_tracer
from oat.integrations import patch_openai
from openai import AsyncOpenAI

tracer = get_tracer(
    service_name="coding-agent",
    export_url="http://localhost:8787"
)

patch_openai()

@trace(name="agent.code_question", span_type="agent")
async def ask_coding_question(question: str):
    client = AsyncOpenAI()

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a Python expert."},
            {"role": "user", "content": question}
        ]
    )

    return response.choices[0].message.content

# Run
import asyncio
result = asyncio.run(ask_coding_question(
    "Write a function to check if a number is prime"
))
print(result)
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Traces not appearing in dashboard

**Symptoms**: Agent runs but no traces show up

**Solutions**:
```bash
# Check if spans are in local database
python -c "import sqlite3; conn = sqlite3.connect('.oat/traces.db'); print(conn.execute('SELECT COUNT(*) FROM spans').fetchone()[0], 'spans found')"

# Check server logs
uvicorn server.main:app --port 8787 --log-level debug

# Verify export URL
python -c "from oat import get_tracer; t = get_tracer(export_url='http://localhost:8787'); print('Exporter:', t.exporter)"
```

#### 2. Auto-instrumentation not working

**Symptoms**: OpenAI calls not traced automatically

**Solutions**:
```python
# Ensure patch is called BEFORE importing OpenAI
from oat.integrations import patch_openai
patch_openai()  # Call this FIRST!

# Then import OpenAI
from openai import AsyncOpenAI

# Verify patch worked
import openai
print("Patched:", hasattr(openai, '_oat_patched'))
```

#### 3. High memory usage

**Symptoms**: Agent uses too much memory with tracing enabled

**Solutions**:
```python
# Reduce flush interval
tracer = get_tracer(
    service_name="my-agent",
    flush_interval=0.5  # Flush more frequently
)

# Disable input/output capture for large payloads
@trace(span_type="llm", capture_input=False, capture_output=False)
async def large_llm_call(huge_prompt: str):
    return await call_llm(huge_prompt)

# Use sampling
import random
if random.random() < 0.1:  # Trace only 10%
    @trace(span_type="agent")
    async def agent(query):
        return await process(query)
```

#### 4. Trace hierarchy broken

**Symptoms**: Parent-child relationships incorrect

**Solutions**:
This was a known bug that has been fixed in the latest version. Update to the latest code and ensure:

```python
# Each agent execution creates ONE trace
@trace(span_type="agent")  # Creates new trace
async def my_agent():
    # All child operations inherit this trace
    await llm_call()  # Child span
    await tool_call()  # Child span
```

#### 5. Dashboard shows wrong outputs

**Symptoms**: Span outputs don't match actual execution

**Solutions**:
- Ensure you're running the latest version with trace context fixes
- Check that `include_blobs=true` is set when fetching trace details
- Verify blob storage is working: `ls .oat/blobs/`

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourusername/openagent-trace.git
cd openagent-trace

# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linter
ruff check .

# Run type checker
mypy oat/
```

### Code Style

- **Python**: Follow PEP 8, use `black` for formatting
- **JavaScript**: Follow Airbnb style guide, use Prettier
- **Type hints**: Required for all public APIs
- **Docstrings**: Google style for all modules, classes, and functions

---

## 🗺️ Roadmap

### v0.2.0 (Q2 2024)
- [ ] Google Gemini integration
- [ ] Mistral AI integration
- [ ] Prometheus metrics export
- [ ] OpenTelemetry compatibility layer
- [ ] Trace comparison UI

### v0.3.0 (Q3 2024)
- [ ] Guardrail library integration (NeMo, Guardrails)
- [ ] Prompt versioning and management
- [ ] A/B testing framework
- [ ] Cost budgeting and alerts
- [ ] Multi-tenant support

### v1.0.0 (Q4 2024)
- [ ] Production-ready stability
- [ ] Enterprise features (SSO, RBAC)
- [ ] Cloud-hosted option
- [ ] Alerting and monitoring
- [ ] Data retention policies

### Future
- [ ] AutoML for prompt optimization
- [ ] Anomaly detection for agent behavior
- [ ] Hallucination detection
- [ ] Evaluation framework
- [ ] Agent collaboration tracing

---

## 🙏 Acknowledgments

OpenAgentTrace is inspired by and builds upon ideas from:

- **OpenTelemetry**: Distributed tracing standards
- **LangSmith**: Agent-specific observability patterns
- **Arize Phoenix**: ML observability concepts
- **Grafana/Prometheus**: Time-series monitoring best practices

Special thanks to the open-source community for their tools and frameworks.

---

## 📄 License

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

## 📞 Support

- **Documentation**: [docs.openagent-trace.dev](https://docs.openagent-trace.dev)
- **GitHub Issues**: [github.com/yourusername/openagent-trace/issues](https://github.com/yourusername/openagent-trace/issues)
- **Discord**: [discord.gg/openagent-trace](https://discord.gg/openagent-trace)
- **Email**: support@openagent-trace.dev

---

<div align="center">

**⭐ Star us on GitHub if you find OpenAgentTrace useful!**

Made with ❤️ by the OpenAgentTrace team

[Documentation](https://docs.openagent-trace.dev) • [Examples](./examples) • [Changelog](./CHANGELOG.md) • [Contributing](./CONTRIBUTING.md)

</div>
