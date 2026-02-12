"""
OpenAgentTrace Server - The observability backend.

Features:
- Span ingestion API
- Trace querying and search
- Real-time WebSocket updates
- DuckDB-powered analytics
- Feedback/scoring system

Run:
    uvicorn server.main:app --reload --port 8787
"""

import json
import sqlite3
import hashlib
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import duckdb
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============ CONFIGURATION ============
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / ".oat"
DB_PATH = DATA_DIR / "traces.db"
BLOB_DIR = DATA_DIR / "blobs"
ANALYTICS_DB = DATA_DIR / "analytics.duckdb"


# ============ PYDANTIC MODELS ============

class SpanIngest(BaseModel):
    """Span data for ingestion."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    span_type: str = "function"
    service_name: Optional[str] = None
    
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
    service_provider: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    
    retrieval_query: Optional[str] = None
    retrieval_k: Optional[int] = None
    retrieval_scores: Optional[List[float]] = None
    
    guardrail_triggered: Optional[bool] = None
    guardrail_action: Optional[str] = None
    
    # Multimodal inputs/outputs
    media_inputs: Optional[List[Dict[str, Any]]] = None
    media_outputs: Optional[List[Dict[str, Any]]] = None
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """User feedback for a span."""
    score: int = Field(..., ge=-1, le=1)
    feedback: Optional[str] = None


class SearchRequest(BaseModel):
    """Search query parameters."""
    query: str
    limit: int = 50
    span_types: Optional[List[str]] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# ============ DATABASE SETUP ============

def init_sqlite():
    """Initialize SQLite database with schema."""
    DATA_DIR.mkdir(exist_ok=True)
    BLOB_DIR.mkdir(exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            span_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            service_name TEXT,
            parent_span_id TEXT,
            name TEXT NOT NULL,
            span_type TEXT NOT NULL,
            
            start_time REAL NOT NULL,
            end_time REAL,
            duration_ms REAL,
            
            status TEXT NOT NULL DEFAULT 'running',
            error_message TEXT,
            error_type TEXT,
            
            input_hash TEXT,
            output_hash TEXT,
            input_preview TEXT,
            output_preview TEXT,
            
            model TEXT,
            service_provider TEXT,
            usage_json TEXT,
            
            tool_name TEXT,
            tool_parameters_json TEXT,
            
            retrieval_query TEXT,
            retrieval_k INTEGER,
            retrieval_scores_json TEXT,
            
            guardrail_triggered INTEGER,
            guardrail_action TEXT,
            
            media_inputs_json TEXT,
            media_outputs_json TEXT,
            
            metadata_json TEXT,
            tags_json TEXT,
            
            score INTEGER,
            feedback TEXT,
            
            created_at REAL DEFAULT (unixepoch('now')),
            updated_at REAL DEFAULT (unixepoch('now'))
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_type ON spans(span_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status)")
    
    # Migration: Add media columns if they don't exist (for existing databases)
    try:
        conn.execute("ALTER TABLE spans ADD COLUMN media_inputs_json TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE spans ADD COLUMN media_outputs_json TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces_summary (
            trace_id TEXT PRIMARY KEY,
            name TEXT,
            service_name TEXT,
            start_time REAL,
            end_time REAL,
            duration_ms REAL,
            status TEXT,
            span_count INTEGER DEFAULT 0,
            llm_calls INTEGER DEFAULT 0,
            tool_calls INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            error_count INTEGER DEFAULT 0,
            updated_at REAL DEFAULT (unixepoch('now'))
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_start ON traces_summary(start_time DESC)")
    
    # Auto-migration: Add service_provider if missing
    try:
        conn.execute("ALTER TABLE spans ADD COLUMN service_provider TEXT")
    except sqlite3.OperationalError:
        pass  # Column likely exists
    
    # Auto-migration: Add service_name if missing
    try:
        conn.execute("ALTER TABLE spans ADD COLUMN service_name TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE traces_summary ADD COLUMN service_name TEXT")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()


def init_duckdb():
    """Initialize DuckDB for analytics."""
    conn = duckdb.connect(str(ANALYTICS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp TIMESTAMP,
            trace_id VARCHAR,
            span_id VARCHAR,
            name VARCHAR,
            span_type VARCHAR,
            status VARCHAR,
            duration_ms DOUBLE,
            tokens_total INTEGER DEFAULT 0,
            cost DOUBLE DEFAULT 0.0,
            model VARCHAR,
            service_provider VARCHAR,
            service_name VARCHAR
        )
    """)
    
    # Auto-migration for DuckDB
    try:
        conn.execute("ALTER TABLE metrics ADD COLUMN service_provider VARCHAR")
    except:
        pass
    try:
        conn.execute("ALTER TABLE metrics ADD COLUMN service_name VARCHAR")
    except:
        pass
    conn.close()


# ============ STORAGE HELPERS ============

def save_blob(data: Any) -> Optional[str]:
    """Save data to blob storage."""
    if data is None:
        return None
    
    try:
        text = json.dumps(data, default=str, sort_keys=True)
    except:
        text = str(data)
    
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    blob_path = BLOB_DIR / content_hash
    
    if not blob_path.exists():
        blob_path.write_text(text, encoding='utf-8')
    
    return content_hash


def get_blob(content_hash: Optional[str]) -> Any:
    """Retrieve data from blob storage."""
    if not content_hash:
        return None
    
    blob_path = BLOB_DIR / content_hash
    if not blob_path.exists():
        return None
    
    try:
        return json.loads(blob_path.read_text(encoding='utf-8'))
    except:
        return None


# ============ WEBSOCKET MANAGER ============

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ============ APP LIFECYCLE ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup
    print("🚀 Starting OpenAgentTrace Server...")
    init_sqlite()
    init_duckdb()
    print(f"📁 Data directory: {DATA_DIR.absolute()}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down OpenAgentTrace Server...")


# ============ FASTAPI APP ============

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="OpenAgentTrace",
        description="The Open Standard for AI Agent Observability",
        version="0.1.0",
        lifespan=lifespan,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============ INGESTION ENDPOINTS ============

@app.post("/ingest")
async def ingest_span(span: SpanIngest):
    """Ingest a span from the SDK."""
    conn = sqlite3.connect(DB_PATH)
    
    # Save blobs
    input_hash = save_blob(span.inputs)
    output_hash = save_blob(span.outputs)
    
    # Upsert span
    conn.execute("""
        INSERT OR REPLACE INTO spans (
            span_id, trace_id, service_name, parent_span_id, name, span_type,
            start_time, end_time, duration_ms,
            status, error_message, error_type,
            input_hash, output_hash, input_preview, output_preview,
            model, service_provider, usage_json,
            tool_name, tool_parameters_json,
            retrieval_query, retrieval_k, retrieval_scores_json,
            guardrail_triggered, guardrail_action,
            media_inputs_json, media_outputs_json,
            metadata_json, tags_json,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        span.span_id,
        span.trace_id,
        span.service_name,
        span.parent_span_id,
        span.name,
        span.span_type,
        span.start_time,
        span.end_time,
        span.duration_ms,
        span.status,
        span.error_message,
        span.error_type,
        input_hash,
        output_hash,
        span.input_preview,
        span.output_preview,
        span.model,
        span.service_provider,
        json.dumps(span.usage) if span.usage else None,
        span.tool_name,
        json.dumps(span.tool_parameters) if span.tool_parameters else None,
        span.retrieval_query,
        span.retrieval_k,
        json.dumps(span.retrieval_scores) if span.retrieval_scores else None,
        1 if span.guardrail_triggered else (0 if span.guardrail_triggered is False else None),
        span.guardrail_action,
        json.dumps(getattr(span, 'media_inputs', None)) if getattr(span, 'media_inputs', None) else None,
        json.dumps(getattr(span, 'media_outputs', None)) if getattr(span, 'media_outputs', None) else None,
        json.dumps(span.metadata) if span.metadata else None,
        json.dumps(span.tags) if span.tags else None,
        datetime.now().timestamp(),
    ))
    
    # Update trace summary
    conn.execute("""
        INSERT OR REPLACE INTO traces_summary (
            trace_id, name, service_name, start_time, end_time, duration_ms, status,
            span_count, llm_calls, tool_calls, total_tokens, total_cost, error_count,
            updated_at
        )
        SELECT 
            trace_id,
            (SELECT name FROM spans WHERE trace_id = ? AND parent_span_id IS NULL LIMIT 1),
            MAX(service_name),
            MIN(start_time),
            MAX(end_time),
            CASE WHEN MAX(end_time) IS NOT NULL THEN (MAX(end_time) - MIN(start_time)) * 1000 ELSE NULL END,
            CASE 
                WHEN SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) > 0 THEN 'error'
                WHEN COUNT(CASE WHEN status = 'running' THEN 1 END) > 0 THEN 'running'
                ELSE 'success'
            END,
            COUNT(*),
            SUM(CASE WHEN span_type = 'llm' THEN 1 ELSE 0 END),
            SUM(CASE WHEN span_type = 'tool' THEN 1 ELSE 0 END),
            COALESCE(SUM(json_extract(usage_json, '$.total_tokens')), 0),
            COALESCE(SUM(json_extract(usage_json, '$.total_cost')), 0.0),
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END),
            unixepoch('now')
        FROM spans
        WHERE trace_id = ?
        GROUP BY trace_id
    """, (span.trace_id, span.trace_id))
    
    conn.commit()
    conn.close()
    
    # Write to DuckDB for analytics (completed spans only)
    if span.end_time:
        duck = duckdb.connect(str(ANALYTICS_DB))
        duck.execute("""
            INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.fromtimestamp(span.start_time),
            span.trace_id,
            span.span_id,
            span.name,
            span.span_type,
            span.status,
            span.duration_ms or 0,
            span.usage.get('total_tokens', 0) if span.usage else 0,
            span.usage.get('total_cost', 0) if span.usage else 0,
            span.model,
            span.service_provider,
            span.service_name,
        ))
        duck.close()
    
    # Broadcast to WebSocket clients
    await manager.broadcast({
        "type": "span_update",
        "trace_id": span.trace_id,
        "span_id": span.span_id,
        "status": span.status,
    })
    
    return {"status": "ok", "span_id": span.span_id}


# ============ TRACE ENDPOINTS ============
@app.get("/services")
def list_services():
    """Get a list of all unique agents/services."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("SELECT DISTINCT service_name FROM spans WHERE service_name IS NOT NULL").fetchall()
    conn.close()
    
    return [row[0] for row in rows]

@app.get("/traces")
def list_traces(
    limit: int = 100, 
    offset: int = 0, 
    status: Optional[str] = None,
    service_name: Optional[str] = None
):
    """List traces, optionally filtered by status or agent name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Query traces_summary which has the correct root span name
    query = """
        SELECT 
            trace_id, 
            name, 
            service_name,
            start_time, 
            end_time, 
            duration_ms,
            status,
            span_count, 
            llm_calls,
            tool_calls,
            total_tokens,
            total_cost,
            error_count
        FROM traces_summary
    """
    
    conditions = []
    params = []
    
    if status:
        if status == 'success':
            conditions.append("status != 'error'")
        else:
            conditions.append("status = ?")
            params.append(status)
            
    if service_name:
        conditions.append("service_name = ?")
        params.append(service_name)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

@app.get("/traces/{trace_id}")
def get_trace(trace_id: str, include_blobs: bool = False):
    """Get all spans for a trace."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute(
        "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
        (trace_id,)
    ).fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    spans = []
    for row in rows:
        span_dict = dict(row)
        
        # Parse JSON fields
        for json_field in ['usage_json', 'tool_parameters_json', 'retrieval_scores_json', 'metadata_json', 'tags_json', 'media_inputs_json', 'media_outputs_json']:
            key = json_field.replace('_json', '')
            if span_dict.get(json_field):
                span_dict[key] = json.loads(span_dict[json_field])
            else:
                span_dict[key] = None if key != 'metadata' else {}
            del span_dict[json_field]
        
        if include_blobs:
            span_dict['inputs'] = get_blob(span_dict.get('input_hash'))
            span_dict['outputs'] = get_blob(span_dict.get('output_hash'))
        
        spans.append(span_dict)
    
    return spans


@app.get("/spans/{span_id}")
def get_span(span_id: str, include_blobs: bool = True):
    """Get a single span by ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    row = conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,)).fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Span not found")
    
    span_dict = dict(row)
    
    # Parse JSON fields
    for json_field in ['usage_json', 'tool_parameters_json', 'retrieval_scores_json', 'metadata_json', 'tags_json', 'media_inputs_json', 'media_outputs_json']:
        key = json_field.replace('_json', '')
        if span_dict.get(json_field):
            span_dict[key] = json.loads(span_dict[json_field])
        else:
            span_dict[key] = None if key != 'metadata' else {}
        del span_dict[json_field]
    
    if include_blobs:
        span_dict['inputs'] = get_blob(span_dict.get('input_hash'))
        span_dict['outputs'] = get_blob(span_dict.get('output_hash'))
    
    return span_dict


@app.delete("/traces/{trace_id}")
def delete_trace(trace_id: str):
    """Delete a trace and all its spans."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace_id,))
    conn.execute("DELETE FROM traces_summary WHERE trace_id = ?", (trace_id,))
    conn.commit()
    conn.close()
    
    # Also delete from DuckDB
    duck = duckdb.connect(str(ANALYTICS_DB))
    duck.execute("DELETE FROM metrics WHERE trace_id = ?", (trace_id,))
    duck.close()
    
    return {"status": "deleted", "trace_id": trace_id}


# ============ FEEDBACK ENDPOINTS ============

@app.post("/spans/{span_id}/feedback")
def score_span(span_id: str, req: FeedbackRequest):
    """Add user feedback/score to a span."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "UPDATE spans SET score = ?, feedback = ?, updated_at = ? WHERE span_id = ?",
        (req.score, req.feedback, datetime.now().timestamp(), span_id)
    )
    
    if cursor.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Span not found")
    
    conn.commit()
    conn.close()
    
    return {"status": "updated", "span_id": span_id}


# ============ SEARCH ENDPOINTS ============

@app.post("/search")
def search_traces(req: SearchRequest):
    """Full-text search across traces."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    query = """
        SELECT DISTINCT t.* FROM traces_summary t
        JOIN spans s ON t.trace_id = s.trace_id
        WHERE (
            s.name LIKE ? 
            OR s.input_preview LIKE ? 
            OR s.output_preview LIKE ?
            OR s.error_message LIKE ?
            OR s.model LIKE ?
        )
    """
    params = [f"%{req.query}%"] * 5
    
    if req.span_types:
        placeholders = ','.join(['?' for _ in req.span_types])
        query += f" AND s.span_type IN ({placeholders})"
        params.extend(req.span_types)
    
    if req.status:
        query += " AND t.status = ?"
        params.append(req.status)
    
    if req.start_date:
        query += " AND t.start_time >= ?"
        params.append(datetime.fromisoformat(req.start_date).timestamp())
    
    if req.end_date:
        query += " AND t.start_time <= ?"
        params.append(datetime.fromisoformat(req.end_date).timestamp())
    
    query += f" ORDER BY t.start_time DESC LIMIT ?"
    params.append(req.limit)
    
    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


# ============ ANALYTICS ENDPOINTS ============

@app.get("/analytics/overview")
def get_analytics_overview(
    hours: int = Query(24, le=168), 
    service_name: Optional[str] = None  # <--- NEW PARAMETER
):
    """Get overview analytics for the dashboard, optionally filtered by agent."""
    duck = duckdb.connect(str(ANALYTICS_DB))
    cutoff = datetime.now() - timedelta(hours=hours)
    
    # helper to manage dynamic filters
    base_condition = "timestamp >= ?"
    params = [cutoff]
    
    if service_name:
        base_condition += " AND service_name = ?"
        params.append(service_name)

    try:
        # 1. Overall stats
        stats_query = f"""
            SELECT 
                COUNT(DISTINCT trace_id) as trace_count,
                COUNT(*) as span_count,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                AVG(duration_ms) as avg_duration_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms,
                SUM(tokens_total) as total_tokens,
                SUM(cost) as total_cost
            FROM metrics
            WHERE {base_condition}
        """
        stats = duck.execute(stats_query, params).fetchone()
        
        # 2. By span type
        type_query = f"""
            SELECT 
                span_type,
                COUNT(*) as count,
                AVG(duration_ms) as avg_duration_ms,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration_ms,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM metrics
            WHERE {base_condition}
            GROUP BY span_type
            ORDER BY count DESC
        """
        by_type = duck.execute(type_query, params).fetchall()
        
        # 3. By model
        model_query = f"""
            SELECT 
                model,
                COUNT(*) as count,
                SUM(tokens_total) as tokens,
                SUM(cost) as cost,
                AVG(duration_ms) as avg_duration_ms
            FROM metrics
            WHERE {base_condition} AND model IS NOT NULL
            GROUP BY model
            ORDER BY count DESC
        """
        by_model = duck.execute(model_query, params).fetchall()
        
        # 4. Time series (hourly)
        ts_query = f"""
            SELECT 
                DATE_TRUNC('hour', timestamp) as hour,
                COUNT(*) as spans,
                SUM(tokens_total) as tokens,
                SUM(cost) as cost,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM metrics
            WHERE {base_condition}
            GROUP BY 1
            ORDER BY 1 ASC
        """
        timeseries = duck.execute(ts_query, params).fetchall()
        
        return {
            "overview": {
                "trace_count": stats[0] or 0,
                "span_count": stats[1] or 0,
                "error_count": stats[2] or 0,
                "error_rate": round((stats[2] or 0) / max(stats[1] or 1, 1) * 100, 2),
                "avg_duration_ms": round(stats[3] or 0, 2),
                "p95_duration_ms": round(stats[4] or 0, 2),
                "total_tokens": stats[5] or 0,
                "total_cost": round(stats[6] or 0, 4),
            },
            "by_type": [
                {
                    "span_type": r[0],
                    "count": r[1],
                    "avg_duration_ms": round(r[2] or 0, 2),
                    "p95_duration_ms": round(r[3] or 0, 2),
                    "errors": r[4],
                }
                for r in by_type
            ],
            "by_model": [
                {
                    "model": r[0],
                    "count": r[1],
                    "tokens": r[2] or 0,
                    "cost": round(r[3] or 0, 4),
                    "avg_duration_ms": round(r[4] or 0, 2),
                }
                for r in by_model
            ],
            "timeseries": [
                {
                    "hour": r[0].isoformat() if r[0] else None,
                    "spans": r[1],
                    "tokens": r[2] or 0,
                    "cost": round(r[3] or 0, 4),
                    "errors": r[4],
                }
                for r in timeseries
            ],
        }
    except Exception as e:
        print(f"Analytics Error: {e}")
        return {"error": str(e), "overview": {}, "timeseries": []}
    finally:
        duck.close()

@app.get("/analytics/latency")
def get_latency_analytics(hours: int = Query(24, le=168)):
    """Get latency percentiles by span type."""
    duck = duckdb.connect(str(ANALYTICS_DB))
    cutoff = datetime.now() - timedelta(hours=hours)
    
    result = duck.execute("""
        SELECT 
            span_type,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY duration_ms) as p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY duration_ms) as p90,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99,
            MIN(duration_ms) as min_ms,
            MAX(duration_ms) as max_ms,
            COUNT(*) as count
        FROM metrics
        WHERE timestamp >= ?
        GROUP BY span_type
        ORDER BY count DESC
    """, [cutoff]).fetchall()
    
    duck.close()
    
    return [
        {
            "span_type": r[0],
            "p50": round(r[1] or 0, 2),
            "p90": round(r[2] or 0, 2),
            "p95": round(r[3] or 0, 2),
            "p99": round(r[4] or 0, 2),
            "min": round(r[5] or 0, 2),
            "max": round(r[6] or 0, 2),
            "count": r[7],
        }
        for r in result
    ]


# ============ WEBSOCKET ENDPOINT ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time trace updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============ ADMIN ENDPOINTS ============

@app.delete("/admin/reset")
def reset_all_data():
    """Reset all trace data (use with caution!)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM spans")
    conn.execute("DELETE FROM traces_summary")
    conn.commit()
    conn.close()
    
    duck = duckdb.connect(str(ANALYTICS_DB))
    duck.execute("DELETE FROM metrics")
    duck.close()
    
    # Clear blobs
    for blob_file in BLOB_DIR.glob("*"):
        blob_file.unlink()
    
    return {"status": "all_data_cleared"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "data_dir": str(DATA_DIR.absolute()),
    }

@app.get("/analytics/prompts")
def get_prompt_registry(limit: int = 100):
    """
    Returns unique prompts grouped by their preview text.
    aggregates usage count, avg latency, and cost.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # We group by input_preview as a proxy for the unique prompt
        # In a production DB, you would hash the full prompt content
        rows = conn.execute("""
            SELECT 
                input_preview,
                COUNT(*) as usage_count,
                AVG(duration_ms) as avg_latency,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                AVG(json_extract(usage_json, '$.total_cost')) as avg_cost,
                MAX(start_time) as last_used
            FROM spans 
            WHERE span_type = 'llm' 
            AND input_preview IS NOT NULL
            GROUP BY input_preview
            ORDER BY last_used DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
    return [
        {
            "prompt_text": r[0],
            "count": r[1],
            "avg_latency": round(r[2] or 0, 2),
            "error_rate": round((r[3] / r[1]) * 100, 1) if r[1] > 0 else 0,
            "avg_cost": round(r[4] or 0, 5),
            "last_used": r[5]
        }
        for r in rows
    ]


# ============ 2. FLOW GRAPH ============

@app.get("/analytics/flow-map")
def get_flow_map(trace_id: str = None):
    """
    Constructs a node-link diagram of span relationships.
    Returns ReactFlow compatible nodes and edges.
    If trace_id is provided, shows flow for that specific trace only.
    """
    with sqlite3.connect(DB_PATH) as conn:
        if trace_id:
            # Flow for a specific trace
            links = conn.execute("""
                SELECT 
                    p.name as source,
                    s.name as target,
                    s.span_type as target_type,
                    COUNT(*) as weight,
                    AVG(s.duration_ms) as avg_duration
                FROM spans s
                JOIN spans p ON s.parent_span_id = p.span_id
                WHERE s.trace_id = ?
                GROUP BY p.name, s.name
                HAVING weight > 0
            """, (trace_id,)).fetchall()
        else:
            # Aggregate view across all traces
            links = conn.execute("""
                SELECT 
                    p.name as source,
                    s.name as target,
                    s.span_type as target_type,
                    COUNT(*) as weight,
                    AVG(s.duration_ms) as avg_duration
                FROM spans s
                JOIN spans p ON s.parent_span_id = p.span_id
                GROUP BY p.name, s.name
                HAVING weight > 0
            """).fetchall()

    nodes_dict = {}
    edges = []
    
    for row in links:
        source, target, target_type, weight, duration = row
        
        # Ensure nodes exist
        if source not in nodes_dict:
            nodes_dict[source] = {"id": source, "type": "default"}
        if target not in nodes_dict:
            nodes_dict[target] = {"id": target, "type": target_type} # Use type for coloring if needed

        # Create Edge
        edges.append({
            "id": f"{source}->{target}",
            "source": source,
            "target": target,
            "label": f"{weight}",
            "animated": True,
            "style": { "strokeWidth": max(1, min(weight, 5)) }, # Thicker lines for heavy traffic
            "data": { "duration": f"{int(duration)}ms" }
        })

    # Convert nodes dict to list - include type for coloring
    nodes = [{"id": n["id"], "type": n.get("type", "default"), "data": {"label": n["id"]}, "position": {"x": 0, "y": 0}} for n in nodes_dict.values()]

    return {
        "nodes": nodes,
        "edges": edges
    }


# ============ 3. MODEL COMPARISON ARENA ============

# Available models by provider
ARENA_MODELS = {
    "openai": [
        {"id": "gpt-4o", "name": "GPT-4o", "provider": "openai"},
        {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "openai"},
        {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "provider": "openai"},
        {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "provider": "openai"},
        {"id": "o1-preview", "name": "O1 Preview", "provider": "openai"},
        {"id": "o1-mini", "name": "O1 Mini", "provider": "openai"},
    ],
    "anthropic": [
        {"id": "claude-3-5-sonnet-latest", "name": "Claude 3.5 Sonnet", "provider": "anthropic"},
        {"id": "claude-3-5-haiku-latest", "name": "Claude 3.5 Haiku", "provider": "anthropic"},
        {"id": "claude-3-opus-latest", "name": "Claude 3 Opus", "provider": "anthropic"},
    ],
}

@app.get("/arena/models")
def get_arena_models():
    """Return available models for the arena, filtered by which API keys are set."""
    available = []
    
    if os.getenv("OPENAI_API_KEY"):
        available.extend(ARENA_MODELS["openai"])
    
    if os.getenv("ANTHROPIC_API_KEY"):
        available.extend(ARENA_MODELS["anthropic"])
    
    # If no keys set, return all models but mark as unavailable
    if not available:
        for provider_models in ARENA_MODELS.values():
            for m in provider_models:
                available.append({**m, "available": False})
        return {"models": available, "warning": "No API keys configured. Add keys to server/.env"}
    
    return {"models": [{"available": True, **m} for m in available]}

class ArenaRequest(BaseModel):
    prompt: str
    model: str
    provider: Optional[str] = None
    system_prompt: Optional[str] = None

@app.post("/arena/run")
async def run_arena_comparison(req: ArenaRequest):
    """
    Proxy endpoint to run a prompt against a specific model.
    Supports OpenAI and Anthropic.
    """
    # Detect provider from model name if not specified
    provider = req.provider
    if not provider:
        if req.model.startswith("claude"):
            provider = "anthropic"
        else:
            provider = "openai"
    
    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    try:
        if provider == "anthropic":
            return await _run_anthropic(req.model, messages, req.system_prompt)
        else:
            return await _run_openai(req.model, messages)
    except Exception as e:
        return {"status": "error", "error": str(e), "model": req.model}

async def _run_openai(model: str, messages: list):
    """Run completion with OpenAI."""
    from openai import AsyncOpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "error", "error": "OPENAI_API_KEY not set. Add it to server/.env", "model": model}
    
    client = AsyncOpenAI(api_key=api_key)
    
    start = datetime.now()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    duration = (datetime.now() - start).total_seconds() * 1000
    
    return {
        "status": "success",
        "output": response.choices[0].message.content,
        "duration_ms": round(duration, 1),
        "model": model,
        "provider": "openai",
        "usage": response.usage.model_dump() if response.usage else {}
    }

async def _run_anthropic(model: str, messages: list, system_prompt: Optional[str]):
    """Run completion with Anthropic."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return {"status": "error", "error": "Anthropic not installed. Run: pip install anthropic", "model": model}
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {"status": "error", "error": "ANTHROPIC_API_KEY not set. Add it to server/.env", "model": model}
    
    client = AsyncAnthropic(api_key=api_key)
    
    # Anthropic uses different message format
    anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
    
    start = datetime.now()
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt or "You are a helpful assistant.",
        messages=anthropic_messages
    )
    duration = (datetime.now() - start).total_seconds() * 1000
    
    return {
        "status": "success",
        "output": response.content[0].text if response.content else "",
        "duration_ms": round(duration, 1),
        "model": model,
        "provider": "anthropic",
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        }
    }

# ============ MAIN ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8787)
