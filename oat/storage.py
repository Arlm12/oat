"""
Local-first storage engine for OpenAgentTrace.
Uses SQLite (WAL mode) for span metadata and file-based blob storage for payloads.
"""

import json
import sqlite3
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from datetime import datetime

from .models import Span, SpanStatus, SpanType


class StorageEngine:
    """
    Local-first storage with SQLite + blob separation.
    
    Design principles:
    - WAL mode for concurrent reads/writes
    - Blob deduplication via content hashing
    - Thread-safe connection pool
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        # Anchor to project root
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = Path(__file__).resolve().parent.parent / ".oat"
            
        self.db_path = self.data_dir / "traces.db"
        self.blob_dir = self.data_dir / "blobs"
        self._local = threading.local()
        self._init_storage()
    
    def _init_storage(self):
        """Initialize storage directories and database schema."""
        self.data_dir.mkdir(exist_ok=True)
        self.blob_dir.mkdir(exist_ok=True)
        
        with self._get_connection() as conn:
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            
            # Spans table - the core entity
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
            
            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace_id ON spans(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_type ON spans(span_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status)")
            
            # Traces summary table (materialized view for fast listing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces_summary (
                    trace_id TEXT PRIMARY KEY,
                    name TEXT,
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
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.row_factory = sqlite3.Row
        
        try:
            yield self._local.conn
        except Exception as e:
            self._local.conn.rollback()
            raise
    
    # ============ BLOB STORAGE ============
    
    def save_blob(self, data: Any) -> Optional[str]:
        """
        Save data to blob storage with content-addressed deduplication.
        Returns the SHA256 hash of the content.
        """
        if data is None:
            return None
        
        try:
            text = json.dumps(data, default=str, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(data)
        
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        blob_path = self.blob_dir / content_hash
        
        # Deduplicate: only write if not exists
        if not blob_path.exists():
            blob_path.write_text(text, encoding='utf-8')
        
        return content_hash
    
    def get_blob(self, content_hash: Optional[str]) -> Optional[Any]:
        """Retrieve data from blob storage by hash."""
        if not content_hash:
            return None
        
        blob_path = self.blob_dir / content_hash
        if not blob_path.exists():
            return None
        
        try:
            text = blob_path.read_text(encoding='utf-8')
            return json.loads(text)
        except (json.JSONDecodeError, IOError):
            return None
    
    # ============ SPAN OPERATIONS ============
    
    def save_span(self, span: Span, inputs: Any = None, outputs: Any = None):
        """Save a span to the database, optionally with input/output blobs."""
        # Save blobs if provided
        if inputs is not None:
            span.input_hash = self.save_blob(inputs)
            span.set_input(inputs)
        
        if outputs is not None:
            span.output_hash = self.save_blob(outputs)
            span.set_output(outputs)
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO spans (
                    span_id, trace_id, service_name, parent_span_id, name, span_type,
                    start_time, end_time, duration_ms,
                    status, error_message, error_type,
                    input_hash, output_hash, input_preview, output_preview,
                    model, usage_json,
                    tool_name, tool_parameters_json,
                    retrieval_query, retrieval_k, retrieval_scores_json,
                    guardrail_triggered, guardrail_action, media_inputs_json, media_outputs_json,
                    metadata_json, tags_json,
                    score, feedback,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                span.span_id,
                span.trace_id,
                span.service_name,
                span.parent_span_id,
                span.name,
                span.span_type.value if isinstance(span.span_type, SpanType) else span.span_type,
                span.start_time,
                span.end_time,
                span.duration_ms,
                span.status.value if isinstance(span.status, SpanStatus) else span.status,
                span.error_message,
                span.error_type,
                span.input_hash,
                span.output_hash,
                span.input_preview,
                span.output_preview,
                span.model,
                json.dumps(span.usage.to_dict()) if span.usage else None,
                span.tool_name,
                json.dumps(span.tool_parameters) if span.tool_parameters else None,
                span.retrieval_query,
                span.retrieval_k,
                json.dumps(span.retrieval_scores) if span.retrieval_scores else None,
                1 if span.guardrail_triggered else (0 if span.guardrail_triggered is False else None),
                span.guardrail_action,
                json.dumps(span.media_inputs) if span.media_inputs else None,
                json.dumps(span.media_outputs) if span.media_outputs else None,
                json.dumps(span.metadata) if span.metadata else None,
                json.dumps(span.tags) if span.tags else None,
                span.score,
                span.feedback,
                datetime.now().timestamp(),
            ))
            conn.commit()
            
            # Update trace summary
            self._update_trace_summary(conn, span.trace_id)
    
    def _update_trace_summary(self, conn: sqlite3.Connection, trace_id: str):
        """Update the materialized trace summary."""
        conn.execute("""
            INSERT OR REPLACE INTO traces_summary (
                trace_id, name, start_time, end_time, duration_ms, status,
                span_count, llm_calls, tool_calls, total_tokens, total_cost, error_count,
                updated_at
            )
            SELECT 
                trace_id,
                (SELECT name FROM spans WHERE trace_id = ? AND parent_span_id IS NULL LIMIT 1),
                MIN(start_time),
                MAX(end_time),
                (MAX(end_time) - MIN(start_time)) * 1000,
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
        """, (trace_id, trace_id))
        conn.commit()
    
    def get_span(self, span_id: str, include_blobs: bool = False) -> Optional[Dict]:
        """Retrieve a single span by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM spans WHERE span_id = ?", (span_id,)
            ).fetchone()
            
            if not row:
                return None
            
            span_dict = self._row_to_span_dict(dict(row))
            
            if include_blobs:
                span_dict['inputs'] = self.get_blob(span_dict.get('input_hash'))
                span_dict['outputs'] = self.get_blob(span_dict.get('output_hash'))
            
            return span_dict
    
    def get_trace_spans(self, trace_id: str, include_blobs: bool = False) -> List[Dict]:
        """Get all spans for a trace."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC",
                (trace_id,)
            ).fetchall()
            
            spans = []
            for row in rows:
                span_dict = self._row_to_span_dict(dict(row))
                if include_blobs:
                    span_dict['inputs'] = self.get_blob(span_dict.get('input_hash'))
                    span_dict['outputs'] = self.get_blob(span_dict.get('output_hash'))
                spans.append(span_dict)
            
            return spans
    
    def list_traces(self, limit: int = 50, offset: int = 0, status: Optional[str] = None) -> List[Dict]:
        """List trace summaries with pagination."""
        with self._get_connection() as conn:
            query = "SELECT * FROM traces_summary"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
            
            query += " ORDER BY start_time DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def search_traces(self, query: str, limit: int = 50) -> List[Dict]:
        """Full-text search across traces."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT t.* FROM traces_summary t
                JOIN spans s ON t.trace_id = s.trace_id
                WHERE s.name LIKE ? 
                   OR s.input_preview LIKE ? 
                   OR s.output_preview LIKE ?
                   OR s.error_message LIKE ?
                ORDER BY t.start_time DESC
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()
            
            return [dict(row) for row in rows]
    
    def delete_trace(self, trace_id: str):
        """Delete a trace and all its spans."""
        with self._get_connection() as conn:
            # Get blob hashes to potentially clean up
            hashes = conn.execute(
                "SELECT input_hash, output_hash FROM spans WHERE trace_id = ?",
                (trace_id,)
            ).fetchall()
            
            # Delete spans
            conn.execute("DELETE FROM spans WHERE trace_id = ?", (trace_id,))
            conn.execute("DELETE FROM traces_summary WHERE trace_id = ?", (trace_id,))
            conn.commit()
            
            # Note: We don't delete blobs as they may be referenced by other spans
            # A separate garbage collection process could clean orphaned blobs
    
    def score_span(self, span_id: str, score: int, feedback: Optional[str] = None):
        """Add user feedback/score to a span."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE spans SET score = ?, feedback = ?, updated_at = ? WHERE span_id = ?",
                (score, feedback, datetime.now().timestamp(), span_id)
            )
            conn.commit()
    
    def _row_to_span_dict(self, row: Dict) -> Dict:
        """Convert database row to span dictionary."""
        # Parse JSON fields
        if row.get('usage_json'):
            row['usage'] = json.loads(row['usage_json'])
        else:
            row['usage'] = None
        del row['usage_json']
        
        if row.get('tool_parameters_json'):
            row['tool_parameters'] = json.loads(row['tool_parameters_json'])
        else:
            row['tool_parameters'] = None
        del row['tool_parameters_json']
        
        if row.get('retrieval_scores_json'):
            row['retrieval_scores'] = json.loads(row['retrieval_scores_json'])
        else:
            row['retrieval_scores'] = None
        del row['retrieval_scores_json']
        
        if row.get('metadata_json'):
            row['metadata'] = json.loads(row['metadata_json'])
        else:
            row['metadata'] = {}
        del row['metadata_json']
        
        if row.get('tags_json'):
            row['tags'] = json.loads(row['tags_json'])
        else:
            row['tags'] = []
        del row['tags_json']
        
        # Convert boolean
        if row.get('guardrail_triggered') is not None:
            row['guardrail_triggered'] = bool(row['guardrail_triggered'])
        
        return row
    
    # ============ ANALYTICS ============
    
    def get_stats(self, hours: int = 24, service_name: Optional[str] = None) -> Dict:
        """Get aggregate statistics, optionally filtered by agent."""
        with self._get_connection() as conn:
            cutoff = datetime.now().timestamp() - (hours * 3600)
            
            # Base query
            query = """
                SELECT 
                    COUNT(DISTINCT trace_id) as trace_count,
                    COUNT(*) as span_count,
                    SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
                    AVG(duration_ms) as avg_duration_ms,
                    SUM(CASE WHEN span_type = 'llm' THEN 1 ELSE 0 END) as llm_calls,
                    SUM(CASE WHEN span_type = 'tool' THEN 1 ELSE 0 END) as tool_calls,
                    COALESCE(SUM(json_extract(usage_json, '$.total_tokens')), 0) as total_tokens,
                    COALESCE(SUM(json_extract(usage_json, '$.total_cost')), 0.0) as total_cost
                FROM spans
                WHERE start_time > ?
            """
            params = [cutoff]
            
            # Apply Filter
            if service_name:
                query += " AND service_name = ?"
                params.append(service_name)
                
            stats = conn.execute(query, params).fetchone()
            
            return dict(stats) if stats else {}
    
    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# Global storage instance
_storage: Optional[StorageEngine] = None


def get_storage(data_dir: Optional[Path] = None) -> StorageEngine:
    """Get or create the global storage engine."""
    global _storage
    if _storage is None:
        _storage = StorageEngine(data_dir)
    return _storage
