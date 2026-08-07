"""PostgreSQL persistence layer for thread state, metadata, long-term memory,
and audit logging.

This module provides:
- A connection pool singleton backed by ``psycopg``.
- Thread metadata CRUD with optional user isolation.
- A simple key-value store for cross-session memory.
- Audit logging for compliance.
- Feedback collection for answer quality.
- Optional ``PostgresSaver`` integration for LangGraph checkpointing.

All table creation is idempotent (``CREATE TABLE IF NOT EXISTS``).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from agent.observability import get_logger, DB_CONNECTIONS_ACTIVE

logger = get_logger(__name__)

try:
    import psycopg
    from psycopg_pool import ConnectionPool

    _HAS_PSYCOPG = True
except Exception as _psycopg_err:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    ConnectionPool = None  # type: ignore[assignment,misc]
    _HAS_PSYCOPG = False

# ---------------------------------------------------------------------------
# Connection pool singleton
# ---------------------------------------------------------------------------

_pool: Optional[Any] = None


def get_connection_string() -> str:
    """Return the PostgreSQL connection string from environment."""
    return os.environ.get(
        "POSTGRES_URI",
        "postgres://postgres:postgres@localhost:5432/postgres?sslmode=disable",
    )


def get_pool(min_size: int = 1, max_size: int = 20) -> Any:
    """Return the global ``ConnectionPool``, creating it on first call."""
    global _pool
    if _pool is not None:
        return _pool
    if not _HAS_PSYCOPG:
        raise RuntimeError("psycopg is not installed")
    conninfo = get_connection_string()
    _pool = ConnectionPool(
        conninfo=conninfo,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
    )
    _pool.wait(timeout=5.0)
    _ensure_tables()
    DB_CONNECTIONS_ACTIVE.set(max_size)
    logger.info("postgres_pool_created", min_size=min_size, max_size=max_size)
    return _pool


def close_pool() -> None:
    """Close the global connection pool if it exists."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        DB_CONNECTIONS_ACTIVE.set(0)
        logger.info("postgres_pool_closed")


@contextmanager
def get_connection() -> Generator[Any, None, None]:
    """Yield a connection from the pool."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------

_THREAD_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS thread_metadata (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT,
    title TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS agent_memory (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (namespace, key)
);
"""

_SESSION_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS session_state (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT,
    state JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_AUDIT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id TEXT,
    action TEXT NOT NULL,
    thread_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    request_summary TEXT,
    response_summary TEXT,
    tokens_used INT,
    cost_usd NUMERIC(10,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    thread_id TEXT,
    message_index INT,
    rating INT CHECK (rating IN (-1, 1)),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_DOCUMENT_INDEX_TABLE = """
CREATE TABLE IF NOT EXISTS document_index (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    chunk_count INT,
    total_chars INT,
    last_indexed TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""


def _ensure_tables() -> None:
    """Create tables if they do not already exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_THREAD_METADATA_TABLE)
            cur.execute(_MEMORY_TABLE)
            cur.execute(_SESSION_STATE_TABLE)
            cur.execute(_AUDIT_LOG_TABLE)
            cur.execute(_FEEDBACK_TABLE)
            cur.execute(_DOCUMENT_INDEX_TABLE)
            logger.debug("tables_ensured")


# ---------------------------------------------------------------------------
# Thread metadata helpers (with optional user isolation)
# ---------------------------------------------------------------------------

@dataclass
class ThreadMeta:
    thread_id: str
    title: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def list_threads(user_id: Optional[str] = None) -> List[ThreadMeta]:
    """Return all thread metadata ordered by most recently updated.

    If *user_id* is provided, only threads belonging to that user are returned.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT thread_id, user_id, title, created_at, updated_at
                    FROM thread_metadata
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (user_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT thread_id, user_id, title, created_at, updated_at
                    FROM thread_metadata
                    ORDER BY updated_at DESC
                    """
                )
            rows = cur.fetchall()
            return [
                ThreadMeta(
                    thread_id=r[0],
                    user_id=r[1],
                    title=r[2],
                    created_at=_fmt_ts(r[3]),
                    updated_at=_fmt_ts(r[4]),
                )
                for r in rows
            ]


def get_thread(thread_id: str, user_id: Optional[str] = None) -> Optional[ThreadMeta]:
    """Return metadata for a single thread."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    """
                    SELECT thread_id, user_id, title, created_at, updated_at
                    FROM thread_metadata
                    WHERE thread_id = %s AND user_id = %s
                    """,
                    (thread_id, user_id),
                )
            else:
                cur.execute(
                    """
                    SELECT thread_id, user_id, title, created_at, updated_at
                    FROM thread_metadata
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )
            row = cur.fetchone()
            if row is None:
                return None
            return ThreadMeta(
                thread_id=row[0],
                user_id=row[1],
                title=row[2],
                created_at=_fmt_ts(row[3]),
                updated_at=_fmt_ts(row[4]),
            )


def upsert_thread(thread_id: str, title: str, user_id: Optional[str] = None) -> None:
    """Insert or update thread metadata."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO thread_metadata (thread_id, user_id, title, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET user_id = COALESCE(EXCLUDED.user_id, thread_metadata.user_id),
                              title = EXCLUDED.title,
                              updated_at = NOW()
                """,
                (thread_id, user_id, title),
            )


def delete_thread(thread_id: str, user_id: Optional[str] = None) -> None:
    """Remove thread metadata and associated session state."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id:
                cur.execute(
                    "DELETE FROM thread_metadata WHERE thread_id = %s AND user_id = %s",
                    (thread_id, user_id),
                )
            else:
                cur.execute(
                    "DELETE FROM thread_metadata WHERE thread_id = %s",
                    (thread_id,),
                )
            cur.execute(
                "DELETE FROM session_state WHERE thread_id = %s",
                (thread_id,),
            )


# ---------------------------------------------------------------------------
# Session-state helpers (full graph snapshot)
# ---------------------------------------------------------------------------

def save_session_state(
    thread_id: str, state: Dict[str, Any], user_id: Optional[str] = None
) -> None:
    """Persist a JSON-serialisable snapshot of the graph state."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_state (thread_id, user_id, state, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET user_id = COALESCE(EXCLUDED.user_id, session_state.user_id),
                              state = EXCLUDED.state,
                              updated_at = NOW()
                """,
                (thread_id, user_id, json.dumps(state, default=str)),
            )


def load_session_state(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously saved session-state snapshot."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT state FROM session_state WHERE thread_id = %s",
                (thread_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]


# ---------------------------------------------------------------------------
# Long-term memory store (namespace → key → value)
# ---------------------------------------------------------------------------

def count_memory(namespace: str) -> int:
    """Return the number of entries in a namespace (excluding compressed keys)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM agent_memory
                WHERE namespace = %s AND key NOT LIKE %s
                """,
                (namespace, "_compressed_%"),
            )
            row = cur.fetchone()
            return row[0] if row else 0


def put_memory(namespace: str, key: str, value: Any) -> None:
    """Upsert a value into the long-term memory store."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_memory (namespace, key, value, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (namespace, key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (namespace, key, json.dumps(value, default=str)),
            )


def get_memory(namespace: str, key: str) -> Any:
    """Retrieve a value from the long-term memory store."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT value FROM agent_memory WHERE namespace = %s AND key = %s",
                (namespace, key),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return row[0]


def list_memory_namespace(
    namespace: str,
    limit: int | None = None,
    order_by: str = "created_at ASC",
) -> Dict[str, Any]:
    """Return key-value pairs for a given namespace, optionally limited and ordered.

    Args:
        namespace: The memory namespace to query.
        limit: Maximum number of entries to return. None for unlimited.
        order_by: SQL ORDER BY clause (e.g. "created_at ASC" or "created_at DESC").
    """
    # Whitelist allowed order_by values to prevent SQL injection
    allowed_orders = {
        "created_at ASC",
        "created_at DESC",
        "updated_at ASC",
        "updated_at DESC",
        "key ASC",
        "key DESC",
    }
    safe_order = order_by if order_by in allowed_orders else "created_at ASC"

    with get_connection() as conn:
        with conn.cursor() as cur:
            sql = f"""
                SELECT key, value FROM agent_memory
                WHERE namespace = %s
                ORDER BY {safe_order}
            """
            params: list = [namespace]
            if limit is not None:
                sql += " LIMIT %s"
                params.append(limit)
            cur.execute(sql, params)
            return {r[0]: r[1] for r in cur.fetchall()}


def delete_memory(namespace: str, key: str) -> None:
    """Delete a single memory entry."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_memory WHERE namespace = %s AND key = %s",
                (namespace, key),
            )


def delete_memory_batch(namespace: str, keys: List[str]) -> int:
    """Delete multiple memory entries in a single query.

    Returns:
        Number of rows deleted.
    """
    if not keys:
        return 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use ANY with array for efficient batch delete
            cur.execute(
                "DELETE FROM agent_memory WHERE namespace = %s AND key = ANY(%s)",
                (namespace, keys),
            )
            return cur.rowcount


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

def insert_audit_log(
    user_id: Optional[str],
    action: str,
    thread_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    request_summary: Optional[str] = None,
    response_summary: Optional[str] = None,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> None:
    """Write a lightweight audit row."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_log
                    (user_id, action, thread_id, ip_address, user_agent,
                     request_summary, response_summary, tokens_used, cost_usd)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        action,
                        thread_id,
                        ip_address,
                        user_agent,
                        request_summary,
                        response_summary,
                        tokens_used,
                        cost_usd,
                    ),
                )
    except Exception as exc:
        logger.warning("insert_audit_log_failed", error=str(exc))


def update_audit_log_response(
    thread_id: str,
    response_summary: Optional[str] = None,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None,
) -> None:
    """Update the most recent audit row for a thread with response metadata."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE audit_log
                    SET response_summary = COALESCE(%s, response_summary),
                        tokens_used = COALESCE(%s, tokens_used),
                        cost_usd = COALESCE(%s, cost_usd)
                    WHERE id = (
                        SELECT id FROM audit_log
                        WHERE thread_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                    """,
                    (response_summary, tokens_used, cost_usd, thread_id),
                )
    except Exception as exc:
        logger.warning("update_audit_log_response_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------

def save_feedback(
    thread_id: str,
    message_index: int,
    rating: int,
    comment: Optional[str] = None,
) -> None:
    """Save user feedback for a specific message."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (thread_id, message_index, rating, comment)
                VALUES (%s, %s, %s, %s)
                """,
                (thread_id, message_index, rating, comment),
            )


# ---------------------------------------------------------------------------
# Document index tracking
# ---------------------------------------------------------------------------

def upsert_document_index(
    filename: str,
    chunk_count: int,
    total_chars: int,
) -> None:
    """Record or update indexing metadata for a document."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO document_index (filename, chunk_count, total_chars, last_indexed)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (filename)
                DO UPDATE SET chunk_count = EXCLUDED.chunk_count,
                              total_chars = EXCLUDED.total_chars,
                              last_indexed = NOW()
                """,
                (filename, chunk_count, total_chars),
            )


def list_document_index() -> List[Dict[str, Any]]:
    """Return all tracked document index entries."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT filename, chunk_count, total_chars, last_indexed
                FROM document_index
                ORDER BY last_indexed DESC
                """
            )
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# PostgresSaver helper (LangGraph checkpointer)
# ---------------------------------------------------------------------------

def get_postgres_saver() -> Any:
    """Return a ``PostgresSaver`` instance wired to our connection pool.

    Falls back to ``None`` if ``langgraph-checkpoint-postgres`` is not
    installed or the database is unreachable.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover
        logger.warning("langgraph_checkpoint_postgres_unavailable", error=str(exc))
        return None

    try:
        pool = get_pool()
        saver = PostgresSaver(sync_connection=pool)
        # Ensure LangGraph checkpoint tables exist
        saver.setup()
        logger.info("postgres_saver_initialised")
        return saver
    except Exception as exc:  # pragma: no cover
        logger.warning("postgres_saver_initialisation_failed", error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_ts(ts: Any) -> Optional[str]:
    """Format a Postgres timestamp to ISO-8601 string."""
    if ts is None:
        return None
    # psycopg returns datetime objects
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
