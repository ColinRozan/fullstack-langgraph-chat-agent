"""PostgreSQL persistence layer for thread state, metadata, and long-term memory.

This module provides:
- A connection pool singleton backed by ``psycopg``.
- Thread metadata CRUD (title, created_at, updated_at).
- A simple key-value store for cross-session memory.
- Optional ``PostgresSaver`` integration for LangGraph checkpointing.

All table creation is idempotent (``CREATE TABLE IF NOT EXISTS``).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

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


def get_pool(min_size: int = 1, max_size: int = 10) -> Any:
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
    return _pool


def close_pool() -> None:
    """Close the global connection pool if it exists."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


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
    state JSONB NOT NULL DEFAULT '{}',
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""


def _ensure_tables() -> None:
    """Create tables if they do not already exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_THREAD_METADATA_TABLE)
            cur.execute(_MEMORY_TABLE)
            cur.execute(_SESSION_STATE_TABLE)


# ---------------------------------------------------------------------------
# Thread metadata helpers
# ---------------------------------------------------------------------------

@dataclass
class ThreadMeta:
    thread_id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


def list_threads() -> List[ThreadMeta]:
    """Return all thread metadata ordered by most recently updated."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, title, created_at, updated_at
                FROM thread_metadata
                ORDER BY updated_at DESC
                """
            )
            rows = cur.fetchall()
            return [
                ThreadMeta(
                    thread_id=r[0],
                    title=r[1],
                    created_at=_fmt_ts(r[2]),
                    updated_at=_fmt_ts(r[3]),
                )
                for r in rows
            ]


def get_thread(thread_id: str) -> Optional[ThreadMeta]:
    """Return metadata for a single thread."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, title, created_at, updated_at
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
                title=row[1],
                created_at=_fmt_ts(row[2]),
                updated_at=_fmt_ts(row[3]),
            )


def upsert_thread(thread_id: str, title: str) -> None:
    """Insert or update thread metadata."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO thread_metadata (thread_id, title, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET title = EXCLUDED.title, updated_at = NOW()
                """,
                (thread_id, title),
            )


def delete_thread(thread_id: str) -> None:
    """Remove thread metadata and associated session state."""
    with get_connection() as conn:
        with conn.cursor() as cur:
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

def save_session_state(thread_id: str, state: Dict[str, Any]) -> None:
    """Persist a JSON-serialisable snapshot of the graph state."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO session_state (thread_id, state, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (thread_id)
                DO UPDATE SET state = EXCLUDED.state, updated_at = NOW()
                """,
                (thread_id, json.dumps(state, default=str)),
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


def list_memory_namespace(namespace: str) -> Dict[str, Any]:
    """Return all key-value pairs for a given namespace."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT key, value FROM agent_memory WHERE namespace = %s",
                (namespace,),
            )
            return {r[0]: r[1] for r in cur.fetchall()}


def delete_memory(namespace: str, key: str) -> None:
    """Delete a single memory entry."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM agent_memory WHERE namespace = %s AND key = %s",
                (namespace, key),
            )


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
        print(f"[persistence] langgraph-checkpoint-postgres not available: {exc}")
        return None

    try:
        pool = get_pool()
        saver = PostgresSaver(sync_connection=pool)
        # Ensure LangGraph checkpoint tables exist
        saver.setup()
        print("[persistence] PostgresSaver initialised successfully")
        return saver
    except Exception as exc:  # pragma: no cover
        print(f"[persistence] Failed to initialise PostgresSaver: {exc}")
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
