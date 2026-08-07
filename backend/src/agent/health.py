"""Health check endpoints for liveness, readiness, and basic status.

Three endpoints are provided:
- ``/health``  → always 200 (process is alive)
- ``/live``    → same as /health (K8s livenessProbe convention)
- ``/ready``   → 200 only when dependencies (Postgres, Chroma) are reachable
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, Response

from agent import persistence
from agent.observability import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/live")
def health_check() -> Dict[str, str]:
    """Liveness probe — returns 200 as long as the process is up."""
    return {"status": "ok"}


@router.get("/ready")
def readiness_check() -> Response:
    """Readiness probe — verifies critical dependencies.

    Checks:
        1. PostgreSQL connection pool is healthy.
        2. Chroma vector store directory exists (lightweight proxy).
    """
    checks: Dict[str, Any] = {}
    all_ok = True

    # 1. Postgres
    try:
        pool = persistence.get_pool()
        # A lightweight ping
        with persistence.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.warning("readiness_postgres_failed", error=str(exc))
        checks["postgres"] = f"unavailable: {exc}"
        all_ok = False

    # 2. Chroma (lightweight — just check dir exists)
    try:
        from agent.knowledge_base import DEFAULT_CHROMA_DIR

        if DEFAULT_CHROMA_DIR.exists():
            checks["chroma"] = "ok"
        else:
            checks["chroma"] = "not_initialized"
    except Exception as exc:
        logger.warning("readiness_chroma_failed", error=str(exc))
        checks["chroma"] = f"error: {exc}"
        all_ok = False

    if all_ok:
        return Response(
            content=json.dumps({"status": "ready", "checks": checks}),
            media_type="application/json",
            status_code=200,
        )

    return Response(
        content=json.dumps({"status": "not_ready", "checks": checks}),
        media_type="application/json",
        status_code=503,
    )
