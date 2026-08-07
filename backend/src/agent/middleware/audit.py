"""Audit-log middleware.

Records every incoming request to the ``audit_log`` table.  Because we are
inside an ASGI middleware, the response body is not easily accessible; we
therefore log metadata (user, action, IP, UA) and let the application layer
fill in ``response_summary``, ``tokens_used``, and ``cost_usd`` after the
LLM run completes.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from agent import persistence
from agent.observability import get_logger

logger = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """Write a lightweight audit row for every API request."""

    async def dispatch(self, request: Request, call_next):
        # Skip non-API and health/metrics paths
        path = request.url.path
        if not path.startswith("/api") or path in ("/api/threads", "/api/memory"):
            # We still pass through, just don't audit-list every list request
            pass

        # Extract metadata
        user_id = getattr(request.state, "user_id", None)
        ip = request.client.host if request.client else None
        ua = request.headers.get("User-Agent")
        method = request.method

        # Determine action heuristically from path + method
        action = _infer_action(method, path)

        # Extract thread_id from path if present
        thread_id: Optional[str] = None
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[1] == "threads":
            thread_id = parts[2]

        # Fire-and-forget audit write (don't block the request)
        try:
            persistence.insert_audit_log(
                user_id=user_id,
                action=action,
                thread_id=thread_id,
                ip_address=ip,
                user_agent=ua,
            )
        except Exception as exc:
            logger.warning("audit_log_write_failed", error=str(exc))

        return await call_next(request)


def _infer_action(method: str, path: str) -> str:
    """Map HTTP method + path to a coarse-grained action name."""
    if path.startswith("/api/threads"):
        if method == "POST" or method == "PUT":
            return "update_thread"
        if method == "DELETE":
            return "delete_thread"
        return "list_threads"
    if path.startswith("/api/memory"):
        return "memory_access"
    if path.startswith("/api/feedback"):
        return "submit_feedback"
    return "api_request"
