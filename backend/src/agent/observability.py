"""Observability stack: structured JSON logging, Prometheus metrics, and trace IDs.

This module centralises every production-oriented observability primitive so
that the rest of the codebase only calls ``get_logger(__name__)`` and the
occasional metric increment.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Optional

import structlog
from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.environ.get("LOG_FORMAT", "json").lower()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

AGENT_REQUESTS_TOTAL = Counter(
    "agent_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

AGENT_REQUEST_DURATION = Histogram(
    "agent_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["model", "stage", "token_type"],  # token_type: prompt | completion
)

LLM_COST_USD_TOTAL = Counter(
    "llm_cost_usd_total",
    "Total estimated LLM cost in USD",
    ["model", "stage"],
)

SEARCH_REQUESTS_TOTAL = Counter(
    "search_requests_total",
    "Total web search requests",
    ["provider", "status"],
)

RAG_RETRIEVAL_DURATION = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

RESEARCH_LOOP_COUNT = Histogram(
    "research_loop_count",
    "Number of research reflection loops executed",
    buckets=[1, 2, 3, 5, 10, 15],
)

DB_CONNECTIONS_ACTIVE = Gauge(
    "db_connections_active",
    "Number of active DB connections",
)

# ---------------------------------------------------------------------------
# Structlog setup
# ---------------------------------------------------------------------------


def _setup_structlog() -> None:
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    if _LOG_FORMAT == "json":
        shared_processors.append(structlog.processors.format_exc_info)
        formatter = structlog.processors.JSONRenderer()
    else:
        formatter = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=shared_processors + [formatter],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, _LOG_LEVEL, logging.INFO)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_setup_structlog()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structured logger for *name*."""
    return structlog.get_logger(name)


def get_trace_id() -> str:
    """Return the current trace ID from structlog context, or a new UUID."""
    ctx = structlog.contextvars.get_contextvars()
    return ctx.get("trace_id", str(uuid.uuid4()))


def set_trace_id(trace_id: Optional[str] = None) -> str:
    """Set (or generate) the trace ID in structlog context."""
    tid = trace_id or str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(trace_id=tid)
    return tid


def clear_trace_id() -> None:
    """Remove the trace ID from structlog context."""
    structlog.contextvars.unbind_contextvars("trace_id")


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------

class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Inject / propagate ``X-Request-ID`` and bind it to structlog context."""

    async def dispatch(self, request: Request, call_next):
        header_tid = request.headers.get("X-Request-ID")
        tid = set_trace_id(header_tid)
        request.state.trace_id = tid

        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = tid
            return response
        finally:
            clear_trace_id()


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request counts and latencies for Prometheus."""

    async def dispatch(self, request: Request, call_next):
        from time import perf_counter

        start = perf_counter()
        method = request.method
        # Use the route path if available, otherwise the raw path
        endpoint = getattr(request.scope.get("route"), "path", request.url.path)

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception:
            status = "500"
            raise
        finally:
            duration = perf_counter() - start
            AGENT_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
            AGENT_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        return response


# ---------------------------------------------------------------------------
# Metrics endpoint helpers
# ---------------------------------------------------------------------------


def metrics_response() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
