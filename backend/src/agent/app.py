"""FastAPI application with production middleware, health checks, and static files."""

# mypy: disable-error-code = "no-untyped-def,misc"
import os
import pathlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import persistence
from agent.health import router as health_router
from agent.middleware.auth import AuthMiddleware, verify_token
from agent.middleware.audit import AuditMiddleware
from agent.middleware.rate_limit import RateLimitMiddleware
from agent.observability import (
    CorrelationIdMiddleware,
    PrometheusMiddleware,
    metrics_response,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI()

# Observability middleware (must be first to capture everything)
app.add_middleware(CorrelationIdMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(AuditMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS — tightened for production
_cors_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:8123",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Health & metrics
# ---------------------------------------------------------------------------

app.include_router(health_router)


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics scrape endpoint."""
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_error_detail() -> str:
    return "Database unavailable. Ensure POSTGRES_URI is set and PostgreSQL is running."


def _current_user_id(request: Request) -> Optional[str]:
    return getattr(request.state, "user_id", None)


# ---------------------------------------------------------------------------
# Thread metadata API (replaces frontend localStorage)
# ---------------------------------------------------------------------------


class ThreadMetaOut(BaseModel):
    id: str
    title: str
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


class ThreadListResponse(BaseModel):
    threads: List[ThreadMetaOut]


@app.get("/api/threads", response_model=ThreadListResponse)
def list_threads_api(
    request: Request,
    _user: Optional[str] = Depends(verify_token),
) -> ThreadListResponse:
    """Return all thread metadata ordered by most recently updated."""
    user_id = _current_user_id(request)
    try:
        rows = persistence.list_threads(user_id=user_id)
    except Exception as exc:
        logger.warning("list_threads_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return ThreadListResponse(
        threads=[
            ThreadMetaOut(
                id=r.thread_id,
                title=r.title,
                createdAt=r.created_at,
                updatedAt=r.updated_at,
            )
            for r in rows
        ]
    )


@app.get("/api/threads/{thread_id}", response_model=ThreadMetaOut)
def get_thread_api(
    thread_id: str,
    request: Request,
    _user: Optional[str] = Depends(verify_token),
) -> ThreadMetaOut:
    """Return metadata for a single thread."""
    user_id = _current_user_id(request)
    try:
        row = persistence.get_thread(thread_id, user_id=user_id)
    except Exception as exc:
        logger.warning("get_thread_failed", thread_id=thread_id, error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    if row is None:
        return ThreadMetaOut(id=thread_id, title="新对话")
    return ThreadMetaOut(
        id=row.thread_id,
        title=row.title,
        createdAt=row.created_at,
        updatedAt=row.updated_at,
    )


class ThreadTitleUpdate(BaseModel):
    title: str


@app.put("/api/threads/{thread_id}")
def update_thread_api(
    thread_id: str,
    body: ThreadTitleUpdate,
    request: Request,
    _user: Optional[str] = Depends(verify_token),
) -> Dict[str, Any]:
    """Update thread title (upserts if missing)."""
    user_id = _current_user_id(request)
    try:
        persistence.upsert_thread(thread_id, body.title, user_id=user_id)
    except Exception as exc:
        logger.warning("update_thread_failed", thread_id=thread_id, error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"thread_id": thread_id, "title": body.title}


@app.delete("/api/threads/{thread_id}")
def delete_thread_api(
    thread_id: str,
    request: Request,
    _user: Optional[str] = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete thread metadata and associated session state."""
    user_id = _current_user_id(request)
    try:
        persistence.delete_thread(thread_id, user_id=user_id)
    except Exception as exc:
        logger.warning("delete_thread_failed", thread_id=thread_id, error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"thread_id": thread_id, "deleted": True}


# ---------------------------------------------------------------------------
# Memory API (cross-session long-term memory)
# ---------------------------------------------------------------------------


class MemoryPutBody(BaseModel):
    namespace: str
    key: str
    value: Any


class MemoryGetResponse(BaseModel):
    namespace: str
    key: str
    value: Optional[Any] = None


class MemoryListResponse(BaseModel):
    namespace: str
    items: Dict[str, Any]


@app.get("/api/memory/{namespace}/{key}", response_model=MemoryGetResponse)
def get_memory_api(
    namespace: str,
    key: str,
    _user: Optional[str] = Depends(verify_token),
) -> MemoryGetResponse:
    """Retrieve a single memory entry."""
    try:
        value = persistence.get_memory(namespace, key)
    except Exception as exc:
        logger.warning("get_memory_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return MemoryGetResponse(namespace=namespace, key=key, value=value)


@app.get("/api/memory/{namespace}", response_model=MemoryListResponse)
def list_memory_api(
    namespace: str,
    _user: Optional[str] = Depends(verify_token),
) -> MemoryListResponse:
    """List all entries in a memory namespace."""
    try:
        items = persistence.list_memory_namespace(namespace)
    except Exception as exc:
        logger.warning("list_memory_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return MemoryListResponse(namespace=namespace, items=items)


@app.post("/api/memory")
def put_memory_api(
    body: MemoryPutBody,
    _user: Optional[str] = Depends(verify_token),
) -> Dict[str, Any]:
    """Store a value in long-term memory."""
    try:
        persistence.put_memory(body.namespace, body.key, body.value)
    except Exception as exc:
        logger.warning("put_memory_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"namespace": body.namespace, "key": body.key, "stored": True}


@app.delete("/api/memory/{namespace}/{key}")
def delete_memory_api(
    namespace: str,
    key: str,
    _user: Optional[str] = Depends(verify_token),
) -> Dict[str, Any]:
    """Delete a single memory entry."""
    try:
        persistence.delete_memory(namespace, key)
    except Exception as exc:
        logger.warning("delete_memory_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"namespace": namespace, "key": key, "deleted": True}


# ---------------------------------------------------------------------------
# Feedback API
# ---------------------------------------------------------------------------


class FeedbackBody(BaseModel):
    thread_id: str
    message_index: int
    rating: int  # -1 or 1
    comment: Optional[str] = None


@app.post("/api/feedback")
def submit_feedback(
    body: FeedbackBody,
    _user: Optional[str] = Depends(verify_token),
) -> Dict[str, Any]:
    """Save user feedback for a specific message."""
    if body.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="Rating must be -1 or 1")
    try:
        persistence.save_feedback(
            thread_id=body.thread_id,
            message_index=body.message_index,
            rating=body.rating,
            comment=body.comment,
        )
    except Exception as exc:
        logger.warning("save_feedback_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"saved": True}


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------


def create_frontend_router(build_dir: str = "../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        logger.warning(
            "frontend_build_not_found",
            build_path=str(build_path),
        )
        from starlette.routing import Route

        async def dummy_frontend(request: Request) -> Response:
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
