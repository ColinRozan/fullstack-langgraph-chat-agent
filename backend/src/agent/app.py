# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import persistence

# Define the FastAPI app
app = FastAPI()

# Allow CORS for local development (frontend on :5173 and :2024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db_error_detail() -> str:
    """Return a user-friendly message when the DB is unreachable."""
    return "Database unavailable. Ensure POSTGRES_URI is set and PostgreSQL is running."


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
def list_threads_api() -> ThreadListResponse:
    """Return all thread metadata ordered by most recently updated."""
    try:
        rows = persistence.list_threads()
    except Exception as exc:
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
def get_thread_api(thread_id: str) -> ThreadMetaOut:
    """Return metadata for a single thread."""
    try:
        row = persistence.get_thread(thread_id)
    except Exception as exc:
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
def update_thread_api(thread_id: str, body: ThreadTitleUpdate) -> Dict[str, Any]:
    """Update thread title (upserts if missing)."""
    try:
        persistence.upsert_thread(thread_id, body.title)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"thread_id": thread_id, "title": body.title}


@app.delete("/api/threads/{thread_id}")
def delete_thread_api(thread_id: str) -> Dict[str, Any]:
    """Delete thread metadata and associated session state."""
    try:
        persistence.delete_thread(thread_id)
    except Exception as exc:
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
def get_memory_api(namespace: str, key: str) -> MemoryGetResponse:
    """Retrieve a single memory entry."""
    try:
        value = persistence.get_memory(namespace, key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return MemoryGetResponse(namespace=namespace, key=key, value=value)


@app.get("/api/memory/{namespace}", response_model=MemoryListResponse)
def list_memory_api(namespace: str) -> MemoryListResponse:
    """List all entries in a memory namespace."""
    try:
        items = persistence.list_memory_namespace(namespace)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return MemoryListResponse(namespace=namespace, items=items)


@app.post("/api/memory")
def put_memory_api(body: MemoryPutBody) -> Dict[str, Any]:
    """Store a value in long-term memory."""
    try:
        persistence.put_memory(body.namespace, body.key, body.value)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"namespace": body.namespace, "key": body.key, "stored": True}


@app.delete("/api/memory/{namespace}/{key}")
def delete_memory_api(namespace: str, key: str) -> Dict[str, Any]:
    """Delete a single memory entry."""
    try:
        persistence.delete_memory(namespace, key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=_db_error_detail()) from exc
    return {"namespace": namespace, "key": key, "deleted": True}


# ---------------------------------------------------------------------------
# Frontend static files
# ---------------------------------------------------------------------------

def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
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
