"""Authentication middleware for FastAPI and ASGI.

Supports two mechanisms (checked in order):
1. Bearer Token  – ``Authorization: Bearer <token>``
2. API Key       – ``X-API-Key: <key>``

If neither ``API_TOKEN`` nor ``API_KEY`` environment variables are set,
authentication is skipped (development convenience).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_API_TOKEN = os.environ.get("API_TOKEN")
_API_KEY = os.environ.get("API_KEY")
_AUTH_ENABLED = bool(_API_TOKEN or _API_KEY)

# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> Optional[str]:
    """FastAPI dependency that validates the incoming auth header.

    Returns the authenticated user identifier (here simply ``"api"``) or
    ``None`` when auth is disabled.
    """
    if not _AUTH_ENABLED:
        return None

    token_valid = False

    # 1. Bearer token
    if _API_TOKEN and credentials and credentials.scheme == "Bearer":
        if credentials.credentials == _API_TOKEN:
            token_valid = True

    # 2. API Key header
    if _API_KEY and not token_valid:
        api_key = request.headers.get("X-API-Key")
        if api_key == _API_KEY:
            token_valid = True

    if not token_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return "api"


# ---------------------------------------------------------------------------
# ASGI middleware (for LangGraph API layer where dependency injection is
# not available)
# ---------------------------------------------------------------------------

class AuthMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces Bearer / API-Key authentication."""

    async def dispatch(self, request: Request, call_next):
        if not _AUTH_ENABLED:
            request.state.user_id = "anonymous"
            return await call_next(request)

        # Try Bearer
        auth_header = request.headers.get("Authorization", "")
        token_valid = False
        if _API_TOKEN and auth_header.startswith("Bearer "):
            if auth_header[7:] == _API_TOKEN:
                token_valid = True

        # Try API Key
        if _API_KEY and not token_valid:
            if request.headers.get("X-API-Key") == _API_KEY:
                token_valid = True

        if not token_valid:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing authentication credentials"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        request.state.user_id = "api"
        return await call_next(request)
