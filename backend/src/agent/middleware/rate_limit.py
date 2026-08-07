"""Rate-limiting middleware backed by Redis (sliding window).

If Redis is unavailable or ``RATE_LIMIT_ENABLED`` is false, the middleware
becomes a no-op.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from agent.observability import get_logger

logger = get_logger(__name__)

RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() in ("1", "true", "yes")
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))

# ---------------------------------------------------------------------------
# Redis client (lazy)
# ---------------------------------------------------------------------------

_redis_client: Optional[Any] = None  # type: ignore[name-defined]


def _get_redis() -> Optional[Any]:  # type: ignore[name-defined]
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis as _redis_lib

        uri = os.environ.get("REDIS_URI", "redis://localhost:6379")
        _redis_client = _redis_lib.from_url(uri, decode_responses=True)
        _redis_client.ping()
        logger.info("rate_limiter_redis_connected")
        return _redis_client
    except Exception as exc:
        logger.warning("rate_limiter_redis_unavailable", error=str(exc))
        _redis_client = False  # type: ignore[assignment]
        return None


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP.

    Uses a Redis sorted set (ZADD + ZREMRANGEBYSCORE) to track timestamps.
    """

    async def dispatch(self, request: Request, call_next):
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)

        redis = _get_redis()
        if redis is None:
            return await call_next(request)

        # Identify client (IP-based; in production you may prefer user_id)
        client_id = request.client.host if request.client else "unknown"
        user_id = getattr(request.state, "user_id", None)
        if user_id and user_id != "anonymous":
            client_id = user_id

        key = f"ratelimit:{client_id}"
        window = 60  # seconds
        limit = RATE_LIMIT_RPM
        now = time.time()

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window)
        _, current_count, _, _ = pipe.execute()

        if current_count >= limit:
            from fastapi.responses import JSONResponse

            logger.warning("rate_limit_exceeded", client_id=client_id, limit=limit)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: {limit} requests per minute."},
                headers={"Retry-After": str(window)},
            )

        return await call_next(request)
