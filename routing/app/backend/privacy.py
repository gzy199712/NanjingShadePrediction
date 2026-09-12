"""Privacy controls: memory-only requests and coordinate-free logs."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response


LOGGER = logging.getLogger("thermal_route_app")


async def privacy_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception(
            "request_failed request_id=%s path=%s method=%s",
            request_id,
            request.url.path,
            request.method,
        )
        raise
    elapsed = time.perf_counter() - started
    LOGGER.info(
        "request_complete request_id=%s path=%s method=%s status=%s elapsed_ms=%.1f",
        request_id,
        request.url.path,
        request.method,
        response.status_code,
        elapsed * 1000,
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


def privacy_status() -> dict[str, object]:
    return {
        "mode": "strict_local_memory_only",
        "exact_location_persisted": False,
        "request_body_logged": False,
        "search_history_logged": False,
        "external_requests_enabled": False,
    }

