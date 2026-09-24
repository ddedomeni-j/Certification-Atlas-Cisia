from __future__ import annotations

import time
import uuid

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Ajoute un identifiant de requête, mesure la latence et journalise l'appel."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.bind(request_id=request_id).exception("Erreur non gérée")
            raise

        latency_ms = round((time.perf_counter() - start) * 1_000, 2)

        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
        ).info("Requête traitée")

        response.headers["X-Request-ID"] = request_id
        return response