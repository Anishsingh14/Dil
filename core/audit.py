import time
import uuid
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from db.session import async_session_maker
from db.models import AuditLog
from core.config import settings


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all inference requests to audit log."""

    def __init__(self, app, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or ["/healthz", "/docs", "/openapi.json", "/redoc"]

    async def dispatch(self, request: Request, call_next):
        # Skip audit logging for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Extract API key prefix (first 12 chars)
        api_key = request.headers.get("X-API-Key", "")
        api_key_prefix = api_key[:12] if api_key else "anonymous"

        # Get client IP
        client_ip = request.client.host if request.client else None

        # Get user agent
        user_agent = request.headers.get("User-Agent")

        # Start timing
        start_time = time.perf_counter()

        # Process request
        response: StarletteResponse = await call_next(request)

        # Calculate latency
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract error code from response if error
        error_code = None
        if response.status_code >= 400:
            try:
                # Try to get error code from response body
                if hasattr(response, 'body'):
                    import json
                    body = json.loads(response.body.decode())
                    error_code = body.get("error_code")
            except Exception:
                pass

        # Log to database (async, non-blocking)
        await self._log_audit(
            request_id=request_id,
            api_key_prefix=api_key_prefix,
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
            client_ip=client_ip,
            user_agent=user_agent,
            error_code=error_code,
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response

    async def _log_audit(
        self,
        request_id: str,
        api_key_prefix: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: int,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_code: Optional[str] = None,
    ):
        """Log audit entry to database asynchronously."""
        try:
            from db.session import async_session_maker
            from db.models import AuditLog
            
            async with async_session_maker() as session:
                audit_log = AuditLog(
                    request_id=request_id,
                    api_key_prefix=api_key_prefix,
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    latency_ms=latency_ms,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    error_code=error_code,
                )
                session.add(audit_log)
                await session.commit()
        except Exception:
            # Silently fail - audit logging should never break the request
            pass


def get_audit_middleware():
    """Factory function to create audit middleware with settings."""
    return AuditLoggingMiddleware