"""
Correlation ID Middleware.

Generates and propagates correlation IDs for request tracing
across the entire request lifecycle.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.observability.logging import (
    get_correlation_id,
    log_request,
    set_correlation_id,
)


# Header names for correlation ID
CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


def generate_correlation_id() -> str:
    """Generate a unique correlation ID.

    Returns:
        A UUID-based correlation ID.
    """
    return str(uuid.uuid4())


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request.

    Args:
        request: The incoming request.

    Returns:
        Client IP address.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation ID generation and propagation.

    - Extracts correlation ID from incoming request headers
    - Generates a new correlation ID if not present
    - Sets the correlation ID in context for logging
    - Adds correlation ID to response headers
    - Logs request completion with timing
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with correlation ID tracking.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in chain.

        Returns:
            Response with correlation ID headers.
        """
        # Get or generate correlation ID
        correlation_id = (
            request.headers.get(CORRELATION_ID_HEADER)
            or request.headers.get(REQUEST_ID_HEADER)
            or generate_correlation_id()
        )

        # Set in context for logging
        set_correlation_id(correlation_id)

        # Track request timing
        start_time = time.perf_counter()

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Add correlation ID to response headers
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[REQUEST_ID_HEADER] = correlation_id

            # Log request (skip health checks to reduce noise)
            if not request.url.path.startswith("/health"):
                log_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=response.status_code,
                    duration_ms=duration_ms,
                    client_ip=_get_client_ip(request),
                    query_string=str(request.query_params) if request.query_params else None,
                )

            return response

        except Exception as e:
            # Calculate duration even on error
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log error
            log_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_ms=duration_ms,
                client_ip=_get_client_ip(request),
                error=str(e),
                error_type=type(e).__name__,
            )

            raise

        finally:
            # Clear correlation ID from context
            set_correlation_id(None)
