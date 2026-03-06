"""
Middleware module for request handling.

Provides rate limiting, timeout handling, correlation ID tracking,
and other request processing utilities.
"""

from app.middleware.correlation import (
    CORRELATION_ID_HEADER,
    CorrelationIdMiddleware,
    generate_correlation_id,
)
from app.middleware.rate_limiter import (
    RateLimitMiddleware,
    RateLimiter,
    get_rate_limiter,
)
from app.middleware.timeout import TimeoutError, with_timeout

__all__ = [
    "CORRELATION_ID_HEADER",
    "CorrelationIdMiddleware",
    "generate_correlation_id",
    "RateLimitMiddleware",
    "RateLimiter",
    "get_rate_limiter",
    "TimeoutError",
    "with_timeout",
]
