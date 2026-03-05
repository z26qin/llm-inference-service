"""
Rate Limiter Middleware.

Provides per-IP rate limiting with both in-memory (single instance)
and Redis-based (distributed) implementations using sliding window.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.cache.redis_client import get_redis_client


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    enabled: bool
    requests_per_minute: int
    window_seconds: int
    use_redis: bool

    @classmethod
    def from_env(cls) -> RateLimitConfig:
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_minute=int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            use_redis=os.getenv("RATE_LIMIT_USE_REDIS", "true").lower() == "true",
        )


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_at: int
    retry_after: int | None = None


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def check(self, key: str) -> RateLimitResult:
        """Check if a request is allowed.

        Args:
            key: Unique identifier for the client (e.g., IP address).

        Returns:
            RateLimitResult indicating if request is allowed.
        """
        pass

    @abstractmethod
    def get_config(self) -> RateLimitConfig:
        """Get the rate limiter configuration."""
        pass


class InMemoryRateLimiter(RateLimiter):
    """In-memory sliding window rate limiter.

    Suitable for single-instance deployments.
    Uses a simple sliding window algorithm with timestamp tracking.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize the in-memory rate limiter.

        Args:
            config: Rate limit configuration. If None, loads from environment.
        """
        self.config = config or RateLimitConfig.from_env()
        # Store timestamps of requests per key
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def check(self, key: str) -> RateLimitResult:
        """Check if a request is allowed using sliding window.

        Args:
            key: Unique identifier for the client.

        Returns:
            RateLimitResult indicating if request is allowed.
        """
        now = time.time()
        window_start = now - self.config.window_seconds

        # Clean old requests outside window
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > window_start
        ]

        current_count = len(self._requests[key])
        reset_at = int(now + self.config.window_seconds)

        if current_count >= self.config.requests_per_minute:
            # Find when the oldest request in window will expire
            oldest_in_window = min(self._requests[key]) if self._requests[key] else now
            retry_after = int(oldest_in_window + self.config.window_seconds - now) + 1

            return RateLimitResult(
                allowed=False,
                limit=self.config.requests_per_minute,
                remaining=0,
                reset_at=reset_at,
                retry_after=max(1, retry_after),
            )

        # Record this request
        self._requests[key].append(now)

        return RateLimitResult(
            allowed=True,
            limit=self.config.requests_per_minute,
            remaining=self.config.requests_per_minute - current_count - 1,
            reset_at=reset_at,
        )

    def get_config(self) -> RateLimitConfig:
        """Get the rate limiter configuration."""
        return self.config


class RedisRateLimiter(RateLimiter):
    """Redis-based sliding window rate limiter.

    Suitable for distributed deployments.
    Uses Redis sorted sets for efficient sliding window implementation.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize the Redis rate limiter.

        Args:
            config: Rate limit configuration. If None, loads from environment.
        """
        self.config = config or RateLimitConfig.from_env()
        self._key_prefix = "ratelimit:"

    async def check(self, key: str) -> RateLimitResult:
        """Check if a request is allowed using Redis sorted sets.

        Args:
            key: Unique identifier for the client.

        Returns:
            RateLimitResult indicating if request is allowed.
        """
        redis_client = await get_redis_client()
        if redis_client is None or not redis_client.is_connected():
            # Fallback: allow request if Redis unavailable
            return RateLimitResult(
                allowed=True,
                limit=self.config.requests_per_minute,
                remaining=self.config.requests_per_minute,
                reset_at=int(time.time() + self.config.window_seconds),
            )

        now = time.time()
        window_start = now - self.config.window_seconds
        redis_key = f"{self._key_prefix}{key}"

        try:
            # Remove expired entries
            await redis_client.zremrangebyscore(redis_key, 0, window_start)

            # Count current requests in window
            current_count = await redis_client.zcount(redis_key, window_start, now)

            reset_at = int(now + self.config.window_seconds)

            if current_count >= self.config.requests_per_minute:
                return RateLimitResult(
                    allowed=False,
                    limit=self.config.requests_per_minute,
                    remaining=0,
                    reset_at=reset_at,
                    retry_after=self.config.window_seconds,
                )

            # Add current request with timestamp as score
            await redis_client.zadd(redis_key, {str(now): now})

            # Set key expiration
            await redis_client.expire(redis_key, self.config.window_seconds * 2)

            return RateLimitResult(
                allowed=True,
                limit=self.config.requests_per_minute,
                remaining=self.config.requests_per_minute - current_count - 1,
                reset_at=reset_at,
            )
        except Exception:
            # Fallback: allow request if Redis operation fails
            return RateLimitResult(
                allowed=True,
                limit=self.config.requests_per_minute,
                remaining=self.config.requests_per_minute,
                reset_at=int(time.time() + self.config.window_seconds),
            )

    def get_config(self) -> RateLimitConfig:
        """Get the rate limiter configuration."""
        return self.config


# Paths to skip rate limiting
SKIP_PATHS = {
    "/health",
    "/ready",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/",
    "/metrics",
}


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request.

    Handles X-Forwarded-For header for proxied requests.

    Args:
        request: The incoming request.

    Returns:
        Client IP address.
    """
    # Check X-Forwarded-For header (set by proxies/load balancers)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        # Take the first IP in the chain (original client)
        return forwarded_for.split(",")[0].strip()

    # Fall back to direct client IP
    if request.client:
        return request.client.host

    return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting.

    Applies per-IP rate limiting to all requests except
    health checks and documentation endpoints.

    Lazily initializes the rate limiter on first request,
    allowing Redis to be connected before the limiter is created.
    """

    def __init__(self, app: object) -> None:
        """Initialize the middleware.

        Args:
            app: The FastAPI application.
        """
        super().__init__(app)  # type: ignore
        self._rate_limiter: RateLimiter | None = None
        self._config = RateLimitConfig.from_env()

    async def _get_rate_limiter(self) -> RateLimiter | None:
        """Get or create the rate limiter.

        Returns:
            The rate limiter instance, or None if disabled.
        """
        if not self._config.enabled:
            return None

        if self._rate_limiter is None:
            self._rate_limiter = await get_rate_limiter()

        return self._rate_limiter

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process a request through the rate limiter.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler in chain.

        Returns:
            Response with rate limit headers.
        """
        # Skip rate limiting for certain paths
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        # Get rate limiter (lazy initialization)
        rate_limiter = await self._get_rate_limiter()
        if rate_limiter is None:
            # Rate limiting disabled
            return await call_next(request)

        # Get client identifier
        client_ip = _get_client_ip(request)

        # Check rate limit
        result = await rate_limiter.check(client_ip)

        # Add rate limit headers to all responses
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_at),
        }

        if not result.allowed:
            headers["Retry-After"] = str(result.retry_after)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded. Please retry after the specified time.",
                        "type": "rate_limit_error",
                        "code": "rate_limit_exceeded",
                    }
                },
                headers=headers,
            )

        # Process request
        response = await call_next(request)

        # Add headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


async def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance.

    Automatically selects Redis or in-memory based on configuration
    and Redis availability.

    Returns:
        The rate limiter instance.
    """
    global _rate_limiter

    if _rate_limiter is not None:
        return _rate_limiter

    config = RateLimitConfig.from_env()

    if config.use_redis:
        redis_client = await get_redis_client()
        if redis_client is not None and redis_client.is_connected():
            _rate_limiter = RedisRateLimiter(config)
        else:
            print("Warning: Redis unavailable, using in-memory rate limiter")
            _rate_limiter = InMemoryRateLimiter(config)
    else:
        _rate_limiter = InMemoryRateLimiter(config)

    return _rate_limiter
