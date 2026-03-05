"""
LLM Inference Service - Main Application.

Production-grade FastAPI application providing OpenAI-compatible
inference endpoints powered by vLLM.
"""

from __future__ import annotations

import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.models import ErrorDetail, ErrorResponse
from app.api.routes import router
from app.cache.prompt_cache import get_prompt_cache
from app.cache.redis_client import get_redis_client, initialize_redis, shutdown_redis
from app.engine.vllm_engine import get_engine, initialize_engine, shutdown_engine
from app.middleware.rate_limiter import RateLimitMiddleware


# =============================================================================
# Configuration
# =============================================================================


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        self.app_name: str = os.getenv("APP_NAME", "LLM Inference Service")
        self.app_version: str = os.getenv("APP_VERSION", "0.1.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() == "true"
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.workers: int = int(os.getenv("WORKERS", "1"))

        # CORS settings
        self.cors_origins: list[str] = os.getenv(
            "CORS_ORIGINS", "*"
        ).split(",")
        self.cors_allow_credentials: bool = os.getenv(
            "CORS_ALLOW_CREDENTIALS", "true"
        ).lower() == "true"

        # Graceful shutdown
        self.shutdown_timeout: int = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))


settings = Settings()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan events.

    Handles startup (model loading, Redis connection) and shutdown (graceful cleanup).

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application.
    """
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")

    # Initialize Redis (optional, continues if unavailable)
    redis_client = await initialize_redis()
    if redis_client is not None:
        print("Redis connected")
    else:
        print("Redis unavailable (caching and distributed rate limiting disabled)")

    print("Loading model... (this may take a while)")

    try:
        await initialize_engine()
        engine = await get_engine()
        print(f"Model loaded: {engine.model_name}")
        print(f"Server ready at http://{settings.host}:{settings.port}")
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        raise

    yield

    # Shutdown
    print("Shutting down gracefully...")
    await shutdown_engine()
    await shutdown_redis()
    print("Shutdown complete")


# =============================================================================
# Application Factory
# =============================================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Production-Grade LLM Inference Service

OpenAI-compatible API powered by vLLM with:

- **High Performance**: Continuous batching with PagedAttention
- **Streaming Support**: Server-Sent Events for real-time generation
- **Production Ready**: Health checks, graceful shutdown, error handling
- **Redis Caching**: Response caching for deterministic requests
- **Rate Limiting**: Per-IP rate limiting with Redis or in-memory backend

### Endpoints

- `POST /v1/completions` - Text completion (OpenAI compatible)
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `GET /metrics` - Cache and rate limit metrics
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate limiting middleware (lazy initialization after Redis connects)
    app.add_middleware(RateLimitMiddleware)

    # Include API routes
    app.include_router(router)

    # Register health endpoints
    _register_health_endpoints(app)

    # Register exception handlers
    _register_exception_handlers(app)

    return app


def _register_health_endpoints(app: FastAPI) -> None:
    """Register health check endpoints for K8s probes.

    Args:
        app: The FastAPI application.
    """

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check",
        description="Basic health check endpoint for liveness probes.",
    )
    async def health_check() -> dict[str, str]:
        """Liveness probe endpoint.

        Returns simple OK status to indicate the service is running.
        """
        return {"status": "healthy"}

    @app.get(
        "/ready",
        tags=["Health"],
        summary="Readiness check",
        description="Readiness probe to verify model and Redis are ready.",
    )
    async def readiness_check() -> JSONResponse:
        """Readiness probe endpoint.

        Verifies that the model is loaded and Redis is connected (if enabled).
        Returns 503 if not ready.
        """
        checks: dict = {"engine": False, "redis": None}
        reasons: list[str] = []

        # Check engine
        try:
            engine = await get_engine()
            if engine.is_ready():
                checks["engine"] = True
            else:
                reasons.append("model_loading")
        except Exception as e:
            reasons.append(f"engine_error: {e}")

        # Check Redis (optional)
        redis_client = await get_redis_client()
        if redis_client is not None:
            try:
                if await redis_client.health_check():
                    checks["redis"] = True
                else:
                    checks["redis"] = False
                    reasons.append("redis_unhealthy")
            except Exception:
                checks["redis"] = False
                reasons.append("redis_error")

        # Determine overall status
        is_ready = checks["engine"] and (checks["redis"] is None or checks["redis"])

        if is_ready:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "ready",
                    "checks": checks,
                    "model": engine.model_name if checks["engine"] else None,
                },
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "not_ready",
                    "checks": checks,
                    "reasons": reasons,
                },
            )

    @app.get(
        "/metrics",
        tags=["Health"],
        summary="Service metrics",
        description="Cache statistics and rate limiter configuration.",
    )
    async def metrics() -> dict:
        """Metrics endpoint for monitoring.

        Returns cache statistics and rate limiter configuration.
        """
        # Cache stats
        cache = get_prompt_cache()
        cache_stats = cache.get_stats().to_dict()

        # Redis status
        redis_client = await get_redis_client()
        redis_connected = redis_client is not None and redis_client.is_connected()

        return {
            "cache": {
                "enabled": cache.config.enabled,
                "connected": redis_connected,
                "stats": cache_stats,
            },
            "rate_limit": {
                "enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
                "requests_per_minute": int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")),
                "window_seconds": int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
            },
        }

    @app.get(
        "/",
        tags=["Health"],
        summary="Root endpoint",
        description="Service information and links to documentation.",
    )
    async def root() -> dict:
        """Root endpoint with service information."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "openapi": "/openapi.json",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
        }


def _register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers.

    Args:
        app: The FastAPI application.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle uncaught exceptions with OpenAI-compatible error format."""
        error_response = ErrorResponse(
            error=ErrorDetail(
                message=str(exc) if settings.debug else "Internal server error",
                type="internal_error",
                code="internal_error",
            )
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(),
        )


# =============================================================================
# Application Instance & Entry Point
# =============================================================================


app = create_app()


def main() -> None:
    """Run the application with uvicorn.

    Entry point for the application when run directly or via console script.
    """
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        print(f"\nReceived signal {sig}, initiating graceful shutdown...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
