"""
LLM Inference Service - Main Application.

Production-grade FastAPI application providing OpenAI-compatible
inference endpoints powered by vLLM.
"""

from __future__ import annotations

import asyncio
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
from app.engine.vllm_engine import get_engine, initialize_engine, shutdown_engine


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

    Handles startup (model loading) and shutdown (graceful cleanup).

    Args:
        app: The FastAPI application instance.

    Yields:
        Control to the application.
    """
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Loading model... (this may take a while)")

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

### Endpoints

- `POST /v1/completions` - Text completion (OpenAI compatible)
- `POST /v1/chat/completions` - Chat completion (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /ready` - Readiness probe
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
        description="Readiness probe to verify model is loaded and ready.",
    )
    async def readiness_check() -> JSONResponse:
        """Readiness probe endpoint.

        Verifies that the model is loaded and ready to serve requests.
        Returns 503 if not ready.
        """
        try:
            engine = await get_engine()
            if engine.is_ready():
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content={"status": "ready", "model": engine.model_name},
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"status": "not_ready", "reason": "model_loading"},
                )
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": str(e)},
            )

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
