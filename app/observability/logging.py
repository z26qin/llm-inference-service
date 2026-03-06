"""
Structured Logging with Correlation IDs.

Provides JSON-formatted structured logging with automatic
correlation ID propagation for request tracing.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

import structlog


# Context variable for correlation ID
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID.

    Returns:
        The correlation ID for the current context, or None.
    """
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set.
    """
    _correlation_id.set(correlation_id)


def add_correlation_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor to add correlation ID to log events.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary with correlation ID.
    """
    correlation_id = get_correlation_id()
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Structlog processor to add service information.

    Args:
        logger: The logger instance.
        method_name: The logging method name.
        event_dict: The event dictionary.

    Returns:
        Updated event dictionary with service info.
    """
    event_dict["service"] = "llm-inference-service"
    event_dict["version"] = os.getenv("APP_VERSION", "0.1.0")
    return event_dict


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str | None = None,
) -> None:
    """Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Whether to output JSON format (True) or console format (False).
        log_file: Optional file path to write logs to.
    """
    # Determine log level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Common processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        add_service_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Add format-specific processors
    if json_format:
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    # Set format based on mode
    if json_format:
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(file_handler)

    # Configure uvicorn loggers to use structured logging
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = []
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.setLevel(log_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name. Uses caller's module name if None.

    Returns:
        A configured structlog logger.
    """
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding temporary logging context."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with context variables.

        Args:
            **kwargs: Key-value pairs to add to log context.
        """
        self.context = kwargs
        self._token: Any = None

    def __enter__(self) -> "LogContext":
        """Enter the context and bind variables."""
        structlog.contextvars.bind_contextvars(**self.context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and unbind variables."""
        structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    client_ip: str | None = None,
    **extra: Any,
) -> None:
    """Log an HTTP request with standard fields.

    Args:
        method: HTTP method.
        path: Request path.
        status_code: Response status code.
        duration_ms: Request duration in milliseconds.
        client_ip: Client IP address.
        **extra: Additional fields to log.
    """
    logger = get_logger("http")

    log_data = {
        "http_method": method,
        "http_path": path,
        "http_status": status_code,
        "duration_ms": round(duration_ms, 2),
    }

    if client_ip:
        log_data["client_ip"] = client_ip

    log_data.update(extra)

    if status_code >= 500:
        logger.error("HTTP request failed", **log_data)
    elif status_code >= 400:
        logger.warning("HTTP request client error", **log_data)
    else:
        logger.info("HTTP request completed", **log_data)


def log_generation(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_ms: float,
    stream: bool = False,
    cached: bool = False,
    **extra: Any,
) -> None:
    """Log an LLM generation event.

    Args:
        model: Model name.
        prompt_tokens: Number of prompt tokens.
        completion_tokens: Number of generated tokens.
        duration_ms: Generation duration in milliseconds.
        stream: Whether this was a streaming request.
        cached: Whether the response was cached.
        **extra: Additional fields to log.
    """
    logger = get_logger("generation")

    tokens_per_sec = (
        completion_tokens / (duration_ms / 1000) if duration_ms > 0 else 0
    )

    log_data = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "duration_ms": round(duration_ms, 2),
        "tokens_per_second": round(tokens_per_sec, 2),
        "stream": stream,
        "cached": cached,
    }

    log_data.update(extra)

    logger.info("Generation completed", **log_data)
