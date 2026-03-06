"""
Observability module for metrics, logging, and tracing.

Provides Prometheus metrics, structured logging with correlation IDs,
and utilities for monitoring the LLM inference service.
"""

from app.observability.logging import (
    configure_logging,
    get_logger,
    get_correlation_id,
    set_correlation_id,
)
from app.observability.metrics import (
    MetricsManager,
    get_metrics,
    setup_metrics,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "get_correlation_id",
    "set_correlation_id",
    "MetricsManager",
    "get_metrics",
    "setup_metrics",
]
