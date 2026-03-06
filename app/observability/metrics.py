"""
Prometheus Metrics for LLM Inference Service.

Provides comprehensive metrics for monitoring:
- Request counts and latencies
- Token generation rates
- Cache hit/miss rates
- Rate limiting statistics
- Engine health and status
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


class MetricsManager:
    """Manages Prometheus metrics for the LLM inference service."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Initialize metrics manager.

        Args:
            registry: Optional custom registry. Uses default if None.
        """
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Initialize all Prometheus metrics."""

        # Request metrics
        self.request_total = Counter(
            "llm_requests_total",
            "Total number of LLM requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )

        self.request_latency = Histogram(
            "llm_request_latency_seconds",
            "Request latency in seconds",
            ["endpoint", "method"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry,
        )

        self.request_in_progress = Gauge(
            "llm_requests_in_progress",
            "Number of requests currently being processed",
            ["endpoint"],
            registry=self.registry,
        )

        # Token metrics
        self.tokens_generated = Counter(
            "llm_tokens_generated_total",
            "Total number of tokens generated",
            ["model", "endpoint"],
            registry=self.registry,
        )

        self.prompt_tokens = Counter(
            "llm_prompt_tokens_total",
            "Total number of prompt tokens processed",
            ["model", "endpoint"],
            registry=self.registry,
        )

        self.tokens_per_second = Histogram(
            "llm_tokens_per_second",
            "Token generation rate (tokens/second)",
            ["model"],
            buckets=(1, 5, 10, 25, 50, 100, 200, 500),
            registry=self.registry,
        )

        # Generation metrics
        self.generation_latency = Histogram(
            "llm_generation_latency_seconds",
            "Time to generate response (excluding queue time)",
            ["model", "stream"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
            registry=self.registry,
        )

        self.time_to_first_token = Histogram(
            "llm_time_to_first_token_seconds",
            "Time to generate first token (streaming)",
            ["model"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits = Counter(
            "llm_cache_hits_total",
            "Total number of cache hits",
            registry=self.registry,
        )

        self.cache_misses = Counter(
            "llm_cache_misses_total",
            "Total number of cache misses",
            registry=self.registry,
        )

        self.cache_stores = Counter(
            "llm_cache_stores_total",
            "Total number of cache stores",
            registry=self.registry,
        )

        self.cache_errors = Counter(
            "llm_cache_errors_total",
            "Total number of cache errors",
            registry=self.registry,
        )

        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            "llm_rate_limit_hits_total",
            "Total number of rate limit rejections",
            ["client_ip"],
            registry=self.registry,
        )

        self.rate_limit_remaining = Gauge(
            "llm_rate_limit_remaining",
            "Remaining requests in current window",
            ["client_ip"],
            registry=self.registry,
        )

        # Engine metrics
        self.engine_ready = Gauge(
            "llm_engine_ready",
            "Whether the LLM engine is ready (1=ready, 0=not ready)",
            registry=self.registry,
        )

        self.engine_requests_queued = Gauge(
            "llm_engine_requests_queued",
            "Number of requests in the engine queue",
            registry=self.registry,
        )

        self.model_info = Gauge(
            "llm_model_info",
            "Model information",
            ["model_name", "version"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "llm_errors_total",
            "Total number of errors",
            ["endpoint", "error_type"],
            registry=self.registry,
        )

        self.timeout_total = Counter(
            "llm_timeouts_total",
            "Total number of timeout errors",
            ["endpoint"],
            registry=self.registry,
        )

        # Redis metrics
        self.redis_connected = Gauge(
            "llm_redis_connected",
            "Whether Redis is connected (1=connected, 0=disconnected)",
            registry=self.registry,
        )

        self.redis_operations = Counter(
            "llm_redis_operations_total",
            "Total Redis operations",
            ["operation", "status"],
            registry=self.registry,
        )

    @contextmanager
    def track_request(
        self,
        endpoint: str,
        method: str = "POST",
    ) -> Generator[None, None, None]:
        """Context manager to track request metrics.

        Args:
            endpoint: The API endpoint.
            method: HTTP method.

        Yields:
            None
        """
        self.request_in_progress.labels(endpoint=endpoint).inc()
        start_time = time.perf_counter()

        try:
            yield
            status = "success"
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start_time
            self.request_latency.labels(endpoint=endpoint, method=method).observe(duration)
            self.request_total.labels(endpoint=endpoint, method=method, status=status).inc()
            self.request_in_progress.labels(endpoint=endpoint).dec()

    def record_generation(
        self,
        model: str,
        endpoint: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
        stream: bool = False,
    ) -> None:
        """Record generation metrics.

        Args:
            model: Model name.
            endpoint: API endpoint.
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of generated tokens.
            duration_seconds: Generation duration.
            stream: Whether this was a streaming request.
        """
        self.prompt_tokens.labels(model=model, endpoint=endpoint).inc(prompt_tokens)
        self.tokens_generated.labels(model=model, endpoint=endpoint).inc(completion_tokens)
        self.generation_latency.labels(model=model, stream=str(stream)).observe(duration_seconds)

        if duration_seconds > 0 and completion_tokens > 0:
            tokens_per_sec = completion_tokens / duration_seconds
            self.tokens_per_second.labels(model=model).observe(tokens_per_sec)

    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits.inc()

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        self.cache_misses.inc()

    def record_cache_store(self) -> None:
        """Record a cache store."""
        self.cache_stores.inc()

    def record_cache_error(self) -> None:
        """Record a cache error."""
        self.cache_errors.inc()

    def record_rate_limit_hit(self, client_ip: str) -> None:
        """Record a rate limit rejection.

        Args:
            client_ip: The client IP that was rate limited.
        """
        self.rate_limit_hits.labels(client_ip=client_ip).inc()

    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record an error.

        Args:
            endpoint: The API endpoint.
            error_type: Type of error.
        """
        self.errors_total.labels(endpoint=endpoint, error_type=error_type).inc()

    def record_timeout(self, endpoint: str) -> None:
        """Record a timeout error.

        Args:
            endpoint: The API endpoint.
        """
        self.timeout_total.labels(endpoint=endpoint).inc()

    def set_engine_ready(self, ready: bool) -> None:
        """Set engine ready status.

        Args:
            ready: Whether engine is ready.
        """
        self.engine_ready.set(1 if ready else 0)

    def set_redis_connected(self, connected: bool) -> None:
        """Set Redis connection status.

        Args:
            connected: Whether Redis is connected.
        """
        self.redis_connected.set(1 if connected else 0)

    def set_model_info(self, model_name: str, version: str = "1.0") -> None:
        """Set model information.

        Args:
            model_name: Name of the loaded model.
            version: Model version.
        """
        self.model_info.labels(model_name=model_name, version=version).set(1)

    def generate_metrics(self) -> bytes:
        """Generate Prometheus metrics output.

        Returns:
            Prometheus metrics in text format.
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get the Prometheus content type.

        Returns:
            Content type string for Prometheus metrics.
        """
        return CONTENT_TYPE_LATEST


# Global metrics instance
_metrics_instance: MetricsManager | None = None


def get_metrics() -> MetricsManager:
    """Get the global metrics manager instance.

    Returns:
        The singleton MetricsManager instance.
    """
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = MetricsManager()

    return _metrics_instance


def setup_metrics() -> MetricsManager:
    """Initialize and return the global metrics manager.

    Returns:
        The initialized MetricsManager.
    """
    return get_metrics()
